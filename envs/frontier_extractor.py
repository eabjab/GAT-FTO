"""Frontier edge detection, connected component clustering, k-means splitting, medoids, normals, and EDT."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import distance_transform_edt, label

from envs.sensor import bresenham_line

if TYPE_CHECKING:
    from config import EnvConfig


@dataclass
class FrontierNode:
    """Topological graph node representation of an active frontier cluster."""

    position: tuple[int, int]          # (row, col) medoid in belief_map == 0
    orientation: float                 # Inward normal angle in radians [-pi, pi]
    cluster_size: int                  # Number of frontier cells in cluster
    obstacle_distance: float           # EDT clearance to nearest known wall
    unmapped_area_est: float           # Normalized [0, 1] visible unknown area
    cells: np.ndarray                  # Shape (M, 2) array of cluster cells


@functools.lru_cache(maxsize=16)
def _precompute_los_rays(sensor_range: int) -> list[tuple[float, np.ndarray]]:
    """Precompute relative Bresenham line-of-sight rays from (0, 0) to circle perimeter of radius `sensor_range`."""
    rays: list[tuple[float, np.ndarray]] = []
    r_max = int(sensor_range)
    if r_max <= 0:
        return rays

    r_max_sq = r_max * r_max
    for dr in range(-r_max, r_max + 1):
        for dc in range(-r_max, r_max + 1):
            dist = math.hypot(dr, dc)
            if (r_max - 1.0) < dist <= (r_max + 0.5):
                angle = math.atan2(dr, dc)
                raw_pts = bresenham_line(0, 0, dr, dc)
                valid_pts = [(pr, pc) for pr, pc in raw_pts if (pr * pr + pc * pc) <= r_max_sq]
                if valid_pts:
                    rays.append((angle, np.array(valid_pts, dtype=np.int32)))
    return rays


def compute_frontier_mask(belief_map: np.ndarray) -> np.ndarray:
    """Compute boolean mask of frontier cells (free cells with at least one 4-connected unknown neighbor)."""
    unknown = belief_map == -1
    has_unknown_neighbor = np.zeros_like(belief_map, dtype=bool)
    has_unknown_neighbor[:-1, :] |= unknown[1:, :]
    has_unknown_neighbor[1:, :] |= unknown[:-1, :]
    has_unknown_neighbor[:, :-1] |= unknown[:, 1:]
    has_unknown_neighbor[:, 1:] |= unknown[:, :-1]
    return (belief_map == 0) & has_unknown_neighbor


def compute_obstacle_clearance_map(belief_map: np.ndarray) -> np.ndarray:
    """Compute Euclidean Distance Transform (EDT) clearance to nearest known obstacle (`belief_map == 1`)."""
    return distance_transform_edt(belief_map != 1)


def _split_large_component(cells: np.ndarray, max_cluster_size: int) -> list[np.ndarray]:
    """Split a frontier component of size |C| > max_cluster_size into k = ceil(|C| / max_cluster_size) sub-clusters."""
    num_cells = len(cells)
    if num_cells <= max_cluster_size:
        return [cells]

    k = math.ceil(num_cells / max_cluster_size)
    cells_float = cells.astype(np.float64)

    # Order cells along principal spatial axis for deterministic, non-degenerate k-means initialization
    centered = cells_float - cells_float.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[0]
    order = np.lexsort((cells[:, 1], cells[:, 0], proj))
    ordered_cells = cells[order]
    ordered_float = cells_float[order]

    init_indices = np.floor((np.arange(k, dtype=np.float64) + 0.5) * num_cells / k).astype(int)
    init_indices = np.clip(init_indices, 0, num_cells - 1)
    init_centroids = ordered_float[init_indices].copy()

    centroids, labels = kmeans2(
        ordered_float,
        init_centroids,
        iter=20,
        minit="matrix",
        missing="warn",
    )

    # Guarantee each sub-cluster satisfies cluster_size <= max_cluster_size
    counts = np.bincount(labels, minlength=k).copy()
    while np.any(counts > max_cluster_size):
        over_clusters = np.where(counts > max_cluster_size)[0]
        under_clusters = np.where(counts < max_cluster_size)[0]
        if len(under_clusters) == 0:
            break

        c_over = int(over_clusters[0])
        member_indices = np.where(labels == c_over)[0]
        member_pts = ordered_float[member_indices]

        # Compute distances from members of oversized cluster to all non-full cluster centroids
        under_centroids = centroids[under_clusters]  # (U, 2)
        diff = member_pts[:, None, :] - under_centroids[None, :, :]  # (M, U, 2)
        dists_sq = np.sum(diff * diff, axis=-1)  # (M, U)

        best_flat = int(np.argmin(dists_sq))
        best_member_local, best_under_local = divmod(best_flat, len(under_clusters))
        target_idx = int(member_indices[best_member_local])
        c_under = int(under_clusters[best_under_local])

        labels[target_idx] = c_under
        counts[c_over] -= 1
        counts[c_under] += 1

    sub_clusters: list[np.ndarray] = []
    for c_idx in range(k):
        sub = ordered_cells[labels == c_idx]
        if len(sub) > 0:
            sub_clusters.append(sub)

    return sub_clusters


def _compute_inward_normal(belief_map: np.ndarray, r_i: int, c_i: int) -> float:
    """Compute canonical viewing orientation psi_i in [-pi, pi] using a 7x7 window around (r_i, c_i)."""
    height, width = belief_map.shape
    r_min = max(0, r_i - 3)
    r_max = min(height, r_i + 4)
    c_min = max(0, c_i - 3)
    c_max = min(width, c_i + 4)

    window = belief_map[r_min:r_max, c_min:c_max]
    free_coords = np.argwhere(window == 0)
    unknown_coords = np.argwhere(window == -1)

    if len(free_coords) == 0 or len(unknown_coords) == 0:
        return 0.0

    free_centroid = free_coords.mean(axis=0)
    unknown_centroid = unknown_coords.mean(axis=0)
    delta_r = float(unknown_centroid[0] - free_centroid[0])
    delta_c = float(unknown_centroid[1] - free_centroid[1])
    return float(math.atan2(delta_r, delta_c))


def _estimate_unmapped_area(
    belief_map: np.ndarray,
    r_i: int,
    c_i: int,
    psi_i: float,
    sensor_range: int,
    fov_deg: float,
) -> float:
    """Estimate normalized [0, 1] visible unknown area within sensor_range and FOV from (r_i, c_i)."""
    if sensor_range <= 0:
        return 0.0

    height, width = belief_map.shape
    rays = _precompute_los_rays(sensor_range)
    is_omni = fov_deg >= 360.0 - 1e-6
    half_fov = 0.5 * math.radians(fov_deg)

    local_dim = 2 * sensor_range + 1
    vis_unknown = np.zeros((local_dim, local_dim), dtype=bool)

    for ray_angle, rel_coords in rays:
        if not is_omni:
            diff = (ray_angle - psi_i + math.pi) % (2.0 * math.pi) - math.pi
            if abs(diff) > half_fov + 1e-9:
                continue

        for idx in range(1, len(rel_coords)):
            dr = int(rel_coords[idx, 0])
            dc = int(rel_coords[idx, 1])
            rr = r_i + dr
            cc = c_i + dc

            if rr < 0 or rr >= height or cc < 0 or cc >= width:
                break

            cell_val = belief_map[rr, cc]
            if cell_val == 1:
                break
            if cell_val == -1:
                vis_unknown[dr + sensor_range, dc + sensor_range] = True

    unknown_count = int(np.count_nonzero(vis_unknown))
    if is_omni:
        norm_area = unknown_count / (math.pi * float(sensor_range * sensor_range))
    else:
        fov_fraction = min(1.0, max(1e-6, fov_deg / 360.0))
        sector_area = fov_fraction * math.pi * float(sensor_range * sensor_range)
        norm_area = unknown_count / sector_area
    return float(min(1.0, max(0.0, norm_area)))


def extract_frontiers(
    belief_map: np.ndarray,
    min_frontier_size: int = 3,
    max_cluster_size: int = 20,
    sensor_range: int = 20,
    fov_deg: float = 360.0,
) -> list[FrontierNode]:
    """Extract active frontier nodes from the current occupancy belief map.

    Args:
        belief_map: 2D array of shape (H, W) with values {-1: Unknown, 0: Free, 1: Obstacle}.
        min_frontier_size: Minimum connected component size to retain.
        max_cluster_size: Maximum cluster size before k-means splitting.
        sensor_range: Sensor radius for unmapped area estimation.
        fov_deg: Sensor field of view in degrees.

    Returns:
        List of `FrontierNode` instances sorted deterministically by medoid position.
    """
    frontier_mask = compute_frontier_mask(belief_map)
    if not np.any(frontier_mask):
        return []

    structure = np.ones((3, 3), dtype=np.int8)
    labeled_array, num_features = label(frontier_mask, structure=structure)
    if num_features == 0:
        return []

    edt = compute_obstacle_clearance_map(belief_map)
    clusters: list[np.ndarray] = []

    for comp_id in range(1, num_features + 1):
        comp_cells = np.argwhere(labeled_array == comp_id)
        if len(comp_cells) < min_frontier_size:
            continue

        if len(comp_cells) > max_cluster_size:
            sub_clusters = _split_large_component(comp_cells, max_cluster_size)
            clusters.extend(sub_clusters)
        else:
            clusters.append(comp_cells)

    frontier_nodes: list[FrontierNode] = []
    for cluster_cells in clusters:
        if len(cluster_cells) == 0:
            continue

        centroid = cluster_cells.mean(axis=0)
        dists_sq = np.sum((cluster_cells - centroid) ** 2, axis=1)
        medoid_idx = int(np.argmin(dists_sq))
        r_i = int(cluster_cells[medoid_idx, 0])
        c_i = int(cluster_cells[medoid_idx, 1])

        psi_i = _compute_inward_normal(belief_map, r_i, c_i)
        obs_dist = float(edt[r_i, c_i])
        unmapped_est = _estimate_unmapped_area(
            belief_map=belief_map,
            r_i=r_i,
            c_i=c_i,
            psi_i=psi_i,
            sensor_range=sensor_range,
            fov_deg=fov_deg,
        )

        frontier_nodes.append(
            FrontierNode(
                position=(r_i, c_i),
                orientation=psi_i,
                cluster_size=int(len(cluster_cells)),
                obstacle_distance=obs_dist,
                unmapped_area_est=unmapped_est,
                cells=cluster_cells.astype(np.int64),
            )
        )

    frontier_nodes.sort(key=lambda node: node.position)
    return frontier_nodes


class FrontierExtractor:
    """Stateful wrapper for extracting `FrontierNode` clusters from occupancy belief maps."""

    def __init__(
        self,
        min_frontier_size: int | EnvConfig = 3,
        max_cluster_size: int = 20,
        sensor_range: int = 20,
        fov_deg: float = 360.0,
    ) -> None:
        if hasattr(min_frontier_size, "min_frontier_size"):
            cfg = min_frontier_size
            self.min_frontier_size = int(cfg.min_frontier_size)
            self.max_cluster_size = int(cfg.max_cluster_size)
            self.sensor_range = int(cfg.sensor_range)
            self.fov_deg = float(getattr(cfg, "sensor_fov_deg", fov_deg))
        else:
            self.min_frontier_size = int(min_frontier_size)
            self.max_cluster_size = int(max_cluster_size)
            self.sensor_range = int(sensor_range)
            self.fov_deg = float(fov_deg)

    def extract(self, belief_map: np.ndarray) -> list[FrontierNode]:
        """Extract frontier nodes from `belief_map`."""
        return extract_frontiers(
            belief_map=belief_map,
            min_frontier_size=self.min_frontier_size,
            max_cluster_size=self.max_cluster_size,
            sensor_range=self.sensor_range,
            fov_deg=self.fov_deg,
        )

    def extract_frontiers(self, belief_map: np.ndarray) -> list[FrontierNode]:
        """Alias for `extract(belief_map)`."""
        return self.extract(belief_map)

    def __call__(self, belief_map: np.ndarray) -> list[FrontierNode]:
        """Allow calling the extractor instance directly on `belief_map`."""
        return self.extract(belief_map)
