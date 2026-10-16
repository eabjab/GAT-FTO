"""Dynamic topological graph construction for GAT-based frontier exploration."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data

from envs.frontier_extractor import FrontierNode, compute_obstacle_clearance_map
from envs.pathfinding import compute_geodesic_info, wrap_angle

if TYPE_CHECKING:
    from config import EnvConfig


class GraphBuilder:
    """Constructs PyTorch Geometric `Data` objects from the agent pose, active frontiers, and belief map.

    Implements Section 3.2 of SPECS.md:
    - Node 0 is strictly reserved for the Agent Pose Node; Nodes 1..K_t are Frontier Nodes 0..K_t-1.
    - 8-dimensional node features bounded in [-1.0, 1.0].
    - 4-dimensional directed edge attributes bounded in [0.0, 2.0] using 8-connected Dijkstra.
    - Complete directed graph without self-loops when K_t + 1 <= 16; when K_t + 1 > 16,
      frontier-to-frontier edges are pruned to k=12 nearest geodesic neighbors while
      retaining all edges from and to Node 0.
    """

    def __init__(
        self,
        sensor_range: int | EnvConfig = 20,
        max_cluster_size: int = 20,
        knn_k: int = 12,
        knn_threshold: int = 16,
    ) -> None:
        if hasattr(sensor_range, "sensor_range"):
            cfg = sensor_range
            self.sensor_range = int(cfg.sensor_range)
            self.max_cluster_size = int(cfg.max_cluster_size)
        else:
            self.sensor_range = int(sensor_range)
            self.max_cluster_size = int(max_cluster_size)

        self.knn_k = int(knn_k)
        self.knn_threshold = int(knn_threshold)

    def build_graph(
        self,
        agent_pose: tuple[int, int, float],
        frontiers: list[FrontierNode],
        belief_map: np.ndarray,
    ) -> Data:
        """Construct a PyG `Data` topological graph for the current decision step.

        Args:
            agent_pose: Current agent pose `(row, col, theta_rad)`.
            frontiers: List of `K_t` extracted `FrontierNode` instances.
            belief_map: 2D occupancy belief grid of shape `(H, W)` with values `{-1, 0, 1}`.

        Returns:
            PyG `Data` instance populated with `x`, `edge_index`, `edge_attr`,
            `agent_to_frontier_edge_attr`, `is_frontier`, and `node_positions`.
        """
        height, width = belief_map.shape
        d_diag = math.hypot(float(height), float(width))
        r_max = float(max(1, self.sensor_range))
        max_cluster = float(max(1, self.max_cluster_size))

        r_0 = int(agent_pose[0])
        c_0 = int(agent_pose[1])
        theta_0 = float(agent_pose[2])

        k_t = len(frontiers)
        n_t = k_t + 1

        # Compute EDT obstacle clearance map for agent node (and fallback for frontiers)
        edt = compute_obstacle_clearance_map(belief_map)
        if 0 <= r_0 < height and 0 <= c_0 < width:
            d_obs_0 = float(edt[r_0, c_0])
        else:
            d_obs_0 = 0.0
        obs_clearance_0 = float(min(1.0, max(0.0, d_obs_0 / r_max)))

        # 1. Node Feature Matrix X in R^{(K_t + 1) x 8}
        x_np = np.zeros((n_t, 8), dtype=np.float32)
        # Node 0: Agent Pose Node
        x_np[0, 0] = 1.0
        x_np[0, 1] = 0.0
        x_np[0, 2] = 0.0
        x_np[0, 3] = float(math.cos(theta_0))
        x_np[0, 4] = float(math.sin(theta_0))
        x_np[0, 5] = 0.0
        x_np[0, 6] = obs_clearance_0
        x_np[0, 7] = 0.0

        nodes_rc: list[tuple[int, int]] = [(r_0, c_0)]
        orientations: list[float] = [theta_0]

        for idx, f in enumerate(frontiers, start=1):
            r_i, c_i = int(f.position[0]), int(f.position[1])
            psi_i = float(f.orientation)
            nodes_rc.append((r_i, c_i))
            orientations.append(psi_i)

            rel_row = float(np.clip((r_i - r_0) / float(height), -1.0, 1.0))
            rel_col = float(np.clip((c_i - c_0) / float(width), -1.0, 1.0))
            cos_orient = float(math.cos(psi_i))
            sin_orient = float(math.sin(psi_i))
            cluster_norm = float(min(1.0, max(0.0, float(f.cluster_size) / max_cluster)))

            f_obs = float(f.obstacle_distance)
            if not math.isfinite(f_obs) or f_obs < 0.0:
                if 0 <= r_i < height and 0 <= c_i < width:
                    f_obs = float(edt[r_i, c_i])
                else:
                    f_obs = 0.0
            obs_clearance_i = float(min(1.0, max(0.0, f_obs / r_max)))
            unmapped_est_i = float(min(1.0, max(0.0, float(f.unmapped_area_est))))

            x_np[idx, 0] = 0.0
            x_np[idx, 1] = rel_row
            x_np[idx, 2] = rel_col
            x_np[idx, 3] = cos_orient
            x_np[idx, 4] = sin_orient
            x_np[idx, 5] = cluster_norm
            x_np[idx, 6] = obs_clearance_i
            x_np[idx, 7] = unmapped_est_i

        # Ensure strict [-1.0, 1.0] bounds on node features
        np.clip(x_np, -1.0, 1.0, out=x_np)

        # 2. Geodesic path computation via 8-connected Dijkstra over belief_map == 0
        dist_matrix, dep_angle_matrix, arr_angle_matrix = compute_geodesic_info(
            belief_map, nodes_rc, free_val=0
        )

        def _compute_edge_feat(u: int, v: int) -> tuple[float, float, float, float]:
            d_geo = float(dist_matrix[u, v])
            if math.isnan(d_geo) or d_geo < 0.0:
                geo_dist = 2.0
            elif math.isinf(d_geo):
                geo_dist = 2.0
            else:
                geo_dist = float(min(2.0, max(0.0, d_geo / d_diag)))

            ru, cu = nodes_rc[u]
            rv, cv = nodes_rc[v]
            euc_dist = float(min(1.0, max(0.0, math.hypot(rv - ru, cv - cu) / d_diag)))

            phi_start = float(dep_angle_matrix[u, v])
            phi_end = float(arr_angle_matrix[u, v])
            psi_u = orientations[u]
            psi_v = orientations[v]

            dep_angle = float(min(1.0, max(0.0, abs(float(wrap_angle(phi_start - psi_u))) / math.pi)))
            arr_angle = float(min(1.0, max(0.0, abs(float(wrap_angle(psi_v - phi_end))) / math.pi)))
            return geo_dist, euc_dist, dep_angle, arr_angle

        # 3. Direct agent-to-frontier edge attributes (shape: (K_t, 4))
        agent_to_frontier_np = np.zeros((k_t, 4), dtype=np.float32)
        for m in range(k_t):
            agent_to_frontier_np[m] = _compute_edge_feat(0, m + 1)

        # 4. Graph topology & edge attributes
        src_list: list[int] = []
        dst_list: list[int] = []
        edge_attr_list: list[tuple[float, float, float, float]] = []

        if n_t <= self.knn_threshold:
            # Complete directed graph without self-loops
            for u in range(n_t):
                for v in range(n_t):
                    if u == v:
                        continue
                    src_list.append(u)
                    dst_list.append(v)
                    edge_attr_list.append(_compute_edge_feat(u, v))
        else:
            # Retain all outgoing edges from Node 0 to all frontiers 1..K_t
            for v in range(1, n_t):
                src_list.append(0)
                dst_list.append(v)
                edge_attr_list.append(_compute_edge_feat(0, v))

            # For each frontier node u in 1..K_t, retain edge to Node 0 and k=12 nearest geodesic frontier neighbors
            for u in range(1, n_t):
                src_list.append(u)
                dst_list.append(0)
                edge_attr_list.append(_compute_edge_feat(u, 0))

                ru, cu = nodes_rc[u]
                candidates: list[tuple[float, float, int]] = []
                for v in range(1, n_t):
                    if u == v:
                        continue
                    rv, cv = nodes_rc[v]
                    d_g = float(dist_matrix[u, v])
                    d_e = math.hypot(rv - ru, cv - cu)
                    candidates.append((d_g, d_e, v))

                candidates.sort(key=lambda item: (item[0], item[1], item[2]))
                selected_neighbors = [item[2] for item in candidates[: self.knn_k]]
                selected_neighbors.sort()

                for v in selected_neighbors:
                    src_list.append(u)
                    dst_list.append(v)
                    edge_attr_list.append(_compute_edge_feat(u, v))

        if len(src_list) > 0:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr_np = np.array(edge_attr_list, dtype=np.float32)
            np.clip(edge_attr_np, 0.0, 2.0, out=edge_attr_np)
            edge_attr = torch.from_numpy(edge_attr_np)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float32)

        is_frontier = torch.zeros(n_t, dtype=torch.bool)
        if k_t > 0:
            is_frontier[1:] = True

        node_positions = torch.tensor(nodes_rc, dtype=torch.float32)

        return Data(
            x=torch.from_numpy(x_np),
            edge_index=edge_index,
            edge_attr=edge_attr,
            agent_to_frontier_edge_attr=torch.from_numpy(agent_to_frontier_np),
            is_frontier=is_frontier,
            node_positions=node_positions,
        )

    def __call__(
        self,
        agent_pose: tuple[int, int, float],
        frontiers: list[FrontierNode],
        belief_map: np.ndarray,
    ) -> Data:
        """Alias for `build_graph(agent_pose, frontiers, belief_map)`."""
        return self.build_graph(agent_pose, frontiers, belief_map)


def build_graph(
    agent_pose: tuple[int, int, float],
    frontiers: list[FrontierNode],
    belief_map: np.ndarray,
    sensor_range: int = 20,
    max_cluster_size: int = 20,
) -> Data:
    """Functional helper to build a topological frontier graph."""
    builder = GraphBuilder(sensor_range=sensor_range, max_cluster_size=max_cluster_size)
    return builder.build_graph(agent_pose, frontiers, belief_map)
