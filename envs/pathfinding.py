"""8-connected A* path planner and Multi-Source Dijkstra geodesic computation on 2D grids."""

from __future__ import annotations

import heapq
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra

SQRT2: float = math.sqrt(2.0)

# 8-connected neighbor offsets: (dr, dc, step_cost)
NEIGHBORS_8: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, SQRT2),
    (-1, 1, SQRT2),
    (1, -1, SQRT2),
    (1, 1, SQRT2),
)


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) in radians to [-pi, pi]."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if isinstance(angle, (int, float, np.floating)):
        return float(wrapped)
    return wrapped


def astar_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    free_val: int = 0,
) -> tuple[list[tuple[int, int]], float]:
    """Compute shortest 8-connected path between start and goal over grid == free_val.

    Args:
        grid: 2D occupancy grid of shape (H, W).
        start: Starting cell (row, col).
        goal: Target cell (row, col).
        free_val: Grid cell value considered traversable (default 0).

    Returns:
        (path, cost):
            path: List of (row, col) tuples from start to goal inclusive, or [] if unreachable.
            cost: Total 8-connected Euclidean path length, or float('inf') if unreachable.
    """
    height, width = grid.shape
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])

    if not (0 <= sr < height and 0 <= sc < width and 0 <= gr < height and 0 <= gc < width):
        return [], float("inf")

    if grid[sr, sc] != free_val or grid[gr, gc] != free_val:
        return [], float("inf")

    if sr == gr and sc == gc:
        return [(sr, sc)], 0.0

    g_score = np.full((height, width), np.inf, dtype=np.float64)
    parent_r = np.full((height, width), -1, dtype=np.int32)
    parent_c = np.full((height, width), -1, dtype=np.int32)
    closed = np.zeros((height, width), dtype=bool)

    g_score[sr, sc] = 0.0
    dr0 = abs(sr - gr)
    dc0 = abs(sc - gc)
    h0 = (dr0 + dc0) + (SQRT2 - 2.0) * min(dr0, dc0)

    open_heap: list[tuple[float, float, int, int]] = [(h0, 0.0, sr, sc)]

    while open_heap:
        _, g_curr, r, c = heapq.heappop(open_heap)
        if closed[r, c]:
            continue
        if r == gr and c == gc:
            # Reconstruct path
            path: list[tuple[int, int]] = []
            cr, cc = gr, gc
            while not (cr == sr and cc == sc):
                path.append((int(cr), int(cc)))
                pr, pc = int(parent_r[cr, cc]), int(parent_c[cr, cc])
                cr, cc = pr, pc
            path.append((sr, sc))
            path.reverse()
            return path, float(g_curr)

        closed[r, c] = True

        for dr, dc, step_cost in NEIGHBORS_8:
            if dr != 0 and dc != 0 and (grid[r + dr, c] != free_val or grid[r, c + dc] != free_val):
                continue    # prevent corner cutting
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and not closed[nr, nc]:
                if grid[nr, nc] != free_val:
                    continue
                tentative_g = g_curr + step_cost
                if tentative_g < g_score[nr, nc]:
                    g_score[nr, nc] = tentative_g
                    parent_r[nr, nc] = r
                    parent_c[nr, nc] = c
                    dr_goal = abs(nr - gr)
                    dc_goal = abs(nc - gc)
                    h = (dr_goal + dc_goal) + (SQRT2 - 2.0) * min(dr_goal, dc_goal)
                    heapq.heappush(open_heap, (tentative_g + h, tentative_g, nr, nc))

    return [], float("inf")


# Alias for convenience
astar = astar_path


def _build_grid_csr(grid: np.ndarray, free_val: int = 0) -> csr_matrix:
    """Build an undirected 8-connected sparse adjacency matrix over grid == free_val."""
    height, width = grid.shape
    num_cells = height * width
    node_ids = np.arange(num_cells, dtype=np.int32).reshape(height, width)
    free_mask = grid == free_val

    u_list: list[np.ndarray] = []
    v_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []

    # 1. Vertical edges (dr=1, dc=0), weight=1.0
    mask_v = free_mask[:-1, :] & free_mask[1:, :]
    u_v = node_ids[:-1, :][mask_v]
    v_v = node_ids[1:, :][mask_v]
    u_list.append(u_v)
    v_list.append(v_v)
    w_list.append(np.ones(u_v.size, dtype=np.float64))

    # 2. Horizontal edges (dr=0, dc=1), weight=1.0
    mask_h = free_mask[:, :-1] & free_mask[:, 1:]
    u_h = node_ids[:, :-1][mask_h]
    v_h = node_ids[:, 1:][mask_h]
    u_list.append(u_h)
    v_list.append(v_h)
    w_list.append(np.ones(u_h.size, dtype=np.float64))

    # 3. Diagonal down-right (dr=1, dc=1), weight=sqrt(2)
    mask_d1 = (
            free_mask[:-1, :-1] & free_mask[1:, 1:]
            & free_mask[1:, :-1] & free_mask[:-1, 1:]   # prevent corner cutting
    )
    u_d1 = node_ids[:-1, :-1][mask_d1]
    v_d1 = node_ids[1:, 1:][mask_d1]
    u_list.append(u_d1)
    v_list.append(v_d1)
    w_list.append(np.full(u_d1.size, SQRT2, dtype=np.float64))

    # 4. Diagonal down-left (dr=1, dc=-1), weight=sqrt(2)
    mask_d2 = (
            free_mask[:-1, 1:] & free_mask[1:, :-1]
            & free_mask[:-1, :-1] & free_mask[1:, 1:]   # prevent corner cutting
    )
    u_d2 = node_ids[:-1, 1:][mask_d2]
    v_d2 = node_ids[1:, :-1][mask_d2]
    u_list.append(u_d2)
    v_list.append(v_d2)
    w_list.append(np.full(u_d2.size, SQRT2, dtype=np.float64))

    u_all = np.concatenate(u_list)
    v_all = np.concatenate(v_list)
    w_all = np.concatenate(w_list)

    # Emit symmetric edges for CSR graph
    rows = np.concatenate([u_all, v_all])
    cols = np.concatenate([v_all, u_all])
    data = np.concatenate([w_all, w_all])

    return csr_matrix((data, (rows, cols)), shape=(num_cells, num_cells))


def dijkstra_distances(
    grid: np.ndarray,
    sources: tuple[int, int] | list[tuple[int, int]],
    free_val: int = 0,
) -> np.ndarray:
    """Compute 8-connected Dijkstra shortest-path distances from one or multiple sources.

    Args:
        grid: 2D array of shape (H, W).
        sources: Single (row, col) tuple or list of (row, col) tuples.
        free_val: Traversable cell value (default 0).

    Returns:
        If sources is a single (row, col) tuple:
            dist_map of shape (H, W) with dtype=np.float64.
        If sources is a list of N (row, col) tuples:
            dist_maps of shape (N, H, W) with dtype=np.float64.
    """
    height, width = grid.shape
    csr = _build_grid_csr(grid, free_val=free_val)

    single_source = (
        isinstance(sources, tuple)
        and len(sources) == 2
        and isinstance(sources[0], (int, np.integer))
    )
    if single_source:
        sr, sc = int(sources[0]), int(sources[1])
        if not (0 <= sr < height and 0 <= sc < width) or grid[sr, sc] != free_val:
            return np.full((height, width), np.inf, dtype=np.float64)
        flat_src = sr * width + sc
        dist_flat = csgraph_dijkstra(csr, directed=False, indices=flat_src)
        return dist_flat.reshape(height, width)

    source_list = list(sources)  # type: ignore[arg-type]
    num_sources = len(source_list)
    if num_sources == 0:
        return np.empty((0, height, width), dtype=np.float64)

    flat_indices: list[int] = []
    valid_mask: list[bool] = []
    for r, c in source_list:
        ri, ci = int(r), int(c)
        if 0 <= ri < height and 0 <= ci < width and grid[ri, ci] == free_val:
            flat_indices.append(ri * width + ci)
            valid_mask.append(True)
        else:
            flat_indices.append(0)
            valid_mask.append(False)

    dist_flat = csgraph_dijkstra(csr, directed=False, indices=flat_indices)
    dist_maps = dist_flat.reshape(num_sources, height, width)
    for i, is_valid in enumerate(valid_mask):
        if not is_valid:
            dist_maps[i].fill(np.inf)
    return dist_maps


# Alias for multi-source Dijkstra
multi_source_dijkstra = dijkstra_distances


def dijkstra_with_angles(
    grid: np.ndarray,
    source: tuple[int, int],
    free_val: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run single-source 8-connected Dijkstra returning distance, departure, and arrival angle maps.

    Args:
        grid: 2D array of shape (H, W).
        source: Source cell (row, col).
        free_val: Traversable cell value (default 0).

    Returns:
        (dist_map, dep_angle_map, arr_angle_map), each of shape (H, W) with dtype=np.float64.
    """
    height, width = grid.shape
    sr, sc = int(source[0]), int(source[1])
    dist_map = np.full((height, width), np.inf, dtype=np.float64)
    dep_angle_map = np.zeros((height, width), dtype=np.float64)
    arr_angle_map = np.zeros((height, width), dtype=np.float64)

    if not (0 <= sr < height and 0 <= sc < width) or grid[sr, sc] != free_val:
        return dist_map, dep_angle_map, arr_angle_map

    csr = _build_grid_csr(grid, free_val=free_val)
    flat_src = sr * width + sc
    dist_flat, pred_flat = csgraph_dijkstra(
        csr, directed=False, indices=flat_src, return_predecessors=True
    )

    dist_map = dist_flat.reshape(height, width)
    dep_flat = np.zeros(height * width, dtype=np.float64)
    arr_flat = np.zeros(height * width, dtype=np.float64)

    reachable = np.flatnonzero((pred_flat >= 0) & (np.arange(height * width) != flat_src))
    if reachable.size > 0:
        # Vectorized arrival angle from immediate predecessor
        preds = pred_flat[reachable]
        r_u, c_u = np.divmod(reachable, width)
        r_p, c_p = np.divmod(preds, width)
        arr_flat[reachable] = np.arctan2(r_u - r_p, c_u - c_p)

        # Propagate departure angle along shortest-path tree in ascending distance order
        order = reachable[np.argsort(dist_flat[reachable])]
        for u in order:
            p = int(pred_flat[u])
            if p == flat_src:
                dep_flat[u] = arr_flat[u]
            else:
                dep_flat[u] = dep_flat[p]

    return (
        dist_map,
        dep_flat.reshape(height, width),
        arr_flat.reshape(height, width),
    )


def compute_geodesic_info(
    grid: np.ndarray,
    nodes: list[tuple[int, int]],
    free_val: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise geodesic shortest-path distances, departure angles, and arrival angles.

    Runs 8-connected Multi-Source Dijkstra from each node position p_i in `nodes` over
    `grid == free_val` to obtain exact shortest-path distance d_geo(i, j), initial
    departure path angle phi_start^(i -> j), and final arrival path angle phi_end^(i -> j).

    Args:
        grid: 2D occupancy belief or ground-truth map of shape (H, W).
        nodes: List of N node positions [(r_0, c_0), ..., (r_{N-1}, c_{N-1})].
        free_val: Grid value considered traversable (default 0).

    Returns:
        (dist_matrix, dep_angle_matrix, arr_angle_matrix):
            dist_matrix: Shape (N, N) float64 array of geodesic distances d_geo(i, j).
            dep_angle_matrix: Shape (N, N) float64 array of initial departure angles in [-pi, pi].
            arr_angle_matrix: Shape (N, N) float64 array of final arrival angles in [-pi, pi].
    """
    num_nodes = len(nodes)
    if num_nodes == 0:
        empty = np.empty((0, 0), dtype=np.float64)
        return empty, empty, empty

    height, width = grid.shape
    csr = _build_grid_csr(grid, free_val=free_val)

    flat_nodes = [int(r) * width + int(c) for r, c in nodes]
    valid_nodes = [
        0 <= int(r) < height and 0 <= int(c) < width and grid[int(r), int(c)] == free_val
        for r, c in nodes
    ]
    query_indices = [idx if valid else 0 for idx, valid in zip(flat_nodes, valid_nodes)]

    dist_raw, pred_raw = csgraph_dijkstra(
        csr, directed=False, indices=query_indices, return_predecessors=True
    )

    dist_matrix = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
    dep_angle_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    arr_angle_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    for i in range(num_nodes):
        ri, ci = int(nodes[i][0]), int(nodes[i][1])
        flat_i = flat_nodes[i]
        dist_matrix[i, i] = 0.0
        dep_angle_matrix[i, i] = 0.0
        arr_angle_matrix[i, i] = 0.0

        if not valid_nodes[i]:
            for j in range(num_nodes):
                if i != j:
                    rj, cj = int(nodes[j][0]), int(nodes[j][1])
                    fallback = math.atan2(rj - ri, cj - ci)
                    dep_angle_matrix[i, j] = fallback
                    arr_angle_matrix[i, j] = fallback
            continue

        preds_i = pred_raw[i]
        dists_i = dist_raw[i]

        for j in range(num_nodes):
            if i == j:
                continue
            rj, cj = int(nodes[j][0]), int(nodes[j][1])
            flat_j = flat_nodes[j]

            if flat_i == flat_j:
                dist_matrix[i, j] = 0.0
                dep_angle_matrix[i, j] = 0.0
                arr_angle_matrix[i, j] = 0.0
                continue

            d_ij = float(dists_i[flat_j])
            pred_j = int(preds_i[flat_j])

            if valid_nodes[j] and math.isfinite(d_ij) and pred_j >= 0:
                dist_matrix[i, j] = d_ij

                # Final arrival step: pred_j -> flat_j
                r_prev, c_prev = divmod(pred_j, width)
                arr_angle_matrix[i, j] = math.atan2(rj - r_prev, cj - c_prev)

                # Initial departure step: trace back from flat_j until predecessor is flat_i
                curr = flat_j
                while int(preds_i[curr]) != flat_i and int(preds_i[curr]) >= 0:
                    curr = int(preds_i[curr])
                r_first, c_first = divmod(curr, width)
                dep_angle_matrix[i, j] = math.atan2(r_first - ri, c_first - ci)
            else:
                dist_matrix[i, j] = float("inf")
                fallback = math.atan2(rj - ri, cj - ci)
                dep_angle_matrix[i, j] = fallback
                arr_angle_matrix[i, j] = fallback

    return dist_matrix, dep_angle_matrix, arr_angle_matrix
