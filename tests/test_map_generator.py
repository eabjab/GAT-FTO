"""Unit tests for procedural BSP map generation and 8-connected A*/Dijkstra pathfinding."""

import math
import numpy as np
import scipy.ndimage

from envs.map_generator import generate_bsp_map
from envs.pathfinding import (
    astar_path,
    compute_geodesic_info,
    dijkstra_distances,
    dijkstra_with_angles,
)


def test_map_determinism_and_bounds() -> None:
    """Calling generate_bsp_map(100, 100, seed=42) twice returns bit-identical arrays and outer borders are 1."""
    map1 = generate_bsp_map(100, 100, seed=42)
    map2 = generate_bsp_map(100, 100, seed=42)

    assert map1.shape == (100, 100)
    assert map1.dtype == np.int8
    assert np.array_equal(map1, map2)

    # Outer borders (row 0, -1 and col 0, -1) must be strictly 1 (Obstacle)
    assert np.all(map1[0, :] == 1)
    assert np.all(map1[-1, :] == 1)
    assert np.all(map1[:, 0] == 1)
    assert np.all(map1[:, -1] == 1)

    # Values must only be 0 (Free) or 1 (Obstacle), with non-trivial free space
    unique_vals = set(np.unique(map1).tolist())
    assert unique_vals == {0, 1}


def test_single_connected_component() -> None:
    """Across 10 random seeds, 4-connected scipy.ndimage.label(ground_truth == 0) yields num_features == 1."""
    struct_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    for seed in range(10):
        ground_truth = generate_bsp_map(100, 100, seed=seed)
        _, num_features = scipy.ndimage.label(ground_truth == 0, structure=struct_4)
        assert num_features == 1, f"Expected 1 connected component for seed={seed}, got {num_features}"


def test_astar_and_dijkstra_consistency() -> None:
    """A* path length equals Multi-Source Dijkstra distance within 1e-5 and all path cells are free."""
    ground_truth = generate_bsp_map(100, 100, seed=123)
    free_coords = np.argwhere(ground_truth == 0)
    assert len(free_coords) >= 20

    rng = np.random.default_rng(123)
    sampled_indices = rng.choice(len(free_coords), size=10, replace=False)
    nodes = [(int(free_coords[idx, 0]), int(free_coords[idx, 1])) for idx in sampled_indices]

    # Multi-source Dijkstra distance maps and pairwise geodesic info
    dist_maps = dijkstra_distances(ground_truth, nodes, free_val=0)
    dist_matrix, dep_angles, arr_angles = compute_geodesic_info(ground_truth, nodes, free_val=0)

    assert dist_maps.shape == (len(nodes), 100, 100)
    assert dist_matrix.shape == (len(nodes), len(nodes))
    assert not np.isnan(dep_angles).any()
    assert not np.isnan(arr_angles).any()

    # Check A* consistency across pairs
    for i in range(len(nodes)):
        start = nodes[i]
        single_dist, single_dep, single_arr = dijkstra_with_angles(ground_truth, start, free_val=0)
        for j in range(i + 1, len(nodes)):
            goal = nodes[j]
            path, astar_cost = astar_path(ground_truth, start, goal, free_val=0)

            assert len(path) >= 2
            assert path[0] == start
            assert path[-1] == goal

            # Every cell along the returned A* path must have ground_truth == 0
            accumulated_cost = 0.0
            for k, (r, c) in enumerate(path):
                assert ground_truth[r, c] == 0
                if k > 0:
                    pr, pc = path[k - 1]
                    dr, dc = abs(r - pr), abs(c - pc)
                    assert dr <= 1 and dc <= 1 and (dr + dc) > 0
                    accumulated_cost += math.sqrt(2.0) if (dr == 1 and dc == 1) else 1.0

            assert abs(astar_cost - accumulated_cost) < 1e-5
            assert abs(astar_cost - dist_maps[i, goal[0], goal[1]]) < 1e-5
            assert abs(astar_cost - dist_matrix[i, j]) < 1e-5
            assert abs(astar_cost - single_dist[goal[0], goal[1]]) < 1e-5
            assert abs(dep_angles[i, j] - single_dep[goal[0], goal[1]]) < 1e-5
            assert abs(arr_angles[i, j] - single_arr[goal[0], goal[1]]) < 1e-5
