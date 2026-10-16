"""Automated verification suite for 2D raycasting sensor and frontier extraction (WP-3)."""

from __future__ import annotations

import math
import numpy as np

from envs.frontier_extractor import FrontierExtractor, FrontierNode, extract_frontiers
from envs.sensor import RaycastingSensor


def test_raycast_wall_occlusion() -> None:
    """In a synthetic 20x20 room bisected by an obstacle wall at col = 10, scanning from (10, 5) never uncovers any cells with col > 10."""
    ground_truth = np.zeros((20, 20), dtype=np.int8)
    # Enclose outer perimeter with obstacles
    ground_truth[0, :] = 1
    ground_truth[-1, :] = 1
    ground_truth[:, 0] = 1
    ground_truth[:, -1] = 1
    # Bisect room with vertical obstacle wall at col = 10
    ground_truth[:, 10] = 1

    belief_map = np.full((20, 20), -1, dtype=np.int8)
    sensor = RaycastingSensor(sensor_range=20, fov_deg=360.0)

    updated_belief, newly_uncovered = sensor.scan(ground_truth, belief_map, (10, 5, 0.0))

    assert updated_belief is belief_map
    assert newly_uncovered > 0
    # All free cells in the left half (rows 1..18, cols 1..9) should be uncovered (0)
    assert np.all(belief_map[1:19, 1:10] == 0)
    assert newly_uncovered == 18 * 9
    # The bisecting wall at col = 10 (rows 1..18) should be marked as obstacle (1)
    assert np.all(belief_map[1:19, 10] == 1)
    # No cells strictly behind the wall (col > 10) may be uncovered
    assert np.all(belief_map[:, 11:] == -1)

    # Re-scanning from the exact same pose should uncover 0 new free cells
    _, second_uncovered = sensor.scan(ground_truth, belief_map, (10, 5, 0.0))
    assert second_uncovered == 0


def test_frontier_medoid_in_free_space() -> None:
    """Across 20 random partial belief states, every extracted FrontierNode satisfies belief_map[node.position] == 0 and has at least one 4-connected neighbor with belief_map == -1."""
    sensor = RaycastingSensor(sensor_range=15, fov_deg=360.0)
    extractor = FrontierExtractor(min_frontier_size=3, max_cluster_size=20, sensor_range=15, fov_deg=360.0)

    total_nodes_checked = 0
    height, width = 50, 50

    for seed in range(20):
        rng = np.random.default_rng(seed + 100)
        ground_truth = np.zeros((height, width), dtype=np.int8)
        ground_truth[0, :] = 1
        ground_truth[-1, :] = 1
        ground_truth[:, 0] = 1
        ground_truth[:, -1] = 1

        # Add random interior obstacles
        for _ in range(6):
            r_obs = int(rng.integers(3, height - 8))
            c_obs = int(rng.integers(3, width - 8))
            h_obs = int(rng.integers(2, 6))
            w_obs = int(rng.integers(2, 6))
            ground_truth[r_obs : r_obs + h_obs, c_obs : c_obs + w_obs] = 1

        belief_map = np.full((height, width), -1, dtype=np.int8)

        # Perform 1 to 3 scans from random free cells to create a realistic partial belief state
        free_cells = np.argwhere(ground_truth == 0)
        num_scans = int(rng.integers(1, 4))
        for _ in range(num_scans):
            idx = int(rng.integers(0, len(free_cells)))
            r0, c0 = int(free_cells[idx, 0]), int(free_cells[idx, 1])
            sensor.scan(ground_truth, belief_map, (r0, c0, 0.0))

        nodes = extractor.extract(belief_map)
        assert len(nodes) > 0

        for node in nodes:
            assert isinstance(node, FrontierNode)
            r_m, c_m = node.position
            assert isinstance(r_m, int) and isinstance(c_m, int)
            # Medoid must lie in known free space
            assert belief_map[r_m, c_m] == 0

            # Must have at least one 4-connected neighbor with belief_map == -1
            neighbors_4 = []
            if r_m > 0:
                neighbors_4.append(belief_map[r_m - 1, c_m])
            if r_m + 1 < height:
                neighbors_4.append(belief_map[r_m + 1, c_m])
            if c_m > 0:
                neighbors_4.append(belief_map[r_m, c_m - 1])
            if c_m + 1 < width:
                neighbors_4.append(belief_map[r_m, c_m + 1])
            assert any(val == -1 for val in neighbors_4)

            # Verify node attributes
            assert -math.pi <= node.orientation <= math.pi
            assert 1 <= node.cluster_size <= 20
            assert node.cells.shape == (node.cluster_size, 2)
            assert node.obstacle_distance >= 0.0
            assert 0.0 <= node.unmapped_area_est <= 1.0
            total_nodes_checked += 1

    assert total_nodes_checked >= 20


def test_large_frontier_splitting() -> None:
    """A synthetic straight horizontal frontier of length 45 cells is automatically split into ceil(45 / 20) = 3 distinct FrontierNode instances, each with cluster_size <= 20."""
    belief_map = np.full((30, 60), -1, dtype=np.int8)
    # Enclose top and side borders of known region with obstacles so only the bottom horizontal edge is a frontier
    belief_map[0, 5:52] = 1
    belief_map[1:16, 7] = 1
    belief_map[1:16, 53] = 1
    # Known free region: rows 1..15, cols 8..52 (width = 45 cells)
    # Row 15 has unknown (-1) cells at row 16 directly below it -> 45 horizontal frontier cells
    belief_map[1:16, 8:53] = 0

    nodes = extract_frontiers(
        belief_map,
        min_frontier_size=3,
        max_cluster_size=20,
        sensor_range=20,
        fov_deg=360.0,
    )

    assert len(nodes) == 3
    total_cells = 0
    for node in nodes:
        assert node.cluster_size <= 20
        assert node.cluster_size >= 3
        assert node.position[0] == 15
        assert belief_map[node.position] == 0
        # Inward normal should point downward (+row direction -> angle near +pi/2)
        assert math.isclose(node.orientation, math.pi / 2.0, abs_tol=0.25)
        assert node.unmapped_area_est > 0.0
        total_cells += node.cluster_size

    assert total_cells == 45
