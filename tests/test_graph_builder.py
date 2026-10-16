"""Unit tests for dynamic topological graph construction (SPECS.md Section 4.3)."""

from __future__ import annotations

import math
import numpy as np
import torch

from envs.frontier_extractor import FrontierNode
from envs.graph_builder import GraphBuilder


def _make_synthetic_frontiers(count: int, height: int = 100, width: int = 100) -> list[FrontierNode]:
    """Create a deterministic list of `count` synthetic FrontierNode instances in free space."""
    rng = np.random.default_rng(12345)
    frontiers: list[FrontierNode] = []
    for idx in range(count):
        r = int(15 + (idx * 7) % (height - 30))
        c = int(15 + (idx * 11) % (width - 30))
        psi = float(rng.uniform(-math.pi, math.pi))
        cluster_size = int(3 + (idx * 4) % 25)
        obs_dist = float(1.0 + (idx * 3.5) % 25.0)
        unmapped_est = float(min(1.0, 0.1 + 0.12 * (idx % 8)))
        cells = np.array([[r, c], [r, min(width - 2, c + 1)], [min(height - 2, r + 1), c]], dtype=np.int64)
        frontiers.append(
            FrontierNode(
                position=(r, c),
                orientation=psi,
                cluster_size=cluster_size,
                obstacle_distance=obs_dist,
                unmapped_area_est=unmapped_est,
                cells=cells,
            )
        )
    return frontiers


def test_graph_tensor_shapes_and_bounds() -> None:
    """Verify SPECS.md Section 4.3 acceptance criteria for K_t = 6 frontiers."""
    height, width = 100, 100
    belief_map = np.zeros((height, width), dtype=np.int8)
    # Outer boundary obstacles
    belief_map[0, :] = 1
    belief_map[-1, :] = 1
    belief_map[:, 0] = 1
    belief_map[:, -1] = 1
    # Unknown region on right side
    belief_map[:, 85:-1] = -1

    frontiers = _make_synthetic_frontiers(6, height=height, width=width)
    agent_pose = (50, 50, 0.5)

    builder = GraphBuilder(sensor_range=20, max_cluster_size=20)
    data = builder.build_graph(agent_pose=agent_pose, frontiers=frontiers, belief_map=belief_map)

    # 1. Exact tensor shapes
    assert data.x.shape == (7, 8)
    assert data.agent_to_frontier_edge_attr.shape == (6, 4)
    # Complete directed graph without self-loops for N_t = 7 <= 16 -> E_t = 7 * 6 = 42
    assert data.edge_index.shape == (2, 42)
    assert data.edge_attr.shape == (42, 4)
    assert data.is_frontier.shape == (7,)
    assert data.node_positions.shape == (7, 2)

    # 2. NaN / Inf checks
    assert bool(torch.isnan(data.x).any()) is False
    assert bool(torch.isinf(data.x).any()) is False
    assert bool(torch.isnan(data.edge_attr).any()) is False
    assert bool(torch.isinf(data.edge_attr).any()) is False
    assert bool(torch.isnan(data.agent_to_frontier_edge_attr).any()) is False
    assert bool(torch.isinf(data.agent_to_frontier_edge_attr).any()) is False

    # 3. Node 0 reservation & is_agent / is_frontier flags
    assert float(data.x[0, 0].item()) == 1.0
    assert torch.all(data.x[1:, 0] == 0.0)
    assert bool(data.is_frontier[0].item()) is False
    assert bool(torch.all(data.is_frontier[1:]).item()) is True

    # 4. Feature value bounds
    assert bool(torch.all((data.x >= -1.0) & (data.x <= 1.0)).item()) is True
    assert bool(torch.all((data.edge_attr >= 0.0) & (data.edge_attr <= 2.0)).item()) is True
    assert bool(
        torch.all(
            (data.agent_to_frontier_edge_attr >= 0.0)
            & (data.agent_to_frontier_edge_attr <= 2.0)
        ).item()
    ) is True


def test_knn_pruning_for_large_graphs() -> None:
    """Verify k=12 nearest geodesic neighbor pruning when K_t + 1 > 16 while retaining all Node 0 edges."""
    height, width = 100, 100
    belief_map = np.zeros((height, width), dtype=np.int8)
    belief_map[0, :] = 1
    belief_map[-1, :] = 1
    belief_map[:, 0] = 1
    belief_map[:, -1] = 1

    k_t = 18  # N_t = 19 > 16
    frontiers = _make_synthetic_frontiers(k_t, height=height, width=width)
    agent_pose = (50, 50, -0.25)

    builder = GraphBuilder(sensor_range=20, max_cluster_size=20)
    data = builder.build_graph(agent_pose=agent_pose, frontiers=frontiers, belief_map=belief_map)

    assert data.x.shape == (k_t + 1, 8)
    assert data.agent_to_frontier_edge_attr.shape == (k_t, 4)

    # Expected edges:
    # - From Node 0 to all K_t frontiers: K_t
    # - From each of K_t frontiers: 1 edge to Node 0 + 12 edges to nearest frontier neighbors = 13 * K_t
    expected_edges = k_t + k_t * (1 + 12)
    assert data.edge_index.shape == (2, expected_edges)
    assert data.edge_attr.shape == (expected_edges, 4)

    src = data.edge_index[0].tolist()
    dst = data.edge_index[1].tolist()
    edge_set = set(zip(src, dst))

    # No self-loops
    for u in range(k_t + 1):
        assert (u, u) not in edge_set

    # All edges from and to Node 0 are retained
    for f_idx in range(1, k_t + 1):
        assert (0, f_idx) in edge_set
        assert (f_idx, 0) in edge_set

    # Each frontier node has exactly 12 outgoing frontier-to-frontier edges
    for f_idx in range(1, k_t + 1):
        f2f_neighbors = [v for u, v in edge_set if u == f_idx and v != 0]
        assert len(f2f_neighbors) == 12
