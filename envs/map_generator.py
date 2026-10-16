"""Procedural 2D indoor occupancy grid generator via Binary Space Partitioning (BSP)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.ndimage


@dataclass
class _BSPNode:
    """Internal representation of a rectangular BSP partition."""

    r0: int
    r1: int
    c0: int
    c1: int
    split_axis: str | None = None  # 'H' (horizontal wall) or 'V' (vertical wall)
    split_idx: int | None = None
    left: _BSPNode | None = None
    right: _BSPNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _partition_bsp(
    node: _BSPNode,
    depth: int,
    max_depth: int,
    min_leaf: int,
    rng: np.random.Generator,
) -> None:
    """Recursively partition a BSP node up to max_depth with minimum leaf dimensions."""
    if depth >= max_depth:
        return

    height = node.r1 - node.r0
    width = node.c1 - node.c0

    can_split_h = height >= 2 * min_leaf
    can_split_v = width >= 2 * min_leaf

    if not can_split_h and not can_split_v:
        return

    if can_split_h and can_split_v:
        if height > 1.25 * width:
            axis = "H"
        elif width > 1.25 * height:
            axis = "V"
        else:
            axis = "H" if rng.random() < 0.5 else "V"
    elif can_split_h:
        axis = "H"
    else:
        axis = "V"

    node.split_axis = axis
    if axis == "H":
        low = node.r0 + min_leaf
        high = node.r1 - min_leaf
        split_r = int(rng.integers(low, high + 1))
        node.split_idx = split_r
        node.left = _BSPNode(node.r0, split_r, node.c0, node.c1)
        node.right = _BSPNode(split_r, node.r1, node.c0, node.c1)
    else:
        low = node.c0 + min_leaf
        high = node.c1 - min_leaf
        split_c = int(rng.integers(low, high + 1))
        node.split_idx = split_c
        node.left = _BSPNode(node.r0, node.r1, node.c0, split_c)
        node.right = _BSPNode(node.r0, node.r1, split_c, node.c1)

    _partition_bsp(node.left, depth + 1, max_depth, min_leaf, rng)
    _partition_bsp(node.right, depth + 1, max_depth, min_leaf, rng)


def _build_walls(node: _BSPNode, grid: np.ndarray) -> None:
    """Draw thickness-1 walls along all internal partition lines."""
    if node.is_leaf or node.split_axis is None or node.split_idx is None:
        return

    if node.split_axis == "H":
        grid[node.split_idx, node.c0 : node.c1 + 1] = 1
    else:
        grid[node.r0 : node.r1 + 1, node.split_idx] = 1

    if node.left is not None:
        _build_walls(node.left, grid)
    if node.right is not None:
        _build_walls(node.right, grid)


def _carve_doorways(node: _BSPNode, grid: np.ndarray, rng: np.random.Generator) -> None:
    """Carve doorway openings of width 4..6 between sibling partitions (post-order)."""
    if node.is_leaf or node.split_axis is None or node.split_idx is None:
        return

    if node.left is not None:
        _carve_doorways(node.left, grid, rng)
    if node.right is not None:
        _carve_doorways(node.right, grid, rng)

    door_width = int(rng.integers(4, 7))  # 4 to 6 cells inclusive

    if node.split_axis == "H":
        r = node.split_idx
        span_start = node.c0 + 1
        span_end = node.c1 - 1
        avail = span_end - span_start + 1
        w = min(door_width, max(1, avail))
        max_start = span_end - w + 1

        # Prefer doorway placements that do not intersect perpendicular child walls
        candidates = [
            c
            for c in range(span_start, max_start + 1)
            if np.all(grid[r - 1, c : c + w] == 0) and np.all(grid[r + 1, c : c + w] == 0)
        ]
        if candidates:
            c_start = int( candidates[int(rng.integers(0, len(candidates)))] )
        else:
            c_start = int(rng.integers(span_start, max_start + 1))

        grid[r, c_start : c_start + w] = 0
        if r - 1 > 0:
            grid[r - 1, c_start : c_start + w] = 0
        if r + 1 < grid.shape[0] - 1:
            grid[r + 1, c_start : c_start + w] = 0
    else:
        c = node.split_idx
        span_start = node.r0 + 1
        span_end = node.r1 - 1
        avail = span_end - span_start + 1
        w = min(door_width, max(1, avail))
        max_start = span_end - w + 1

        candidates = [
            r
            for r in range(span_start, max_start + 1)
            if np.all(grid[r : r + w, c - 1] == 0) and np.all(grid[r : r + w, c + 1] == 0)
        ]
        if candidates:
            r_start = int( candidates[int(rng.integers(0, len(candidates)))] )
        else:
            r_start = int(rng.integers(span_start, max_start + 1))

        grid[r_start : r_start + w, c] = 0
        if c - 1 > 0:
            grid[r_start : r_start + w, c - 1] = 0
        if c + 1 < grid.shape[1] - 1:
            grid[r_start : r_start + w, c + 1] = 0


def _collect_leaves(node: _BSPNode) -> list[_BSPNode]:
    """Return all leaf nodes of the BSP tree."""
    if node.is_leaf:
        return [node]
    leaves: list[_BSPNode] = []
    if node.left is not None:
        leaves.extend(_collect_leaves(node.left))
    if node.right is not None:
        leaves.extend(_collect_leaves(node.right))
    return leaves


def generate_bsp_map(
    height: int = 100,
    width: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a 2D indoor occupancy grid using Binary Space Partitioning (BSP).

    Args:
        height: Grid height H (default 100).
        width: Grid width W (default 100).
        seed: Optional integer seed for deterministic generation.

    Returns:
        ground_truth: np.ndarray of shape (height, width) with dtype=np.int8
            where 0 = Free and 1 = Obstacle. Guaranteed to have a single
            4-connected free-space component and closed outer perimeter.
    """
    if height < 10 or width < 10:
        raise ValueError(f"Grid dimensions must be at least 10x10, got ({height}, {width}).")

    rng = np.random.default_rng(seed)
    grid = np.zeros((height, width), dtype=np.int8)

    # 1. Enclose the outer perimeter with obstacles (1)
    grid[0, :] = 1
    grid[height - 1, :] = 1
    grid[:, 0] = 1
    grid[:, width - 1] = 1

    # 2. Recursively partition interior space via BSP to depth d in {3, 4}
    # with minimum leaf dimensions of 18x18 (scaled down only if grid < 38x38)
    max_depth = int(rng.integers(3, 5))
    min_leaf = min(18, max(4, min(height - 1, width - 1) // 2))
    root = _BSPNode(0, height - 1, 0, width - 1)
    _partition_bsp(root, depth=0, max_depth=max_depth, min_leaf=min_leaf, rng=rng)

    # 3. Build walls of thickness 1 along partition lines and carve 4..6-cell doorways
    _build_walls(root, grid)
    _carve_doorways(root, grid, rng)

    # 4. Scatter 3 to 8 random rectangular obstacles (2x2 to 5x5) inside rooms
    leaves = _collect_leaves(root)
    num_obstacles = int(rng.integers(3, 9))
    for _ in range(num_obstacles):
        leaf = leaves[int(rng.integers(0, len(leaves)))]
        obs_h = int(rng.integers(2, 6))
        obs_w = int(rng.integers(2, 6))

        # Keep a 2-cell margin from room walls so doorways remain unobstructed
        r_min = leaf.r0 + 2
        r_max = leaf.r1 - 1 - obs_h
        c_min = leaf.c0 + 2
        c_max = leaf.c1 - 1 - obs_w

        if r_max >= r_min and c_max >= c_min:
            r_pos = int(rng.integers(r_min, r_max + 1))
            c_pos = int(rng.integers(c_min, c_max + 1))
            grid[r_pos : r_pos + obs_h, c_pos : c_pos + obs_w] = 1

    # Re-enforce outer perimeter obstacles
    grid[0, :] = 1
    grid[height - 1, :] = 1
    grid[:, 0] = 1
    grid[:, width - 1] = 1

    # 5. Connectivity Invariant: retain only the largest 4-connected free component
    struct_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    labeled, num_features = scipy.ndimage.label(grid == 0, structure=struct_4)
    if num_features > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0  # Ignore background (obstacle label 0)
        largest_label = int(np.argmax(counts))
        grid[labeled != largest_label] = 1

    return grid
