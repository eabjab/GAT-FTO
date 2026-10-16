"""2D Bresenham raycasting sensor simulation with FOV and wall occlusion."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import EnvConfig


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to the interval [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def bresenham_line(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Generate integer grid coordinates along a line from (r0, c0) to (r1, c1) using Bresenham's algorithm."""
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0

    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return points


class RaycastingSensor:
    """Simulated 2D range sensor using integer Bresenham raycasting on occupancy grids."""

    def __init__(self, sensor_range: int | EnvConfig = 20, fov_deg: float = 360.0) -> None:
        if hasattr(sensor_range, "sensor_range"):
            cfg = sensor_range
            self.sensor_range = int(cfg.sensor_range)
            self.fov_deg = float(getattr(cfg, "sensor_fov_deg", fov_deg))
        else:
            self.sensor_range = int(sensor_range)
            self.fov_deg = float(fov_deg)

        self.fov_rad = math.radians(self.fov_deg)
        self._half_fov_rad = 0.5 * self.fov_rad
        self._is_omni = self.fov_deg >= 360.0 - 1e-6

        # Precompute relative Bresenham rays from (0, 0) to all perimeter cells of the
        # bounding circle of radius `sensor_range`. Using the 4-connected discrete
        # boundary annulus (R - 1.0, R + 0.5] truncated to Euclidean radius <= R
        # guarantees 100% coverage of the interior disk without Moire gaps.
        self._rays: list[tuple[float, np.ndarray]] = []
        r_max = self.sensor_range
        r_max_sq = r_max * r_max

        if r_max > 0:
            for dr in range(-r_max, r_max + 1):
                for dc in range(-r_max, r_max + 1):
                    dist = math.hypot(dr, dc)
                    if (r_max - 1.0) < dist <= (r_max + 0.5):
                        angle = math.atan2(dr, dc)
                        raw_pts = bresenham_line(0, 0, dr, dc)
                        valid_pts = [
                            (pr, pc) for pr, pc in raw_pts if (pr * pr + pc * pc) <= r_max_sq
                        ]
                        if valid_pts:
                            self._rays.append((angle, np.array(valid_pts, dtype=np.int32)))

    def scan(
        self,
        ground_truth: np.ndarray,
        belief_map: np.ndarray,
        pose: tuple[int, int, float] | list[float],
    ) -> tuple[np.ndarray, int]:
        """Cast rays from `pose = (r0, c0, theta0)` and update `belief_map` in-place.

        Args:
            ground_truth: 2D array of shape (H, W) where 0 = Free, 1 = Obstacle.
            belief_map: 2D array of shape (H, W) where -1 = Unknown, 0 = Free, 1 = Obstacle.
            pose: Tuple `(r0, c0, theta0)` with grid cell `(r0, c0)` and heading `theta0` in radians.

        Returns:
            Tuple `(belief_map, newly_uncovered_free_cells)` where `newly_uncovered_free_cells`
            is the integer count of cells that transitioned from `-1` to `0` (`ground_truth == 0`).
        """
        height, width = ground_truth.shape
        r0 = int(pose[0])
        c0 = int(pose[1])
        theta0 = float(pose[2]) if len(pose) >= 3 else 0.0

        newly_uncovered = 0

        # Ensure origin cell itself is scanned if within grid bounds
        if 0 <= r0 < height and 0 <= c0 < width:
            if ground_truth[r0, c0] == 1:
                belief_map[r0, c0] = 1
                return belief_map, 0
            if belief_map[r0, c0] == -1:
                belief_map[r0, c0] = 0
                newly_uncovered += 1
            else:
                belief_map[r0, c0] = 0

        if self.sensor_range <= 0:
            return belief_map, newly_uncovered

        is_omni = self._is_omni
        half_fov = self._half_fov_rad

        for ray_angle, rel_coords in self._rays:
            if not is_omni:
                # Angle difference in [-pi, pi]
                diff = (ray_angle - theta0 + math.pi) % (2.0 * math.pi) - math.pi
                if abs(diff) > half_fov + 1e-9:
                    continue

            # Step along integer Bresenham line coordinates
            for i in range(1, len(rel_coords)):
                rr = r0 + int(rel_coords[i, 0])
                cc = c0 + int(rel_coords[i, 1])

                if rr < 0 or rr >= height or cc < 0 or cc >= width:
                    break

                if ground_truth[rr, cc] == 1:
                    belief_map[rr, cc] = 1
                    break

                if belief_map[rr, cc] == -1:
                    belief_map[rr, cc] = 0
                    newly_uncovered += 1
                elif belief_map[rr, cc] != 0:
                    belief_map[rr, cc] = 0

        return belief_map, int(newly_uncovered)
