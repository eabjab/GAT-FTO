"""Gymnasium Semi-MDP environment for autonomous 2D frontier exploration."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from torch_geometric.data import Data

from config import EnvConfig
from envs.frontier_extractor import (
    FrontierExtractor,
    FrontierNode,
    _estimate_unmapped_area,
    compute_frontier_mask,
     compute_obstacle_clearance_map,
    extract_frontiers,
)
from envs.graph_builder import GraphBuilder
from envs.map_generator import generate_bsp_map
from envs.pathfinding import SQRT2, astar_path, wrap_angle
from envs.sensor import RaycastingSensor


class PrimitiveStepRecord(dict):
    """Dictionary record of a primitive step supporting attribute and tuple-index access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from exc

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            tuple_view = (
                super().__getitem__("pose"),
                super().__getitem__("belief_map"),
                super().__getitem__("coverage"),
                super().__getitem__("distance"),
            )
            return tuple_view[key]
        return super().__getitem__(key)


class FrontierExplorationEnv(gym.Env):
    """Semi-MDP environment for graph-based autonomous frontier exploration.

    Implements Section 3.1.4 of SPECS.md:
    - Each macro-step selects a frontier node index ``a_t in {0, ..., K_t - 1}``.
    - Plans the shortest 8-connected A* path over known free cells (``belief_map == 0``)
      from the current agent position to the selected frontier medoid.
    - Steps sequentially along the path up to ``macro_step_horizon`` steps, performing
      2D Bresenham raycasting at each step and checking early termination conditions:
        (a) agent reaches the target medoid,
        (b) all cells in the target frontier's ``cells`` array have been uncovered, or
        (c) overall map coverage reaches ``coverage_target``.
    - If the agent reaches the target medoid, aligns heading ``theta_0 = psi_{a_t}``
      and performs one final sensor scan.
    - Computes the normalized Semi-MDP reward and constructs the next PyG ``Data`` graph.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        base_cfg = config if config is not None else EnvConfig()

        # Support convenient keyword overrides (e.g. grid_height=50, height=50)
        cfg_Overrides: dict[str, Any] = {}
        if "height" in kwargs and "grid_height" not in kwargs:
            cfg_Overrides["grid_height"] = int(kwargs.pop("height"))
        if "width" in kwargs and "grid_width" not in kwargs:
            cfg_Overrides["grid_width"] = int(kwargs.pop("width"))

        for field_name in base_cfg.__dataclass_fields__:
            if field_name in kwargs:
                cfg_Overrides[field_name] = kwargs.pop(field_name)

        if cfg_Overrides:
            base_cfg = replace(base_cfg, **cfg_Overrides)

        self.config: EnvConfig = base_cfg
        self.render_mode = render_mode
        self._init_seed: int | None = kwargs.pop("seed", None)
        self._rng: np.random.Generator = np.random.default_rng(self._init_seed)

        self.height: int = int(self.config.grid_height)
        self.width: int = int(self.config.grid_width)
        self.d_diag: float = math.hypot(float(self.height), float(self.width))

        # Core environment components
        self.sensor = RaycastingSensor(
            sensor_range=self.config.sensor_range,
            fov_deg=self.config.sensor_fov_deg,
        )
        self.frontier_extractor = FrontierExtractor(
            min_frontier_size=self.config.min_frontier_size,
            max_cluster_size=self.config.max_cluster_size,
            sensor_range=self.config.sensor_range,
            fov_deg=self.config.sensor_fov_deg,
        )
        self.graph_builder = GraphBuilder(
            sensor_range=self.config.sensor_range,
            max_cluster_size=self.config.max_cluster_size,
        )

        # Gymnasium spaces (updated dynamically as K_t changes)
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(1, 8), dtype=np.float32
        )

        # Episode state variables
        self.ground_truth: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)
        self.belief_map: np.ndarray = np.full((self.height, self.width), -1, dtype=np.int8)
        self.agent_pose: tuple[int, int, float] = (1, 1, 0.0)
        self.frontiers: list[FrontierNode] = []
        self.current_graph: Data | None = None

        self.total_free_cells: int = 1
        self.uncovered_free_cells: int = 0
        self.step_count: int = 0
        self.primitive_steps: int = 0
        self.backtracking_steps: int = 0
        self.total_distance: float = 0.0
        self.dist_to_95: float | None = None
        self.dist_to_98: float | None = None

        # Historical trajectory & rollout tracking for evaluation and visualizers
        self.visited_cells: set[tuple[int, int]] = set()
        self.trajectory: list[tuple[int, int]] = []
        self.pose_history: list[tuple[int, int, float]] = []
        self.coverage_history: list[tuple[float, float]] = []
        self.distance_history: list[float] = []
        self.primitive_history: list[PrimitiveStepRecord] = []
        self.macro_step_history: list[dict[str, Any]] = []
        self.last_planned_path: list[tuple[int, int]] = []
        self.last_executed_path: list[tuple[int, int]] = []
        self.last_action_idx: int | None = None

    @property
    def coverage(self) -> float:
        """Current fraction of total free cells uncovered in [0.0, 1.0]."""
        if self.total_free_cells <= 0:
            return 1.0
        return float(self.uncovered_free_cells) / float(self.total_free_cells)

    @property
    def distance_traveled(self) -> float:
        """Alias for total cumulative distance traveled in the episode."""
        return float(self.total_distance)

    @property
    def backtracking_ratio(self) -> float:
        """Fraction of primitive path steps spent on previously visited grid cells."""
        if self.primitive_steps <= 0:
            return 0.0
        return float(self.backtracking_steps) / float(self.primitive_steps)

    @property
    def exploration_efficiency(self) -> float:
        """Mean free cells uncovered per unit distance traveled."""
        if self.total_distance <= 1e-8:
            return float(self.uncovered_free_cells)
        return float(self.uncovered_free_cells) / float(self.total_distance)

    def _select_initial_pose(
        self, ground_truth: np.ndarray, rng: np.random.Generator
    ) -> tuple[int, int, float]:
        """Select a deterministic initial free cell (preferring room interiors with clearance)."""
        edt = compute_obstacle_clearance_map(ground_truth)
        candidates = np.argwhere((ground_truth == 0) & (edt >= 2.0))
        if len(candidates) == 0:
            candidates = np.argwhere(ground_truth == 0)
        idx = int(rng.integers(0, len(candidates)))
        r0 = int(candidates[idx, 0])
        c0 = int(candidates[idx, 1])
        return (r0, c0, 0.0)

    def _extract_active_frontiers(self) -> list[FrontierNode]:
        """Extract active frontiers from belief_map."""
        frontiers = self.frontier_extractor.extract(self.belief_map)
        return frontiers

    def _record_primitive_snapshot(self) -> None:
        """Append the current state to primitive_history and coverage/distance histories."""
        cov = self.coverage
        dist = float(self.total_distance)
        if self.dist_to_95 is None and cov >= 0.95:
            self.dist_to_95 = dist
        if self.dist_to_98 is None and cov >= 0.98:
            self.dist_to_98 = dist

        self.coverage_history.append((dist, cov))
        self.distance_history.append(dist)

        r, c, theta = self.agent_pose
        record = PrimitiveStepRecord(
            pose=(int(r), int(c), float(theta)),
            position=(int(r), int(c)),
            belief_map=self.belief_map.copy(),
            belief=self.belief_map.copy(),
            coverage=float(cov),
            cov=float(cov),
            distance=dist,
            dist=dist,
            uncovered_free_cells=int(self.uncovered_free_cells),
            frontiers=list(self.frontiers),
            trajectory=list(self.trajectory),
        )
        self.primitive_history.append(record)

    def _update_latest_primitive_snapshot(self) -> None:
        """Update the most recent primitive snapshot after final orientation alignment scan."""
        cov = self.coverage
        dist = float(self.total_distance)
        if self.dist_to_95 is None and cov >= 0.95:
            self.dist_to_95 = dist
        if self.dist_to_98 is None and cov >= 0.98:
            self.dist_to_98 = dist

        if self.coverage_history:
            self.coverage_history[-1] = (dist, cov)
        else:
            self.coverage_history.append((dist, cov))

        r, c, theta = self.agent_pose
        if self.primitive_history:
            rec = self.primitive_history[-1]
            rec["pose"] = (int(r), int(c), float(theta))
            rec["position"] = (int(r), int(c))
            rec["belief_map"] = self.belief_map.copy()
            rec["belief"] = self.belief_map.copy()
            rec["coverage"] = float(cov)
            rec["cov"] = float(cov)
            rec["distance"] = dist
            rec["dist"] = dist
            rec["uncovered_free_cells"] = int(self.uncovered_free_cells)
            rec["frontiers"] = list(self.frontiers)
            rec["trajectory"] = list(self.trajectory)
        else:
            self._record_primitive_snapshot()

    def _is_target_frontier_uncovered(
        self,
        target_frontier: FrontierNode,
        initial_cells_had_unknown: bool,
    ) -> bool:
        """Check early termination condition (b): whether the target frontier has been uncovered."""
        cells = target_frontier.cells
        if cells.size == 0:
            return True

        rows = cells[:, 0]
        cols = cells[:, 1]

        # If cells directly contained unknown (-1) values at step start, check if all are != -1
        if initial_cells_had_unknown:
            return bool(np.all(self.belief_map[rows, cols] != -1))

        # For standard frontier cells (which are free cells bordering unknown space):
        # Check if any cell in the cluster still borders unknown space (-1) AND whether any
        # unknown area remains visible from the frontier medoid.
        frontier_mask = compute_frontier_mask(self.belief_map)
        if bool(np.any(frontier_mask[rows, cols])):
            return False

        r_m, c_m = int(target_frontier.position[0]), int(target_frontier.position[1])
        remaining_unmapped = _estimate_unmapped_area(
            belief_map=self.belief_map,
            r_i=r_m,
            c_i=c_m,
            psi_i=float(target_frontier.orientation),
            sensor_range=self.config.sensor_range,
            fov_deg=self.config.sensor_fov_deg,
        )
        return remaining_unmapped <= 1e-6

    def _build_info(
        self,
        *,
        reached_target: bool = False,
        frontier_uncovered: bool = False,
        macro_dist: float = 0.0,
        macro_uncovered: int = 0,
        delta_theta_init: float = 0.0,
    ) -> dict[str, Any]:
        """Construct the step/reset info dictionary."""
        return {
            "coverage": float(self.coverage),
            "uncovered_free_cells": int(self.uncovered_free_cells),
            "total_free_cells": int(self.total_free_cells),
            "num_frontiers": int(len(self.frontiers)),
            "frontiers": self.frontiers,
            "agent_pose": self.agent_pose,
            "distance_traveled": float(self.total_distance),
            "total_distance": float(self.total_distance),
            "macro_distance": float(macro_dist),
            "macro_uncovered": int(macro_uncovered),
            "delta_theta_init": float(delta_theta_init),
            "reached_target": bool(reached_target),
            "frontier_reached": bool(reached_target),
            "frontier_uncovered": bool(frontier_uncovered),
            "step_count": int(self.step_count),
            "primitive_steps": int(self.primitive_steps),
            "backtracking_steps": int(self.backtracking_steps),
            "backtracking_ratio": float(self.backtracking_ratio),
            "exploration_efficiency": float(self.exploration_efficiency),
            "dist_to_95": self.dist_to_95,
            "dist_to_98": self.dist_to_98,
            "trajectory": self.trajectory,
            "planned_path": self.last_planned_path,
            "executed_path": self.last_executed_path,
            "primitive_history": self.primitive_history,
            "coverage_history": self.coverage_history,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Data, dict[str, Any]]:
        """Reset the environment to a new episode and return ``(initial_graph, info)``."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            map_seed = seed
        else:
            map_seed = int(self._rng.integers(0, 2**31 - 1))

        opts = options if options is not None else {}

        if "ground_truth" in opts and opts["ground_truth"] is not None:
            self.ground_truth = np.array(opts["ground_truth"], dtype=np.int8, copy=True)
            self.height, self.width = self.ground_truth.shape
            self.d_diag = math.hypot(float(self.height), float(self.width))
        else:
            self.height = int(self.config.grid_height)
            self.width = int(self.config.grid_width)
            self.d_diag = math.hypot(float(self.height), float(self.width))
            self.ground_truth = generate_bsp_map(
                height=self.height, width=self.width, seed=map_seed
            )

        self.total_free_cells = int(np.count_nonzero(self.ground_truth == 0))
        self.belief_map = np.full((self.height, self.width), -1, dtype=np.int8)

        # Select initial agent pose
        pose_rng = np.random.default_rng(map_seed)
        if "agent_pose" in opts and opts["agent_pose"] is not None:
            ap = opts["agent_pose"]
            self.agent_pose = (int(ap[0]), int(ap[1]), float(ap[2]) if len(ap) >= 3 else 0.0)
        elif "start_pose" in opts and opts["start_pose"] is not None:
            sp = opts["start_pose"]
            self.agent_pose = (int(sp[0]), int(sp[1]), float(sp[2]) if len(sp) >= 3 else 0.0)
        else:
            self.agent_pose = self._select_initial_pose(self.ground_truth, pose_rng)

        # Perform initial sensor scan
        self.sensor.scan(self.ground_truth, self.belief_map, self.agent_pose)
        self.uncovered_free_cells = int(
            np.count_nonzero((self.belief_map == 0) & (self.ground_truth == 0))
        )

        # Extract initial frontiers and construct topological graph
        self.frontiers = self._extract_active_frontiers()
        self.current_graph = self.graph_builder.build_graph(
            self.agent_pose, self.frontiers, self.belief_map
        )
        self.action_space = spaces.Discrete(max(1, len(self.frontiers)))

        # Reset episode counters and trajectory histories
        r0, c0, theta0 = self.agent_pose
        self.step_count = 0
        self.primitive_steps = 0
        self.backtracking_steps = 0
        self.total_distance = 0.0
        self.dist_to_95 = 0.0 if self.coverage >= 0.95 else None
        self.dist_to_98 = 0.0 if self.coverage >= 0.98 else None

        self.visited_cells = {(int(r0), int(c0))}
        self.trajectory = [(int(r0), int(c0))]
        self.pose_history = [(int(r0), int(c0), float(theta0))]
        self.coverage_history = []
        self.distance_history = []
        self.primitive_history = []
        self.macro_step_history = []
        self.last_planned_path = [(int(r0), int(c0))]
        self.last_executed_path = [(int(r0), int(c0))]
        self.last_action_idx = None

        self._record_primitive_snapshot()
        self.macro_step_history.append(
            {
                "step": 0,
                "agent_pose": self.agent_pose,
                "belief_map": self.belief_map.copy(),
                "frontiers": list(self.frontiers),
                "graph": self.current_graph,
                "action_idx": None,
                "planned_path": list(self.last_planned_path),
                "executed_path": list(self.last_executed_path),
                "trajectory": list(self.trajectory),
                "coverage": float(self.coverage),
                "distance": float(self.total_distance),
            }
        )

        info = self._build_info()
        return self.current_graph, info

    def step(
        self, action_idx: int | np.integer | Any
    ) -> tuple[Data, float, bool, bool, dict[str, Any]]:
        """Execute one Semi-MDP macro-step toward the selected frontier ``action_idx``."""
        if hasattr(action_idx, "item"):
            action_idx = int(action_idx.item())
        else:
            action_idx = int(action_idx)

        self.step_count += 1
        r0, c0, theta0 = self.agent_pose
        prev_uncovered = int(self.uncovered_free_cells)

        macro_dist = 0.0
        delta_theta_init = 0.0
        reached_target = False
        frontier_uncovered = False
        executed_path: list[tuple[int, int]] = [(int(r0), int(c0))]

        # Graceful handling when K_t == 0 (no active frontiers available)
        if len(self.frontiers) == 0:
            self.last_action_idx = None
            self.last_planned_path = [(int(r0), int(c0))]
            self.last_executed_path = executed_path
            self.current_graph = self.graph_builder.build_graph(
                self.agent_pose, self.frontiers, self.belief_map
            )
            self.action_space = spaces.Discrete(1)

            terminated = True
            truncated = False
            r_terminal = (
                float(self.config.r_success)
                if self.coverage >= self.config.coverage_target
                else 0.0
            )
            info = self._build_info(
                reached_target=False,
                frontier_uncovered=False,
                macro_dist=0.0,
                macro_uncovered=0,
                delta_theta_init=0.0,
            )
            return self.current_graph, r_terminal, terminated, truncated, info

        # Clamp action_idx into valid frontier range {0, ..., K_t - 1}
        valid_action_idx = max(0, min(len(self.frontiers) - 1, action_idx))
        self.last_action_idx = valid_action_idx
        target_frontier = self.frontiers[valid_action_idx]
        target_pos = (int(target_frontier.position[0]), int(target_frontier.position[1]))
        psi_target = float(target_frontier.orientation)

        initial_cells_had_unknown = bool(
            target_frontier.cells.size > 0
            and np.any(
                self.belief_map[
                    target_frontier.cells[:, 0], target_frontier.cells[:, 1]
                ]
                == -1
            )
        )

        # Compute shortest 8-connected A* path over known free cells (belief_map == 0)
        path, _ = astar_path(self.belief_map, (int(r0), int(c0)), target_pos, free_val=0)
        if not path:
            path = [(int(r0), int(c0))]
        self.last_planned_path = list(path)

        # Initial heading rotation \Delta \theta_{init}
        if len(path) >= 2:
            r1, c1 = path[1]
            first_step_angle = math.atan2(float(r1 - r0), float(c1 - c0))
            delta_theta_init = abs(float(wrap_angle(first_step_angle - theta0)))
        else:
            delta_theta_init = abs(float(wrap_angle(psi_target - theta0)))

        # Step sequentially along path P for at most macro_step_horizon steps
        max_steps = min(len(path) - 1, int(self.config.macro_step_horizon))

        if len(path) == 1 and (int(r0), int(c0)) == target_pos:
            reached_target = True

        for step_i in range(1, max_steps + 1):
            prev_r, prev_c = path[step_i - 1]
            curr_r, curr_c = path[step_i]
            curr_theta = math.atan2(float(curr_r - prev_r), float(curr_c - prev_c))

            dr = abs(int(curr_r) - int(prev_r))
            dc = abs(int(curr_c) - int(prev_c))
            step_dist = 1.0 if (dr + dc == 1) else SQRT2

            macro_dist += step_dist
            self.total_distance += step_dist
            self.primitive_steps += 1

            pos_tuple = (int(curr_r), int(curr_c))
            if pos_tuple in self.visited_cells:
                self.backtracking_steps += 1
            else:
                self.visited_cells.add(pos_tuple)

            self.agent_pose = (int(curr_r), int(curr_c), float(curr_theta))
            self.trajectory.append(pos_tuple)
            self.pose_history.append(self.agent_pose)
            executed_path.append(pos_tuple)

            # Perform 2D raycasting sensor scan at new pose
            self.sensor.scan(self.ground_truth, self.belief_map, self.agent_pose)
            self.uncovered_free_cells = int(
                np.count_nonzero((self.belief_map == 0) & (self.ground_truth == 0))
            )

            self._record_primitive_snapshot()

            # Early termination condition (a): reached target medoid
            if pos_tuple == target_pos:
                reached_target = True
                break

            # Early termination condition (c): overall map coverage reaches coverage_target
            if self.coverage >= self.config.coverage_target:
                break

            # Early termination condition (b): all cells in target frontier uncovered
            if self._is_target_frontier_uncovered(
                target_frontier, initial_cells_had_unknown
            ):
                frontier_uncovered = True
                break

        # Final orientation alignment + scan if the agent reached the target medoid
        if reached_target:
            self.agent_pose = (target_pos[0], target_pos[1], float(psi_target))
            self.pose_history[-1] = self.agent_pose
            self.sensor.scan(self.ground_truth, self.belief_map, self.agent_pose)
            self.uncovered_free_cells = int(
                np.count_nonzero((self.belief_map == 0) & (self.ground_truth == 0))
            )
            self._update_latest_primitive_snapshot()

        self.last_executed_path = executed_path
        macro_uncovered = max(0, int(self.uncovered_free_cells) - prev_uncovered)

        # Extract updated frontiers and construct next topological graph
        self.frontiers = self._extract_active_frontiers()
        if self.primitive_history:
            self.primitive_history[-1]["frontiers"] = list(self.frontiers)

        self.current_graph = self.graph_builder.build_graph(
            self.agent_pose, self.frontiers, self.belief_map
        )
        self.action_space = spaces.Discrete(max(1, len(self.frontiers)))

        # Episode termination & truncation conditions
        coverage_reached = self.coverage >= self.config.coverage_target
        no_frontiers_left = len(self.frontiers) == 0
        terminated = bool(coverage_reached or no_frontiers_left)
        truncated = bool(
            (not terminated) and (self.step_count >= self.config.max_episode_steps)
        )

        # Terminal reward component
        if coverage_reached:
            r_terminal = float(self.config.r_success)
        elif truncated:
            r_terminal = float(self.config.r_timeout)
        else:
            r_terminal = 0.0

        # Section 3.1.4 Reward Calculation
        cov_term = float(self.config.w_cov) * (
            float(macro_uncovered) / float(max(1, self.total_free_cells))
        )
        dist_term = float(self.config.w_dist) * (
            float(macro_dist) / float(max(1e-8, self.d_diag))
        )
        turn_term = float(self.config.w_turn) * (float(delta_theta_init) / math.pi)
        reward = float(cov_term - dist_term - turn_term + r_terminal)

        self.macro_step_history.append(
            {
                "step": int(self.step_count),
                "agent_pose": self.agent_pose,
                "belief_map": self.belief_map.copy(),
                "frontiers": list(self.frontiers),
                "graph": self.current_graph,
                "action_idx": int(valid_action_idx),
                "planned_path": list(self.last_planned_path),
                "executed_path": list(self.last_executed_path),
                "trajectory": list(self.trajectory),
                "coverage": float(self.coverage),
                "distance": float(self.total_distance),
                "reward": float(reward),
                "reached_target": bool(reached_target),
            }
        )

        info = self._build_info(
            reached_target=reached_target,
            frontier_uncovered=frontier_uncovered,
            macro_dist=macro_dist,
            macro_uncovered=macro_uncovered,
            delta_theta_init=delta_theta_init,
        )
        return self.current_graph, reward, terminated, truncated, info

    def get_state_at_distance(self, target_distance: float) -> PrimitiveStepRecord:
        """Return the primitive step snapshot corresponding to cumulative distance ``target_distance``."""
        if not self.primitive_history:
            raise RuntimeError("Environment has no recorded primitive history. Call reset() first.")
        if target_distance <= 0.0:
            return self.primitive_history[0]
        if target_distance >= self.total_distance:
            return self.primitive_history[-1]

        idx = int(np.searchsorted(self.distance_history, target_distance, side="right")) - 1
        idx = max(0, min(len(self.primitive_history) - 1, idx))
        return self.primitive_history[idx]
