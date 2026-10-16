"""Mode 2: Synchronized Side-by-Side Race Visualizer (SPECS.md Section 3.6.2)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.animation
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines.classical_agents import NearestFrontierAgent
from config import EnvConfig
from envs.exploration_env import FrontierExplorationEnv
from envs.frontier_extractor import FrontierExtractor, FrontierNode
from models.pointer_actor_critic import PointerActorCritic
from visualization.dual_view_visualizer import _belief_map_to_rgb, _create_pose_triangle


def _normalize_primitive_history(
    source: Any,
    frontier_extractor: FrontierExtractor | None = None,
) -> list[dict[str, Any]]:
    """Normalize an environment, primitive history list, or single snapshot into standard dicts.

    Supports:
    - ``FrontierExplorationEnv`` (extracts ``source.primitive_history``)
    - List of ``PrimitiveStepRecord`` / ``dict``
    - List of tuples ``(pose, belief_map, coverage)`` or ``(pose, belief_map, coverage, distance)``
    - Single ``dict`` or tuple snapshot
    """
    if hasattr(source, "primitive_history") and source.primitive_history:
        raw_list = list(source.primitive_history)
    elif isinstance(source, dict):
        raw_list = [source]
    elif isinstance(source, (list, tuple)):
        if (
            len(source) in (3, 4)
            and isinstance(source[1], np.ndarray)
            and not isinstance(source[0], (dict, list))
        ):
            raw_list = [source]
        else:
            raw_list = list(source)
    else:
        raise ValueError(
            f"Unsupported history source type {type(source).__name__} for RaceVisualizer."
        )

    normalized: list[dict[str, Any]] = []
    cum_dist = 0.0
    traj_accum: list[tuple[int, int]] = []

    for idx, item in enumerate(raw_list):
        if isinstance(item, dict):
            pose_raw = item.get("pose", item.get("agent_pose", item.get("position", (0, 0, 0.0))))
            bmap = np.asarray(item.get("belief_map", item.get("belief")), dtype=np.int8)
            cov = float(item.get("coverage", item.get("cov", 0.0)))
            dist_val = item.get("distance", item.get("dist", None))
            frontiers = item.get("frontiers", None)
            traj = item.get("trajectory", None)
        elif isinstance(item, (tuple, list)):
            pose_raw = item[0]
            bmap = np.asarray(item[1], dtype=np.int8)
            cov = float(item[2])
            dist_val = float(item[3]) if len(item) >= 4 else None
            frontiers = item[4] if len(item) >= 5 else None
            traj = item[5] if len(item) >= 6 else None
        else:
            raise ValueError(f"Unsupported primitive history element: {type(item).__name__}")

        r = int(pose_raw[0])
        c = int(pose_raw[1])
        theta = float(pose_raw[2]) if len(pose_raw) >= 3 else 0.0

        if idx == 0:
            if dist_val is not None:
                cum_dist = float(dist_val)
            else:
                cum_dist = 0.0
        else:
            if dist_val is not None:
                cum_dist = float(dist_val)
            else:
                prev_r, prev_c = traj_accum[-1]
                cum_dist += math.hypot(float(r - prev_r), float(c - prev_c))

        traj_accum.append((r, c))
        traj_list = [(int(p[0]), int(p[1])) for p in traj] if traj is not None else list(traj_accum)

        if frontiers is None:
            if frontier_extractor is not None and idx == len(raw_list) - 1:
                frontiers_list = frontier_extractor.extract(bmap)
            else:
                frontiers_list = []
        else:
            frontiers_list = list(frontiers)

        normalized.append(
            {
                "pose": (r, c, theta),
                "belief_map": bmap,
                "coverage": cov,
                "distance": cum_dist,
                "frontiers": frontiers_list,
                "trajectory": traj_list,
            }
        )

    return normalized


def interpolate_history_at_distance(
    history: list[dict[str, Any]],
    target_distance: float,
) -> dict[str, Any]:
    """Interpolate a normalized primitive step history at cumulative distance ``target_distance``."""
    if not history:
        raise ValueError("Cannot interpolate an empty primitive history.")

    dists = np.array([float(step["distance"]) for step in history], dtype=np.float64)
    d_max = float(dists[-1])
    d_query = float(max(0.0, target_distance))

    if d_query <= dists[0] or len(history) == 1:
        snap = history[0]
        return {
            "pose": snap["pose"],
            "interp_rc": (float(snap["pose"][0]), float(snap["pose"][1])),
            "belief_map": snap["belief_map"],
            "coverage": float(snap["coverage"]),
            "distance": float(dists[0]),
            "max_distance": d_max,
            "completed": d_query >= d_max,
            "frontiers": snap["frontiers"],
            "trajectory": snap["trajectory"],
            "dist_curve": np.array([dists[0]], dtype=np.float64),
            "cov_curve": np.array([float(snap["coverage"]) * 100.0], dtype=np.float64),
        }

    if d_query >= d_max:
        snap = history[-1]
        all_covs = np.array([float(s["coverage"]) * 100.0 for s in history], dtype=np.float64)
        return {
            "pose": snap["pose"],
            "interp_rc": (float(snap["pose"][0]), float(snap["pose"][1])),
            "belief_map": snap["belief_map"],
            "coverage": float(snap["coverage"]),
            "distance": d_max,
            "max_distance": d_max,
            "completed": True,
            "frontiers": snap["frontiers"],
            "trajectory": snap["trajectory"],
            "dist_curve": dists,
            "cov_curve": all_covs,
        }

    idx = int(np.searchsorted(dists, d_query, side="right")) - 1
    idx = max(0, min(len(history) - 2, idx))

    s0 = history[idx]
    s1 = history[idx + 1]
    d0 = float(dists[idx])
    d1 = float(dists[idx + 1])

    if d1 > d0 + 1e-8:
        alpha = float((d_query - d0) / (d1 - d0))
    else:
        alpha = 0.0

    r0, c0, theta0 = s0["pose"]
    r1, c1, theta1 = s1["pose"]
    interp_r = (1.0 - alpha) * float(r0) + alpha * float(r1)
    interp_c = (1.0 - alpha) * float(c0) + alpha * float(c1)
    interp_cov = (1.0 - alpha) * float(s0["coverage"]) + alpha * float(s1["coverage"])

    traj_prefix = list(s0["trajectory"])
    dist_curve = np.append(dists[: idx + 1], d_query)
    cov_prefix = [float(history[k]["coverage"]) * 100.0 for k in range(idx + 1)]
    cov_curve = np.append(np.array(cov_prefix, dtype=np.float64), interp_cov * 100.0)

    return {
        "pose": (r0, c0, theta1 if alpha > 0.0 else theta0),
        "interp_rc": (interp_r, interp_c),
        "belief_map": s0["belief_map"],
        "coverage": interp_cov,
        "distance": d_query,
        "max_distance": d_max,
        "completed": False,
        "frontiers": s0["frontiers"],
        "trajectory": traj_prefix,
        "dist_curve": dist_curve,
        "cov_curve": cov_curve,
    }


class RaceVisualizer:
    """Mode 2: Synchronized Side-by-Side Race Visualizer (SPECS.md Section 3.6.2).

    Compares ``NearestFrontierAgent`` vs. ``PointerActorCritic`` on the exact same map seed:
    - Distance-Synchronized Interpolation: Rolls out both episodes and records full
      primitive step histories ``[(p_0, belief_0, cov_0), ...]`` indexed by cumulative
      distance traveled ``d in [0, D_max]``. Renders frames at uniform distance
      increments ``delta_d = 2.0`` grid units so both agents move at the exact same speed.
    - Layout (``matplotlib.gridspec.GridSpec(2, 2, height_ratios=[3, 1])``):
      - Top-Left: Nearest Frontier occupancy belief map, historical trajectory, and active frontiers.
      - Top-Right: Trained GAT Policy occupancy belief map, historical trajectory, and active frontiers.
      - Bottom (Spanning Both Columns): Live animated line plot of ``% Map Covered``
        (y-axis, 0% to 100%) vs. ``Distance Traveled`` (x-axis), with vertical markers
        indicating when each agent completes exploration.
    """

    def __init__(
        self,
        nearest_source: Any = None,
        gat_source: Any = None,
        *,
        delta_d: float = 2.0,
        figsize: tuple[float, float] = (14.0, 8.0),
        dpi: int = 150,
        model: PointerActorCritic | Any | None = None,
    ) -> None:
        self.delta_d = float(delta_d)
        self.figsize = (float(figsize[0]), float(figsize[1]))
        self.dpi = int(dpi)
        self.model = model
        self._frontier_extractor = FrontierExtractor()

        self.nearest_history: list[dict[str, Any]] = []
        self.gat_history: list[dict[str, Any]] = []
        self.frames: list[np.ndarray] = []

        if nearest_source is not None:
            self.nearest_history = _normalize_primitive_history(
                nearest_source, self._frontier_extractor
            )
        if gat_source is not None:
            self.gat_history = _normalize_primitive_history(
                gat_source, self._frontier_extractor
            )

        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.gs = GridSpec(2, 2, figure=self.fig, height_ratios=[3, 1])
        self.ax_nearest = self.fig.add_subplot(self.gs[0, 0])
        self.ax_gat = self.fig.add_subplot(self.gs[0, 1])
        self.ax_cov = self.fig.add_subplot(self.gs[1, :])

    def set_histories(self, nearest_source: Any, gat_source: Any) -> None:
        """Set or replace the primitive step histories for both agents."""
        self.nearest_history = _normalize_primitive_history(
            nearest_source, self._frontier_extractor
        )
        self.gat_history = _normalize_primitive_history(
            gat_source, self._frontier_extractor
        )

    def rollout_episodes(
        self,
        env_config: EnvConfig | None = None,
        seed: int = 42,
        max_steps: int | None = None,
        nearest_agent: Any | None = None,
        gat_agent: PointerActorCritic | Any | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Roll out NearestFrontierAgent and PointerActorCritic on the exact same map seed."""
        cfg = env_config if env_config is not None else EnvConfig()
        n_agent = nearest_agent if nearest_agent is not None else NearestFrontierAgent()
        g_agent = gat_agent if gat_agent is not None else (
            self.model if self.model is not None else PointerActorCritic()
        )
        if isinstance(g_agent, torch.nn.Module):
            g_agent.eval()

        step_limit = max_steps if max_steps is not None else cfg.max_episode_steps

        # Rollout 1: NearestFrontierAgent
        env_nearest = FrontierExplorationEnv(cfg)
        obs_n, _ = env_nearest.reset(seed=seed)
        term_n, trunc_n = False, False
        steps_n = 0
        while not (term_n or trunc_n) and steps_n < step_limit:
            act_n = n_agent.select_action(obs_n, deterministic=True)
            obs_n, _, term_n, trunc_n, _ = env_nearest.step(act_n)
            steps_n += 1

        # Rollout 2: PointerActorCritic on the exact same map & initial pose
        env_gat = FrontierExplorationEnv(cfg)
        obs_g, _ = env_gat.reset(seed=seed)
        term_g, trunc_g = False, False
        steps_g = 0
        while not (term_g or trunc_g) and steps_g < step_limit:
            act_g = g_agent.select_action(obs_g, deterministic=True)
            obs_g, _, term_g, trunc_g, _ = env_gat.step(act_g)
            steps_g += 1

        self.set_histories(env_nearest, env_gat)
        return self.nearest_history, self.gat_history

    def get_synchronized_distances(self, delta_d: float | None = None) -> np.ndarray:
        """Compute uniform distance sample points at increment ``delta_d`` across ``[0, D_max]``."""
        step_d = float(delta_d) if delta_d is not None else self.delta_d
        if step_d <= 0.0:
            raise ValueError("delta_d must be strictly positive.")

        d_nearest = float(self.nearest_history[-1]["distance"]) if self.nearest_history else 0.0
        d_gat = float(self.gat_history[-1]["distance"]) if self.gat_history else 0.0
        d_max = max(d_nearest, d_gat)

        if d_max <= 1e-8:
            return np.array([0.0], dtype=np.float64)

        distances = np.arange(0.0, d_max, step_d, dtype=np.float64)
        if distances.size == 0 or distances[-1] < d_max - 1e-6:
            distances = np.append(distances, d_max)
        return distances

    def _render_agent_panel(
        self,
        ax: matplotlib.axes.Axes,
        interp_state: dict[str, Any],
        title_prefix: str,
        line_color: str,
    ) -> None:
        """Render a single agent's occupancy belief map, trajectory, pose, and active frontiers."""
        bmap: np.ndarray = interp_state["belief_map"]
        pose: tuple[int, int, float] = interp_state["pose"]
        interp_rc: tuple[float, float] = interp_state["interp_rc"]
        traj: list[tuple[int, int]] = interp_state["trajectory"]
        frontiers: list[FrontierNode] = interp_state["frontiers"]
        cov_pct: float = float(interp_state["coverage"]) * 100.0
        cur_dist: float = float(interp_state["distance"])

        h, w = bmap.shape
        scale = max(1.0, min(h, w) / 40.0)

        rgb_map = _belief_map_to_rgb(bmap)
        ax.imshow(rgb_map, origin="upper", interpolation="nearest")

        # Plot active frontiers
        cluster_cmap = matplotlib.colormaps["tab10"]
        for c_idx, f_node in enumerate(frontiers):
            color = cluster_cmap(c_idx % 10)
            if hasattr(f_node, "cells") and f_node.cells is not None and f_node.cells.size > 0:
                ax.scatter(
                    f_node.cells[:, 1],
                    f_node.cells[:, 0],
                    c=[color],
                    s=16.0 * scale,
                    alpha=0.85,
                    edgecolors="none",
                    zorder=4,
                )
            if hasattr(f_node, "position"):
                ax.scatter(
                    [f_node.position[1]],
                    [f_node.position[0]],
                    c=[color],
                    s=32.0 * scale,
                    edgecolors="#000000",
                    linewidths=0.8,
                    zorder=5,
                )

        # Plot cumulative trajectory
        traj_cols = [float(p[1]) for p in traj] + [float(interp_rc[1])]
        traj_rows = [float(p[0]) for p in traj] + [float(interp_rc[0])]
        if len(traj_cols) >= 2:
            ax.plot(
                traj_cols,
                traj_rows,
                color=line_color,
                linewidth=2.0,
                alpha=0.9,
                zorder=6,
            )

        # Plot directional pose triangle
        pose_patch = _create_pose_triangle(
            row=float(interp_rc[0]),
            col=float(interp_rc[1]),
            theta=float(pose[2]),
            scale=1.6 * scale,
            facecolor=line_color,
            edgecolor="#ffffff",
            zorder=8,
        )
        ax.add_patch(pose_patch)

        status_tag = " [DONE]" if interp_state["completed"] else ""
        ax.set_title(
            f"{title_prefix} | Cov: {cov_pct:.1f}% | Dist: {cur_dist:.1f}{status_tag}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Grid Column (c)")
        ax.set_ylabel("Grid Row (r)")

    def render_frame(
        self,
        nearest_source: Any = None,
        gat_source: Any = None,
        *,
        target_distance: float | None = None,
        distance: float | None = None,
        frame_idx: int | None = None,
        delta_d: float | None = None,
        record: bool = True,
    ) -> np.ndarray:
        """Render a single synchronized race frame and return an RGB uint8 array (H_px, W_px, 3)."""
        # Allow passing distance as first positional float/int if histories are already loaded
        if isinstance(nearest_source, (int, float)) and gat_source is None and self.nearest_history:
            target_distance = float(nearest_source)
            nearest_source = None

        if nearest_source is not None and gat_source is not None:
            self.set_histories(nearest_source, gat_source)
        elif nearest_source is not None and gat_source is None:
            norm_hist = _normalize_primitive_history(nearest_source, self._frontier_extractor)
            self.nearest_history = norm_hist
            if not self.gat_history:
                self.gat_history = norm_hist

        if not self.nearest_history or not self.gat_history:
            raise ValueError(
                "RaceVisualizer requires both nearest and GAT histories. "
                "Pass them to __init__, set_histories(), rollout_episodes(), or render_frame()."
            )

        step_d = float(delta_d) if delta_d is not None else self.delta_d
        d_nearest_max = float(self.nearest_history[-1]["distance"])
        d_gat_max = float(self.gat_history[-1]["distance"])
        d_max = max(d_nearest_max, d_gat_max, 1.0)

        if target_distance is None and distance is not None:
            target_distance = float(distance)
        elif target_distance is None and frame_idx is not None:
            target_distance = float(frame_idx) * step_d
        elif target_distance is None:
            target_distance = max(d_nearest_max, d_gat_max)

        state_nearest = interpolate_history_at_distance(self.nearest_history, target_distance)
        state_gat = interpolate_history_at_distance(self.gat_history, target_distance)

        self.ax_nearest.clear()
        self.ax_gat.clear()
        self.ax_cov.clear()

        # Top-Left: Nearest Frontier Agent
        self._render_agent_panel(
            self.ax_nearest,
            state_nearest,
            title_prefix="Nearest Frontier Baseline",
            line_color="#1f77b4",
        )

        # Top-Right: Trained GAT Policy Agent
        self._render_agent_panel(
            self.ax_gat,
            state_gat,
            title_prefix="GATv2 Pointer Policy",
            line_color="#ff7f0e",
        )

        # Bottom (Spanning Both Columns): Live animated line plot of % Map Covered vs Distance Traveled
        self.ax_cov.plot(
            state_nearest["dist_curve"],
            state_nearest["cov_curve"],
            color="#1f77b4",
            linewidth=2.2,
            label="Nearest Frontier",
        )
        self.ax_cov.plot(
            state_gat["dist_curve"],
            state_gat["cov_curve"],
            color="#ff7f0e",
            linewidth=2.2,
            label="GATv2 Pointer Policy",
        )

        # Vertical markers indicating when each agent completes exploration
        if target_distance >= d_nearest_max and d_nearest_max > 0.0:
            self.ax_cov.axvline(
                d_nearest_max,
                color="#1f77b4",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label=f"Nearest Done ({d_nearest_max:.1f})",
            )
        if target_distance >= d_gat_max and d_gat_max > 0.0:
            self.ax_cov.axvline(
                d_gat_max,
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label=f"GAT Done ({d_gat_max:.1f})",
            )

        self.ax_cov.set_xlim(0.0, d_max * 1.05)
        self.ax_cov.set_ylim(0.0, 102.0)
        self.ax_cov.set_xlabel("Distance Traveled (Grid Units)")
        self.ax_cov.set_ylabel("% Map Covered")
        self.ax_cov.set_title("Synchronized Coverage vs. Distance Traveled", fontsize=11, fontweight="bold")
        self.ax_cov.grid(True, linestyle=":", alpha=0.5)
        self.ax_cov.legend(loc="lower right", fontsize=9)

        self.fig.tight_layout()
        self.fig.canvas.draw()

        rgba = np.asarray(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgb_frame = rgba[:, :, :3].copy()

        if record:
            self.frames.append(rgb_frame)
        return rgb_frame

    def render_all_frames(self, delta_d: float | None = None) -> list[np.ndarray]:
        """Render all distance-synchronized frames at uniform ``delta_d`` increments."""
        self.frames.clear()
        distances = self.get_synchronized_distances(delta_d=delta_d)
        for d in distances:
            self.render_frame(target_distance=float(d), record=True)
        return self.frames

    def save_animation(
        self,
        output_path: str | Path,
        frames: list[np.ndarray] | None = None,
        fps: int = 10,
    ) -> Path:
        """Compile rendered race frames into an `.mp4` or `.gif` animation file."""
        frames_to_save = frames if frames is not None else self.frames
        if not frames_to_save:
            frames_to_save = self.render_all_frames()

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()

        h_px, w_px, _ = frames_to_save[0].shape
        anim_fig, anim_ax = plt.subplots(
            figsize=(w_px / self.dpi, h_px / self.dpi), dpi=self.dpi
        )
        anim_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        anim_ax.axis("off")
        im = anim_ax.imshow(frames_to_save[0])

        def _update(idx: int) -> list[Any]:
            im.set_data(frames_to_save[idx])
            return [im]

        anim = matplotlib.animation.FuncAnimation(
            anim_fig,
            _update,
            frames=len(frames_to_save),
            interval=max(1, int(1000 / max(1, fps))),
            blit=True,
        )

        try:
            if suffix == ".gif":
                writer = matplotlib.animation.PillowWriter(fps=fps)
                anim.save(str(out_path), writer=writer, dpi=self.dpi)
            elif suffix == ".mp4":
                if matplotlib.animation.FFMpegWriter.isAvailable():
                    writer = matplotlib.animation.FFMpegWriter(fps=fps)
                    anim.save(str(out_path), writer=writer, dpi=self.dpi)
                else:
                    gif_fallback = out_path.with_suffix(".gif")
                    writer = matplotlib.animation.PillowWriter(fps=fps)
                    anim.save(str(gif_fallback), writer=writer, dpi=self.dpi)
                    if not out_path.exists():
                        out_path.write_bytes(gif_fallback.read_bytes())
            else:
                writer = matplotlib.animation.PillowWriter(fps=fps)
                anim.save(str(out_path), writer=writer, dpi=self.dpi)
        finally:
            plt.close(anim_fig)

        return out_path

    def close(self) -> None:
        """Close the underlying Matplotlib figure."""
        plt.close(self.fig)
