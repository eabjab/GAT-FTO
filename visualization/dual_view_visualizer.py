"""Mode 1: Dual-View Graph & Environment Inspector (SPECS.md Section 3.6.1)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Polygon
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from envs.frontier_extractor import FrontierNode
from envs.graph_builder import GraphBuilder
from envs.pathfinding import astar_path
from models.pointer_actor_critic import PointerActorCritic


def _belief_map_to_rgb(belief_map: np.ndarray) -> np.ndarray:
    """Convert a 2D occupancy belief grid (-1, 0, 1) to an RGB uint8 image.

    Color mapping (SPECS.md Section 3.6.1):
    - ``-1`` (Unknown): dark gray ``#2b2b2b`` -> (43, 43, 43)
    - ``0`` (Free): white ``#ffffff`` -> (255, 255, 255)
    - ``1`` (Obstacle): black ``#000000`` -> (0, 0, 0)
    """
    h, w = belief_map.shape
    rgb = np.full((h, w, 3), 43, dtype=np.uint8)  # Default -1 -> #2b2b2b
    free_mask = belief_map == 0
    obs_mask = belief_map == 1
    rgb[free_mask] = (255, 255, 255)  # 0 -> #ffffff
    rgb[obs_mask] = (0, 0, 0)  # 1 -> #000000
    return rgb


def _create_pose_triangle(
    row: float,
    col: float,
    theta: float,
    scale: float = 1.8,
    facecolor: str = "#1f77b4",
    edgecolor: str = "#ffffff",
    zorder: int = 8,
) -> Polygon:
    """Create a directional triangle patch at (x=col, y=row) oriented at angle theta."""
    # Note: theta is atan2(d_row, d_col), so in (x=col, y=row) space:
    # forward unit vector is (cos(theta), sin(theta))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # Perpendicular vector
    perp_x = -sin_t
    perp_y = cos_t

    tip_x = col + scale * 1.3 * cos_t
    tip_y = row + scale * 1.3 * sin_t

    base_cx = col - scale * 0.75 * cos_t
    base_cy = row - scale * 0.75 * sin_t

    half_w = scale * 0.75
    left_x = base_cx + half_w * perp_x
    left_y = base_cy + half_w * perp_y
    right_x = base_cx - half_w * perp_x
    right_y = base_cy - half_w * perp_y

    verts = np.array([[tip_x, tip_y], [left_x, left_y], [right_x, right_y]], dtype=np.float64)
    return Polygon(
        verts,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        zorder=zorder,
    )


class DualViewVisualizer:
    """Dual-View Graph & Environment Inspector (SPECS.md Section 3.6.1).

    Generates a 1x2 side-by-side figure (figsize=(14, 6), 150 DPI) at every macro-step:
    - Left subplot (`ax_env` — Environment Exploration):
      - Renders `belief_map` (`-1` = `#2b2b2b`, `0` = `#ffffff`, `1` = `#000000`).
      - Plots cumulative historical trajectory as a continuous blue line (`#1f77b4`)
        and current agent pose as a directional triangle.
      - Plots active frontier clusters as scatter points colored by cluster index and
        draws the A* shortest path to the chosen target frontier in dashed crimson (`#d62728`).
    - Right subplot (`ax_graph` — Topological Graph & GAT Attention):
      - Plots nodes at their exact `(c, r)` spatial coordinates overlaid on a faint
        wireframe of the map boundaries.
      - Extracts `(edge_index_attn, alpha)` from the final `GATv2Conv` layer (`alpha`
        averaged across the 4 attention heads).
      - Renders directed graph edges with line thickness `w_ij = 0.5 + 4.0 * alpha_ij`
        and opacity `alpha_ij in [0.15, 1.0]`.
      - Colors Frontier Nodes using the `plasma` colormap mapped to the Actor's
        probability distribution, and highlights the selected frontier node with a
        bold gold outer ring.
    """

    def __init__(
        self,
        model: PointerActorCritic | Any | None = None,
        figsize: tuple[float, float] = (14.0, 6.0),
        dpi: int = 150,
    ) -> None:
        self.model = model
        self.figsize = (float(figsize[0]), float(figsize[1]))
        self.dpi = int(dpi)
        self._fallback_model: PointerActorCritic | None = None
        self._graph_builder = GraphBuilder()
        self.frames: list[np.ndarray] = []

        self.fig, (self.ax_env, self.ax_graph) = plt.subplots(
            1, 2, figsize=self.figsize, dpi=self.dpi
        )

    def _get_inference_model(self) -> PointerActorCritic:
        """Return the configured PointerActorCritic model or a lazy default instance."""
        if self.model is not None:
            if hasattr(self.model, "model") and isinstance(self.model.model, torch.nn.Module):
                return self.model.model  # Unwrap LearnedPolicyAgent
            if isinstance(self.model, torch.nn.Module):
                return self.model
        if self._fallback_model is None:
            self._fallback_model = PointerActorCritic()
            self._fallback_model.eval()
        return self._fallback_model

    def _extract_inputs(
        self,
        env_or_belief: Any = None,
        *,
        env: Any = None,
        belief_map: np.ndarray | None = None,
        agent_pose: tuple[int, int, float] | None = None,
        trajectory: list[tuple[int, int]] | np.ndarray | None = None,
        frontiers: list[FrontierNode] | None = None,
        planned_path: list[tuple[int, int]] | np.ndarray | None = None,
        graph_data: Data | None = None,
        model: PointerActorCritic | Any | None = None,
        probs: torch.Tensor | np.ndarray | None = None,
        attn_tuple: tuple[torch.Tensor, torch.Tensor] | None = None,
        selected_action: int | None = None,
        action_idx: int | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Normalize flexible positional/keyword arguments into a canonical frame dictionary."""
        if selected_action is None and action_idx is not None:
            selected_action = int(action_idx)

        # Case 1: first positional arg is a FrontierExplorationEnv-like object
        if env_or_belief is not None and hasattr(env_or_belief, "belief_map"):
            env = env_or_belief
        # Case 2: first positional arg is a macro_step_history dict
        elif isinstance(env_or_belief, dict):
            rec = env_or_belief
            if belief_map is None:
                belief_map = rec.get("belief_map", rec.get("belief"))
            if agent_pose is None:
                agent_pose = rec.get("agent_pose", rec.get("pose"))
            if trajectory is None:
                trajectory = rec.get("trajectory")
            if frontiers is None:
                frontiers = rec.get("frontiers")
            if planned_path is None:
                planned_path = rec.get("planned_path")
            if graph_data is None:
                graph_data = rec.get("graph", rec.get("graph_data"))
            if selected_action is None:
                selected_action = rec.get("action_idx", rec.get("selected_action"))
            if step is None:
                step = rec.get("step")
        # Case 3: first positional arg is a raw numpy belief_map array
        elif isinstance(env_or_belief, np.ndarray) and belief_map is None:
            belief_map = env_or_belief

        if env is not None:
            if belief_map is None:
                belief_map = env.belief_map
            if agent_pose is None:
                agent_pose = env.agent_pose
            if trajectory is None:
                trajectory = getattr(env, "trajectory", [(agent_pose[0], agent_pose[1])])
            if frontiers is None:
                frontiers = getattr(env, "frontiers", [])
            if planned_path is None:
                planned_path = getattr(env, "last_planned_path", None)
            if graph_data is None:
                graph_data = getattr(env, "current_graph", None)
            if selected_action is None:
                selected_action = getattr(env, "last_action_idx", None)
            if step is None:
                step = getattr(env, "step_count", 0)

        if belief_map is None:
            raise ValueError("DualViewVisualizer.render_frame requires `env` or `belief_map`.")

        belief_map = np.asarray(belief_map, dtype=np.int8)
        h, w = belief_map.shape

        if agent_pose is None:
            agent_pose = (h // 2, w // 2, 0.0)
        else:
            agent_pose = (
                int(agent_pose[0]),
                int(agent_pose[1]),
                float(agent_pose[2]) if len(agent_pose) >= 3 else 0.0,
            )

        if trajectory is None or len(trajectory) == 0:
            trajectory_list = [(agent_pose[0], agent_pose[1])]
        else:
            trajectory_list = [(int(p[0]), int(p[1])) for p in trajectory]

        frontiers_list = list(frontiers) if frontiers is not None else []

        if graph_data is None:
            graph_data = self._graph_builder.build_graph(
                agent_pose, frontiers_list, belief_map
            )

        # Run model forward pass if probs or attn_tuple are missing
        active_model = model if model is not None else self.model
        if probs is None or attn_tuple is None:
            net = (
                active_model.model
                if (active_model is not None and hasattr(active_model, "model"))
                else (active_model if isinstance(active_model, torch.nn.Module) else self._get_inference_model())
            )
            net.eval()
            with torch.no_grad():
                param_device = next(net.parameters()).device
                g_dev = graph_data.to(param_device) if graph_data.x.device != param_device else graph_data
                probs_out, _, _, _, attn_out = net(g_dev, return_attn=True)
                if probs is None:
                    probs = probs_out.detach().cpu().numpy()
                if attn_tuple is None:
                    attn_tuple = (
                        attn_out[0].detach().cpu(),
                        attn_out[1].detach().cpu(),
                    )

        if isinstance(probs, torch.Tensor):
            probs_np = probs.detach().cpu().numpy().astype(np.float32)
        else:
            probs_np = np.asarray(probs, dtype=np.float32)

        num_frontiers = len(frontiers_list)
        if selected_action is None and probs_np.size > 0:
            selected_action = int(np.argmax(probs_np))
        elif selected_action is not None and num_frontiers > 0:
            selected_action = max(0, min(num_frontiers - 1, int(selected_action)))

        # Compute A* shortest path to selected frontier if not already provided or trivial
        if (
            (planned_path is None or len(planned_path) <= 1)
            and selected_action is not None
            and 0 <= selected_action < num_frontiers
        ):
            target_pos = frontiers_list[selected_action].position
            computed_path, _ = astar_path(
                belief_map,
                (agent_pose[0], agent_pose[1]),
                (int(target_pos[0]), int(target_pos[1])),
                free_val=0,
            )
            if computed_path:
                planned_path = computed_path

        planned_path_list = (
            [(int(p[0]), int(p[1])) for p in planned_path]
            if planned_path is not None
            else []
        )

        return {
            "belief_map": belief_map,
            "agent_pose": agent_pose,
            "trajectory": trajectory_list,
            "frontiers": frontiers_list,
            "planned_path": planned_path_list,
            "graph_data": graph_data,
            "probs": probs_np,
            "attn_tuple": attn_tuple,
            "selected_action": selected_action,
            "step": step,
        }

    def render_frame(
        self,
        env_or_belief: Any = None,
        *,
        env: Any = None,
        belief_map: np.ndarray | None = None,
        agent_pose: tuple[int, int, float] | None = None,
        trajectory: list[tuple[int, int]] | np.ndarray | None = None,
        frontiers: list[FrontierNode] | None = None,
        planned_path: list[tuple[int, int]] | np.ndarray | None = None,
        graph_data: Data | None = None,
        model: PointerActorCritic | Any | None = None,
        probs: torch.Tensor | np.ndarray | None = None,
        attn_tuple: tuple[torch.Tensor, torch.Tensor] | None = None,
        selected_action: int | None = None,
        action_idx: int | None = None,
        step: int | None = None,
        record: bool = True,
    ) -> np.ndarray:
        """Render a single Dual-View inspection frame and return an RGB uint8 array (H_px, W_px, 3)."""
        state = self._extract_inputs(
            env_or_belief,
            env=env,
            belief_map=belief_map,
            agent_pose=agent_pose,
            trajectory=trajectory,
            frontiers=frontiers,
            planned_path=planned_path,
            graph_data=graph_data,
            model=model,
            probs=probs,
            attn_tuple=attn_tuple,
            selected_action=selected_action,
            action_idx=action_idx,
            step=step,
        )

        bmap: np.ndarray = state["belief_map"]
        pose: tuple[int, int, float] = state["agent_pose"]
        traj: list[tuple[int, int]] = state["trajectory"]
        frontiers_list: list[FrontierNode] = state["frontiers"]
        path_list: list[tuple[int, int]] = state["planned_path"]
        g_data: Data = state["graph_data"]
        probs_np: np.ndarray = state["probs"]
        attn_pair: tuple[torch.Tensor, torch.Tensor] | None = state["attn_tuple"]
        sel_idx: int | None = state["selected_action"]
        step_num: int | None = state["step"]

        h, w = bmap.shape
        scale = max(1.0, min(h, w) / 40.0)

        self.ax_env.clear()
        self.ax_graph.clear()

        # =====================================================================
        # Left Subplot: ax_env (Environment Exploration)
        # =====================================================================
        rgb_map = _belief_map_to_rgb(bmap)
        self.ax_env.imshow(rgb_map, origin="upper", interpolation="nearest")

        # Plot active frontier clusters colored by cluster index
        cluster_cmap = matplotlib.colormaps["tab10"]
        for c_idx, f_node in enumerate(frontiers_list):
            color = cluster_cmap(c_idx % 10)
            if f_node.cells is not None and f_node.cells.size > 0:
                self.ax_env.scatter(
                    f_node.cells[:, 1],
                    f_node.cells[:, 0],
                    c=[color],
                    s=18.0 * scale,
                    alpha=0.85,
                    edgecolors="none",
                    zorder=4,
                )
            # Also plot cluster medoid
            self.ax_env.scatter(
                [f_node.position[1]],
                [f_node.position[0]],
                c=[color],
                s=36.0 * scale,
                edgecolors="#000000",
                linewidths=0.8,
                zorder=5,
            )

        # Plot cumulative historical trajectory as continuous blue line (#1f77b4)
        if len(traj) >= 2:
            traj_cols = [p[1] for p in traj]
            traj_rows = [p[0] for p in traj]
            self.ax_env.plot(
                traj_cols,
                traj_rows,
                color="#1f77b4",
                linewidth=2.0,
                alpha=0.9,
                zorder=6,
                label="Trajectory",
            )

        # Plot A* shortest path to chosen target frontier in dashed crimson (#d62728)
        if len(path_list) >= 2:
            path_cols = [p[1] for p in path_list]
            path_rows = [p[0] for p in path_list]
            self.ax_env.plot(
                path_cols,
                path_rows,
                color="#d62728",
                linestyle="--",
                linewidth=2.2,
                zorder=7,
                label="A* Target Path",
            )

        # Plot current agent pose as a directional triangle
        pose_patch = _create_pose_triangle(
            row=float(pose[0]),
            col=float(pose[1]),
            theta=float(pose[2]),
            scale=1.6 * scale,
            facecolor="#1f77b4",
            edgecolor="#ffffff",
            zorder=8,
        )
        self.ax_env.add_patch(pose_patch)

        step_str = f" (Step {step_num})" if step_num is not None else ""
        self.ax_env.set_title(f"Environment Exploration & Belief Map{step_str}", fontsize=11, fontweight="bold")
        self.ax_env.set_xlim(-0.5, w - 0.5)
        self.ax_env.set_ylim(h - 0.5, -0.5)
        self.ax_env.set_aspect("equal")
        self.ax_env.set_xlabel("Grid Column (c)")
        self.ax_env.set_ylabel("Grid Row (r)")

        # =====================================================================
        # Right Subplot: ax_graph (Topological Graph & GAT Attention)
        # =====================================================================
        self.ax_graph.set_facecolor("#1e1e1e")

        # Faint wireframe of map boundaries and known obstacles
        border_x = [0.0, float(w - 1), float(w - 1), 0.0, 0.0]
        border_y = [0.0, 0.0, float(h - 1), float(h - 1), 0.0]
        self.ax_graph.plot(
            border_x,
            border_y,
            color="#777777",
            linestyle="-",
            linewidth=1.2,
            alpha=0.5,
            zorder=1,
        )
        obs_rows, obs_cols = np.where(bmap == 1)
        if obs_rows.size > 0:
            self.ax_graph.scatter(
                obs_cols,
                obs_rows,
                c="#555555",
                s=4.0 * scale,
                marker="s",
                alpha=0.25,
                edgecolors="none",
                zorder=1,
            )

        # Extract node (r, c) positions -> plot at exact (c, r) spatial coordinates
        if hasattr(g_data, "node_positions") and g_data.node_positions is not None:
            node_rc = g_data.node_positions.detach().cpu().numpy()
        else:
            coords = [(float(pose[0]), float(pose[1]))]
            for f_node in frontiers_list:
                coords.append((float(f_node.position[0]), float(f_node.position[1])))
            node_rc = np.asarray(coords, dtype=np.float32)

        num_nodes = node_rc.shape[0]
        node_xy = np.column_stack([node_rc[:, 1], node_rc[:, 0]])  # (c, r)

        # Render directed graph edges with GATv2 attention weights averaged across heads
        if attn_pair is not None and num_nodes > 1:
            edge_index_attn, alpha_tensor = attn_pair
            ei_np = edge_index_attn.detach().cpu().numpy()
            alpha_np = alpha_tensor.detach().cpu().numpy()
            if alpha_np.ndim == 2:
                alpha_mean = alpha_np.mean(axis=-1)
            else:
                alpha_mean = alpha_np.reshape(-1)

            segments: list[np.ndarray] = []
            linewidths: list[float] = []
            edge_colors: list[tuple[float, float, float, float]] = []

            # Filter valid non-self-loop edges first and normalize for visual contrast
            valid_mask = (ei_np[0] != ei_np[1]) & (ei_np[0] < num_nodes) & (ei_np[1] < num_nodes)
            max_alpha = float(np.max(alpha_mean[valid_mask])) if np.any(valid_mask) else 1.0
            max_alpha = max(max_alpha, 1e-6)

            for e_idx in range(ei_np.shape[1]):
                u = int(ei_np[0, e_idx])
                v = int(ei_np[1, e_idx])
                if u == v or u >= num_nodes or v >= num_nodes:
                    continue

                # a_ij = float(np.clip(alpha_mean[e_idx], 0.0, 1.0))
                a_ij = float(np.clip(alpha_mean[e_idx] / max_alpha, 0.0, 1.0))
                w_ij = 0.5 + 4.0 * a_ij
                opacity_ij = float(np.clip(a_ij, 0.15, 1.0))

                p_u = node_xy[u]
                p_v = node_xy[v]
                segments.append(np.array([p_u, p_v], dtype=np.float64))
                linewidths.append(w_ij)
                edge_colors.append((0.65, 0.85, 1.0, opacity_ij))

                # Add subtle directional arrow for high-attention edges from/to agent or top edges
                if a_ij >= 0.25 or u == 0:
                    arrow = FancyArrowPatch(
                        posA=(float(p_u[0]), float(p_u[1])),
                        posB=(float(p_v[0]), float(p_v[1])),
                        arrowstyle="-|>",
                        mutation_scale=8.0 + 6.0 * a_ij,
                        linewidth=w_ij * 0.6,
                        color=(0.65, 0.85, 1.0, opacity_ij),
                        zorder=3,
                    )
                    self.ax_graph.add_patch(arrow)

            if segments:
                lc = LineCollection(
                    segments,
                    linewidths=linewidths,
                    colors=edge_colors,
                    zorder=2,
                )
                self.ax_graph.add_collection(lc)

        # Plot Agent Node (Node 0) at (c_0, r_0)
        if num_nodes >= 1:
            self.ax_graph.scatter(
                [node_xy[0, 0]],
                [node_xy[0, 1]],
                c="#1f77b4",
                s=130.0 * scale,
                marker="^",
                edgecolors="#ffffff",
                linewidths=1.5,
                zorder=6,
                label="Agent Node (v_0)",
            )

        # Plot Frontier Nodes (Nodes 1..K_t) colored using `plasma` colormap mapped to Actor probs
        num_frontier_nodes = max(0, num_nodes - 1)
        if num_frontier_nodes > 0:
            frontier_xy = node_xy[1:]
            if probs_np.size == num_frontier_nodes:
                node_probs = probs_np
            elif probs_np.size > 0:
                node_probs = np.resize(probs_np, (num_frontier_nodes,))
            else:
                node_probs = np.full((num_frontier_nodes,), 1.0 / num_frontier_nodes, dtype=np.float32)

            vmax = float(max(np.max(node_probs), 1e-4))
            norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            plasma_cmap = matplotlib.colormaps["plasma"]
            frontier_colors = plasma_cmap(norm(node_probs))

            self.ax_graph.scatter(
                frontier_xy[:, 0],
                frontier_xy[:, 1],
                c=frontier_colors,
                s=95.0 * scale,
                edgecolors="#ffffff",
                linewidths=1.0,
                zorder=5,
                label="Frontier Nodes",
            )

            # Highlight the selected frontier node with a bold gold outer ring
            if sel_idx is not None and 0 <= sel_idx < num_frontier_nodes:
                sel_xy = frontier_xy[sel_idx]
                self.ax_graph.scatter(
                    [sel_xy[0]],
                    [sel_xy[1]],
                    facecolors="none",
                    edgecolors="#ffd700",
                    s=220.0 * scale,
                    linewidths=3.0,
                    zorder=7,
                    label="Selected Frontier",
                )

        self.ax_graph.set_title(
            "Topological Graph & GATv2 Attention (plasma = Actor Probs)",
            fontsize=11,
            fontweight="bold",
        )
        self.ax_graph.set_xlim(-0.5, w - 0.5)
        self.ax_graph.set_ylim(h - 0.5, -0.5)
        self.ax_graph.set_aspect("equal")
        self.ax_graph.set_xlabel("Grid Column (c)")
        self.ax_graph.set_ylabel("Grid Row (r)")

        self.fig.tight_layout()
        self.fig.canvas.draw()

        rgba = np.asarray(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgb_frame = rgba[:, :, :3].copy()

        if record:
            self.frames.append(rgb_frame)
        return rgb_frame

    def save_animation(
        self,
        output_path: str | Path,
        frames: list[np.ndarray] | None = None,
        fps: int = 4,
    ) -> Path:
        """Compile rendered frames into an `.mp4` or `.gif` animation file."""
        frames_to_save = frames if frames is not None else self.frames
        if not frames_to_save:
            raise ValueError("No frames available to save. Call render_frame() first.")

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
                    # Fallback to PillowWriter GIF stream if system ffmpeg binary is absent
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
