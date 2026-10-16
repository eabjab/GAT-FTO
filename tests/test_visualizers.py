"""Smoke verification suite for DualViewVisualizer and RaceVisualizer (SPECS.md Section 4.6)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from baselines.classical_agents import NearestFrontierAgent
from config import EnvConfig, ModelConfig
from envs.exploration_env import FrontierExplorationEnv
from models.gat_encoder import GATEncoder
from models.pointer_actor_critic import PointerActorCritic
from visualization.dual_view_visualizer import DualViewVisualizer
from visualization.race_visualizer import RaceVisualizer


def test_dual_view_and_race_headless_render(tmp_path: Path) -> None:
    """Run a 3-step rollout on a 40x40 map and verify headless rendering for both visualizers."""
    matplotlib.use("Agg")
    torch.manual_seed(42)
    np.random.seed(42)

    env_cfg = EnvConfig(
        grid_height=40,
        grid_width=40,
        sensor_range=15,
        macro_step_horizon=10,
        coverage_target=0.98,
        max_episode_steps=20,
    )
    model_cfg = ModelConfig(
        node_feat_dim=8,
        edge_feat_dim=4,
        hidden_dim=128,
        num_gat_layers=3,
        num_heads=4,
        logit_clip=10.0,
        dropout=0.0,
    )
    model = PointerActorCritic(GATEncoder(model_cfg), config=model_cfg)
    model.eval()
    nearest_agent = NearestFrontierAgent()

    # 1. Run a 3-step rollout on a 40x40 map for the GAT policy and render DualView frames
    env_gat = FrontierExplorationEnv(env_cfg)
    obs_gat, _ = env_gat.reset(seed=7)

    dual_viz = DualViewVisualizer(model=model, figsize=(14, 6), dpi=150)

    for _ in range(3):
        with torch.no_grad():
            probs, _, _, _, attn_tuple = model(obs_gat, return_attn=True)
        action = int(torch.argmax(probs).item()) if probs.numel() > 0 else 0
        obs_gat, _, terminated, truncated, _ = env_gat.step(action)

        frame_dual = dual_viz.render_frame(
            env=env_gat,
            probs=probs,
            attn_tuple=attn_tuple,
            selected_action=action,
        )
        assert isinstance(frame_dual, np.ndarray)
        assert frame_dual.ndim == 3
        assert frame_dual.shape[2] == 3
        assert frame_dual.shape == (900, 2100, 3)
        assert frame_dual.dtype == np.uint8
        assert int(frame_dual.max()) > int(frame_dual.min())

        if terminated or truncated:
            break

    # Verify DualView animation export (.gif)
    gif_dual_path = dual_viz.save_animation(tmp_path / "dual_view_smoke.gif", fps=4)
    assert gif_dual_path.exists() and gif_dual_path.stat().st_size > 0
    dual_viz.close()

    # 2. Run a 3-step rollout on the exact same 40x40 map seed for NearestFrontierAgent
    env_nearest = FrontierExplorationEnv(env_cfg)
    obs_nearest, _ = env_nearest.reset(seed=7)
    for _ in range(3):
        act_n = nearest_agent.select_action(obs_nearest, deterministic=True)
        obs_nearest, _, term_n, trunc_n, _ = env_nearest.step(act_n)
        if term_n or trunc_n:
            break

    # 3. Invoke RaceVisualizer.render_frame() with distance-synchronized interpolation (delta_d = 2.0)
    race_viz = RaceVisualizer(
        nearest_source=env_nearest,
        gat_source=env_gat,
        delta_d=2.0,
        figsize=(14, 8),
        dpi=150,
    )
    sync_dists = race_viz.get_synchronized_distances(delta_d=2.0)
    assert sync_dists.size >= 1
    if sync_dists.size >= 3:
        assert np.isclose(sync_dists[1] - sync_dists[0], 2.0)

    frame_race_start = race_viz.render_frame(target_distance=0.0)
    frame_race_end = race_viz.render_frame(target_distance=float(sync_dists[-1]))

    for frame_race in (frame_race_start, frame_race_end):
        assert isinstance(frame_race, np.ndarray)
        assert frame_race.ndim == 3
        assert frame_race.shape[2] == 3
        assert frame_race.shape == (1200, 2100, 3)
        assert frame_race.dtype == np.uint8
        assert int(frame_race.max()) > int(frame_race.min())

    # Verify RaceVisualizer animation export (.gif)
    gif_race_path = race_viz.save_animation(tmp_path / "race_smoke.gif", fps=8)
    assert gif_race_path.exists() and gif_race_path.stat().st_size > 0
    race_viz.close()
