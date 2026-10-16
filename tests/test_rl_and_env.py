"""Integration verification suite for FrontierExplorationEnv and PPO training (SPECS.md Section 4.5)."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from baselines.classical_agents import (
    LearnedPolicyAgent,
    MyopicTSPAgent,
    NearestFrontierAgent,
    UtilityGreedyAgent,
)
from config import EnvConfig, ModelConfig, PPOConfig
from envs.exploration_env import FrontierExplorationEnv
from models.gat_encoder import GATEncoder
from models.pointer_actor_critic import PointerActorCritic
from rl.buffer import RolloutBuffer
from rl.ppo_trainer import PPOTrainer


def _make_distinct_synthetic_graph(num_frontiers: int = 4) -> Data:
    """Construct a fixed synthetic graph with distinct frontier node & edge features."""
    num_nodes = num_frontiers + 1

    # Node features: shape (5, 8) in [-1.0, 1.0]
    x = torch.zeros((num_nodes, 8), dtype=torch.float32)
    # Node 0: Agent Pose Node
    x[0] = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.0], dtype=torch.float32)

    # Nodes 1..4: 4 distinct Frontier Nodes (corresponding to actions 0, 1, 2, 3)
    x[1] = torch.tensor([0.0, -0.6, -0.5, 1.0, 0.0, 0.25, 0.3, 0.20], dtype=torch.float32)
    x[2] = torch.tensor([0.0, -0.4, 0.6, 0.0, 1.0, 0.40, 0.5, 0.35], dtype=torch.float32)
    x[3] = torch.tensor([0.0, 0.7, -0.3, -1.0, 0.0, 0.90, 0.9, 0.95], dtype=torch.float32)  # Action 2
    x[4] = torch.tensor([0.0, 0.5, 0.5, 0.0, -1.0, 0.15, 0.2, 0.10], dtype=torch.float32)

    # Complete directed graph without self-loops
    src_list: list[int] = []
    dst_list: list[int] = []
    for u in range(num_nodes):
        for v in range(num_nodes):
            if u != v:
                src_list.append(u)
                dst_list.append(v)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Direct agent-to-frontier edges (shape: (4, 4))
    agent_to_frontier_edge_attr = torch.tensor(
        [
            [0.80, 0.78, 0.50, 0.40],  # Action 0
            [0.72, 0.70, 0.45, 0.35],  # Action 1
            [0.15, 0.14, 0.05, 0.05],  # Action 2 (distinct low distance & turn angle)
            [0.95, 0.90, 0.80, 0.70],  # Action 3
        ],
        dtype=torch.float32,
    )

    edge_attr_list: list[torch.Tensor] = []
    for u, v in zip(src_list, dst_list):
        if u == 0 and v >= 1:
            edge_attr_list.append(agent_to_frontier_edge_attr[v - 1])
        else:
            dist_val = 0.2 + 0.1 * abs(u - v)
            edge_attr_list.append(
                torch.tensor([dist_val, dist_val * 0.9, 0.2, 0.2], dtype=torch.float32)
            )
    edge_attr = torch.stack(edge_attr_list, dim=0)

    is_frontier = torch.tensor([False, True, True, True, True], dtype=torch.bool)
    node_positions = torch.tensor(
        [
            [50.0, 50.0],
            [20.0, 25.0],
            [30.0, 80.0],
            [85.0, 35.0],
            [75.0, 75.0],
        ],
        dtype=torch.float32,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        agent_to_frontier_edge_attr=agent_to_frontier_edge_attr,
        is_frontier=is_frontier,
        node_positions=node_positions,
    )


def test_env_episode_completion() -> None:
    """Run NearestFrontierAgent on a 50x50 map until terminated or truncated.

    Verifies:
    - Coverage strictly increases on steps where frontiers are reached.
    - No unhandled exceptions occur when K_t drops to 0.
    - Primitive and macro-step history records are populated consistently.
    """
    cfg = EnvConfig(
        grid_height=50,
        grid_width=50,
        sensor_range=20,
        macro_step_horizon=15,
        coverage_target=0.98,
        max_episode_steps=100,
    )
    env = FrontierExplorationEnv(cfg)
    agent = NearestFrontierAgent()

    obs, info = env.reset(seed=42)
    assert isinstance(obs, Data)
    assert 0.0 < info["coverage"] <= 1.0
    assert len(env.primitive_history) == 1

    terminated = False
    truncated = False
    steps_taken = 0
    frontiers_reached_count = 0

    while not (terminated or truncated):
        prev_coverage = info["coverage"]
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps_taken += 1

        assert isinstance(obs, Data)
        assert isinstance(reward, float)
        assert not np.isnan(reward)

        if info["reached_target"]:
            assert info["coverage"] > prev_coverage
            frontiers_reached_count += 1

    assert steps_taken > 0
    assert frontiers_reached_count > 0
    assert len(env.primitive_history) >= 2
    assert env.total_distance > 0.0

    # Explicitly test graceful handling when K_t drops to 0
    env.frontiers = []
    empty_graph = env.graph_builder.build_graph(env.agent_pose, env.frontiers, env.belief_map)
    assert int(empty_graph.is_frontier.sum().item()) == 0

    for baseline_agent in (
        NearestFrontierAgent(),
        UtilityGreedyAgent(),
        MyopicTSPAgent(),
        LearnedPolicyAgent(),
    ):
        act_empty = baseline_agent.select_action(empty_graph)
        assert act_empty == 0

    obs_zero, rew_zero, term_zero, trunc_zero, info_zero = env.step(0)
    assert term_zero is True
    assert trunc_zero is False
    assert info_zero["num_frontiers"] == 0
    assert int(obs_zero.is_frontier.sum().item()) == 0


def test_single_graph_ppo_overfit() -> None:
    """Overfit PPO on a single synthetic graph with K_t = 4 frontiers where action 2 gives +1.0 and others -1.0."""
    torch.manual_seed(42)
    np.random.seed(42)

    graph = _make_distinct_synthetic_graph(num_frontiers=4)

    model_cfg = ModelConfig(
        node_feat_dim=8,
        edge_feat_dim=4,
        hidden_dim=128,
        num_gat_layers=3,
        num_heads=4,
        logit_clip=10.0,
        dropout=0.0,
    )
    ppo_cfg = PPOConfig(
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.0,
        value_coef=0.5,
        max_grad_norm=0.5,
        rollout_steps=16,
        batch_size=16,
        ppo_epochs=4,
    )

    encoder = GATEncoder(model_cfg)
    model = PointerActorCritic(
        encoder=encoder,
        hidden_dim=model_cfg.hidden_dim,
        logit_clip=model_cfg.logit_clip,
    )
    trainer = PPOTrainer(model=model, config=ppo_cfg, device="cpu")
    buffer = RolloutBuffer()

    for _ in range(30):
        buffer.clear()
        model.eval()
        with torch.no_grad():
            probs, _, _, values, _ = model(graph, return_attn=False)
            val = float(values.squeeze().item())

        # Collect rollout transitions on the single state covering all 4 candidate actions
        for step_idx in range(16):
            if step_idx < 4:
                action = step_idx
                log_prob = float(torch.log(probs[action] + 1e-8).item())
                value = val
            else:
                action, log_prob, value = trainer.select_action_and_value(
                    graph, deterministic=False
                )
            reward = 1.0 if action == 2 else -1.0
            buffer.add(
                graph=graph,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=True,
                value=value,
            )

        buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )
        trainer.update(buffer)

    model.eval()
    with torch.no_grad():
        final_probs, _, _, _, _ = model(graph, return_attn=False)

    prob_action_2 = float(final_probs[2].item())
    assert prob_action_2 > 0.90, f"Expected pi(a=2|G) > 0.90 after 30 PPO updates, got {prob_action_2:.4f}"
