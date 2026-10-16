"""Central dataclass configuration for GAT-FTO."""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    grid_height: int = 100
    grid_width: int = 100
    sensor_range: int = 20
    sensor_fov_deg: float = 360.0      # Supports 360.0 (omni) or <360.0 (directional cone)
    min_frontier_size: int = 3
    max_cluster_size: int = 20
    macro_step_horizon: int = 15       # Max grid steps per RL decision epoch
    coverage_target: float = 0.98      # Terminal coverage fraction
    max_episode_steps: int = 150       # Max macro-steps before timeout
    w_cov: float = 10.0                # Reward weight: uncovered cells fraction
    w_dist: float = 2.0                # Penalty weight: normalized distance traveled
    w_turn: float = 0.2                # Penalty weight: initial heading rotation
    r_success: float = 5.0             # Terminal bonus upon reaching coverage_target
    r_timeout: float = -2.0            # Terminal penalty on episode timeout


@dataclass
class ModelConfig:
    node_feat_dim: int = 8
    edge_feat_dim: int = 4
    hidden_dim: int = 128
    num_gat_layers: int = 3
    num_heads: int = 4
    logit_clip: float = 10.0
    dropout: float = 0.0


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 512
    batch_size: int = 64
    ppo_epochs: int = 4
