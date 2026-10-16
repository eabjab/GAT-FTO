# GAT-FTO: Graph Attention Networks for Frontier Tour Optimization

**GAT-FTO** is a graph-based reinforcement learning system for autonomous robot exploration in unknown 2D indoor environments. Instead of relying on myopic greedy heuristics that cause frequent backtracking, GAT-FTO frames frontier exploration as a dynamic variant of the Traveling Salesman Problem (TSP) / Orienteering Problem over a topological graph, solved using a multi-head **GATv2 (`GATv2Conv`) encoder** and a **Pointer-Network cross-attention Actor-Critic** trained via **Proximal Policy Optimization (PPO)**.

---

## Key Features

- **Semi-MDP Exploration Environment (`envs/exploration_env.py`):**
  - Formulates exploration as a Semi-Markov Decision Process (`FrontierExplorationEnv`) where each macro-step constructs a topological graph $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t)$, selects a target frontier cluster $a_t \in \{0, \dots, K_t - 1\}$, and navigates along an 8-connected A* shortest path over known free space (`belief_map == 0`) while continuously scanning via simulated 2D raycasting.
  - Supports early macro-step termination when the target medoid is reached, when all cells in the target frontier cluster are uncovered, or when global map coverage reaches `coverage_target`.
- **Procedural Indoor Maps, Raycasting & Frontier Extraction (`envs/`):**
  - **BSP Map Generator (`envs/map_generator.py`):** Generates deterministic 2D indoor room layouts via Binary Space Partitioning (`generate_bsp_map`) with doorway openings, scattered room obstacles, and guaranteed single 4-connected free-space component reachability.
  - **2D Raycasting Sensor (`envs/sensor.py`):** Integer Bresenham line-of-sight sensor (`RaycastingSensor`) supporting $360^\circ$ omnidirectional or directional cone FOV with obstacle occlusion.
  - **Frontier Extractor (`envs/frontier_extractor.py`):** Detects boundary cells between known free space (`0`) and unknown space (`-1`), groups 8-connected clusters, automatically splits large frontiers ($> \text{max\_cluster\_size}$) via $k$-means (`kmeans2`), and extracts free-space medoids, $7 \times 7$ inward viewing normals, EDT obstacle clearance, and visible unmapped area estimates.
  - **Geodesic Topological Graph Builder (`envs/graph_builder.py`):** Constructs `torch_geometric.data.Data` graphs where Node `0` is strictly reserved for the Agent Pose and Nodes `1..K_t` represent active frontiers, connected by directed edges annotated with normalized 8-connected Dijkstra geodesic distances, Euclidean distances, and departure/arrival kinematic turning angles.
- **Variable Action-Space Neural Architecture (`models/`):**
  - **GATv2 Backbone (`models/gat_encoder.py`):** 3-layer, 4-head `GATv2Conv` encoder integrating 8D node features and 4D edge attributes with residual connections, `ELU` activations, `LayerNorm`, and optional attention weight extraction (`return_attention_weights=True`).
  - **Pointer Actor-Critic (`models/pointer_actor_critic.py`):** Cross-attention Pointer Actor head producing a probability distribution over a dynamically varying number of frontiers $K_t$ per step (with $\tanh$ logit clipping and `pyg_softmax` over heterogeneous batches), paired with a global-pooled (`agent + mean + max`) Critic head.
- **Heterogeneous Graph PPO Pipeline (`rl/`):**
  - **Rollout Buffer (`rl/buffer.py`):** Stores variable-size PyG `Data` graphs on CPU alongside transition tensors and computes Generalized Advantage Estimation (GAE).
  - **Vectorized PPO Trainer (`rl/ppo_trainer.py`):** Evaluates variable-length batched frontier actions cleanly in vectorized PyTorch using `torch.bincount` cumulative offsets and `scatter_add` per-graph entropy.
- **Classical Baselines & Quantitative Benchmark (`baselines/`, `evaluate.py`):**
  - Implements `NearestFrontierAgent`, `UtilityGreedyAgent`, `MyopicTSPAgent` (via `networkx.approximation.greedy_tsp`), and `LearnedPolicyAgent` under a unified `BaseExplorationAgent` interface.
  - Evaluates path length to $95\%$ and $98\%$ coverage, exploration efficiency, and backtracking ratio across fixed test seeds (`10000..10099`).
- **Presentation Visualization Suite (`visualization/`):**
  - **Mode 1 (`DualViewVisualizer`):** Side-by-side rendering of the occupancy belief map / A* path alongside the live topological graph with edge thickness and opacity driven by `GATv2Conv` attention weights and node colors mapped to Actor selection probabilities.
  - **Mode 2 (`RaceVisualizer`):** Distance-synchronized ($\Delta d = 2.0$) side-by-side race comparing `NearestFrontierAgent` vs. `PointerActorCritic` on the exact same map seed with a live `% Map Covered vs. Distance Traveled` curve.

---

## Project Layout

```text
GAT-FTO/
├── pyproject.toml
├── SPECS.md                       # Technical specification & mathematical contracts
├── STATE.md                       # Work package execution ledger
├── config.py                      # Central dataclass configuration (EnvConfig, ModelConfig, PPOConfig)
├── envs/
│   ├── __init__.py
│   ├── map_generator.py           # BSP procedural indoor grids + connectivity verification
│   ├── sensor.py                  # 2D Bresenham raycasting & FOV occlusion
│   ├── pathfinding.py             # A* path planner & 8-connected Dijkstra on grids
│   ├── frontier_extractor.py      # Edge detection, clustering, medoids, normals, EDT
│   ├── graph_builder.py           # Constructs PyG Data objects from agent + frontiers
│   └── exploration_env.py         # Gymnasium Semi-MDP environment wrapper
├── models/
│   ├── __init__.py
│   ├── gat_encoder.py             # Multi-head GATv2Conv backbone with edge attributes
│   └── pointer_actor_critic.py    # Cross-attention Pointer Actor & Pooled Critic
├── rl/
│   ├── __init__.py
│   ├── buffer.py                  # Rollout buffer for heterogeneous PyG Data graphs
│   └── ppo_trainer.py             # PPO clipped surrogate update loop & GAE
├── baselines/
│   ├── __init__.py
│   └── classical_agents.py        # NearestFrontier, UtilityGreedy, MyopicTSP, LearnedPolicy
├── visualization/
│   ├── __init__.py
│   ├── dual_view_visualizer.py    # Mode 1: Map exploration + GAT attention graph
│   └── race_visualizer.py         # Mode 2: Synchronized Nearest vs. GAT side-by-side race
├── evaluate.py                    # Quantitative benchmark runner across test seeds
└── tests/
    ├── test_map_generator.py
    ├── test_sensor_and_frontiers.py
    ├── test_graph_builder.py
    ├── test_models.py
    ├── test_rl_and_env.py
    └── test_visualizers.py
```

---

## Installation & Setup

Python $\ge 3.10$ is required. Using the preconfigured virtual environment (`.venv`):

```powershell
# Activate the virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Or install dependencies into a fresh environment
pip install -r requirements.txt
```

---

## Example Usage

### 1. Quickstart: Running an Exploration Episode

You can step through `FrontierExplorationEnv` with any classical baseline (`NearestFrontierAgent`, `UtilityGreedyAgent`, `MyopicTSPAgent`) or a `PointerActorCritic` model:

```python
from config import EnvConfig, ModelConfig
from envs.exploration_env import FrontierExplorationEnv
from baselines.classical_agents import NearestFrontierAgent, UtilityGreedyAgent, MyopicTSPAgent
from models.gat_encoder import GATEncoder
from models.pointer_actor_critic import PointerActorCritic

# Initialize the Semi-MDP environment
env = FrontierExplorationEnv(EnvConfig(grid_height=100, grid_width=100, coverage_target=0.98))
obs_graph, info = env.reset(seed=42)

# Option A: Classical baseline agent
agent = NearestFrontierAgent()
# Option B: GATv2 Pointer Actor-Critic policy
# agent = PointerActorCritic(GATEncoder(ModelConfig()))

terminated = truncated = False
total_reward = 0.0

while not (terminated or truncated):
    action_idx = agent.select_action(obs_graph, deterministic=True)
    obs_graph, reward, terminated, truncated, info = env.step(action_idx)
    total_reward += reward

print(
    f"Finished in {info['step']} macro-steps | "
    f"Coverage: {info['coverage'] * 100:.2f}% | "
    f"Distance: {info['total_distance']:.1f} | "
    f"Return: {total_reward:.2f}"
)
```

---

### 2. Training a GATv2 Pointer Policy with PPO

Use `RolloutBuffer` and `PPOTrainer` to collect variable-size `torch_geometric.data.Data` transitions and perform clipped surrogate PPO updates:

```python
import torch
from config import EnvConfig, ModelConfig, PPOConfig
from envs.exploration_env import FrontierExplorationEnv
from models.gat_encoder import GATEncoder
from models.pointer_actor_critic import PointerActorCritic
from rl.buffer import RolloutBuffer
from rl.ppo_trainer import PPOTrainer

env_cfg = EnvConfig(grid_height=60, grid_width=60, max_episode_steps=100)
model_cfg = ModelConfig()
ppo_cfg = PPOConfig(rollout_steps=128, batch_size=32, ppo_epochs=4, lr=3e-4)

device = "cuda" if torch.cuda.is_available() else "cpu"
env = FrontierExplorationEnv(env_cfg)
model = PointerActorCritic(GATEncoder(model_cfg), config=model_cfg)
trainer = PPOTrainer(model=model, config=ppo_cfg, device=device)
buffer = RolloutBuffer()

obs_graph, _ = env.reset(seed=100)

for update_idx in range(1, 11):
    buffer.clear()
    for _ in range(ppo_cfg.rollout_steps):
        if len(env.frontiers) == 0:
            obs_graph, _ = env.reset()

        action_idx, log_prob, value = trainer.select_action_and_value(
            obs_graph, deterministic=False
        )
        next_graph, reward, terminated, truncated, _ = env.step(action_idx)
        done = terminated or truncated

        buffer.add(
            graph=obs_graph,
            action=action_idx,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
        )
        obs_graph = next_graph if not done else env.reset()[0]

    # Bootstrap final value for GAE
    _, _, last_val = trainer.select_action_and_value(obs_graph, deterministic=True)
    buffer.compute_returns_and_advantages(
        last_value=last_val, gamma=ppo_cfg.gamma, gae_lambda=ppo_cfg.gae_lambda
    )

    metrics = trainer.update(buffer)
    print(
        f"Update {update_idx:02d} | "
        f"Policy Loss: {metrics['policy_loss']:.4f} | "
        f"Value Loss: {metrics['value_loss']:.4f} | "
        f"Entropy: {metrics['entropy']:.4f}"
    )

# Save trained model checkpoint
torch.save(model.state_dict(), "build/gat_pointer_policy.pt")
```

---

### 3. Quantitative Evaluation Benchmark (`evaluate.py`)

`evaluate.py` benchmarks `NearestFrontierAgent`, `UtilityGreedyAgent`, `MyopicTSPAgent`, and `PointerActorCritic` across fixed test seeds (`10000..10099` by default), reporting:
1. **Total Path Length to $95\%$ and $98\%$ Coverage**
2. **Exploration Efficiency** ($\frac{\text{Total Uncovered Free Cells}}{\text{Total Distance Traveled}}$)
3. **Backtracking Ratio** (fraction of primitive steps spent on previously visited cells)

```powershell
# Quick smoke benchmark across 2 seeds
.\.venv\Scripts\python.exe evaluate.py --num-seeds 2 --max-steps 30

# Full 100-seed benchmark (seeds 10000..10099) with a trained checkpoint
.\.venv\Scripts\python.exe evaluate.py `
    --start-seed 10000 `
    --num-seeds 100 `
    --checkpoint build/gat_pointer_policy.pt `
    --output-json build/evaluation_results.json
```

---

### 4. Presentation Visualizations (`visualization/`)

#### Mode 1: Dual-View Graph & Environment Inspector (`DualViewVisualizer`)

Renders a $1 \times 2$ side-by-side view of the occupancy map, trajectory, frontier clusters, and A* path (left) alongside the topological graph with `GATv2Conv` attention edge weights and `plasma`-colored Actor probabilities (right):

```python
import matplotlib
matplotlib.use("Agg")

from config import EnvConfig
from envs.exploration_env import FrontierExplorationEnv
from models.pointer_actor_critic import PointerActorCritic
from visualization.dual_view_visualizer import DualViewVisualizer

env = FrontierExplorationEnv(EnvConfig(grid_height=60, grid_width=60))
model = PointerActorCritic()
viz = DualViewVisualizer(model=model)

obs_graph, _ = env.reset(seed=42)
for _ in range(15):
    action_idx = model.select_action(obs_graph, deterministic=True)
    obs_graph, _, terminated, truncated, _ = env.step(action_idx)
    viz.render_frame(env, selected_action=action_idx, record=True)
    if terminated or truncated:
        break

viz.save_animation("build/dual_view_inspector.gif", fps=4)
viz.close()
```

#### Mode 2: Synchronized Side-by-Side Race (`RaceVisualizer`)

Rolls out `NearestFrontierAgent` and `PointerActorCritic` on the exact same map seed and interpolates both trajectories at uniform distance increments ($\Delta d = 2.0$ grid units) so both agents move at identical speed:

```python
import matplotlib
matplotlib.use("Agg")

from config import EnvConfig
from models.pointer_actor_critic import PointerActorCritic
from visualization.race_visualizer import RaceVisualizer

env_cfg = EnvConfig(grid_height=60, grid_width=60)
model = PointerActorCritic()

race_viz = RaceVisualizer(delta_d=2.0, model=model)
race_viz.rollout_episodes(env_config=env_cfg, seed=42, max_steps=60)
race_viz.render_all_frames()
race_viz.save_animation("build/nearest_vs_gat_race.gif", fps=10)
race_viz.close()
```

---

### 5. Running the Automated Test Suite

The `pytest` verification suite covers map determinism and connectivity, A*/Dijkstra equivalence, Bresenham wall occlusion, frontier medoid invariants and $k$-means cluster splitting, PyG graph tensor contracts, batch invariance, attention extraction, gradient flow, single-graph PPO overfitting, and headless visualizer rendering:

```powershell
# Run the complete test suite
.\.venv\Scripts\pytest.exe -v

# Run individual test modules
.\.venv\Scripts\pytest.exe tests/test_map_generator.py -v
.\.venv\Scripts\pytest.exe tests/test_sensor_and_frontiers.py -v
.\.venv\Scripts\pytest.exe tests/test_graph_builder.py -v
.\.venv\Scripts\pytest.exe tests/test_models.py -v
.\.venv\Scripts\pytest.exe tests/test_rl_and_env.py -v
.\.venv\Scripts\pytest.exe tests/test_visualizers.py -v
```

---

## Configuration Reference (`config.py`)

All environment, model, and PPO hyperparameters are centralized in dataclasses in [`config.py`](file:///d:/Repos/GAT-FTO/config.py):

| Dataclass | Key Parameters & Defaults |
| :--- | :--- |
| **`EnvConfig`** | `grid_height=100`, `grid_width=100`, `sensor_range=20`, `sensor_fov_deg=360.0`, `min_frontier_size=3`, `max_cluster_size=20`, `macro_step_horizon=15`, `coverage_target=0.98`, `max_episode_steps=150`, `w_cov=10.0`, `w_dist=2.0`, `w_turn=0.2`, `r_success=5.0`, `r_timeout=-2.0` |
| **`ModelConfig`** | `node_feat_dim=8`, `edge_feat_dim=4`, `hidden_dim=128`, `num_gat_layers=3`, `num_heads=4`, `logit_clip=10.0`, `dropout=0.0` |
| **`PPOConfig`** | `lr=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_eps=0.2`, `entropy_coef=0.01`, `value_coef=0.5`, `max_grad_norm=0.5`, `rollout_steps=512`, `batch_size=64`, `ppo_epochs=4` |
