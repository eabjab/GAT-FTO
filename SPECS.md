# Technical Specification: Graph Attention Networks for Frontier Tour Optimization

## 1. Executive Summary & System Overview

### 1.1 Problem Statement & Objective

This project implements a graph-based reinforcement learning system for autonomous robot exploration in unknown 2D environments by framing frontier tour optimization as a dynamic variant of the Traveling Salesman Problem (TSP) or Orienteering Problem over a topological graph. Classical greedy exploration heuristics (such as nearest-frontier selection) make myopic decisions that cause frequent backtracking and inefficient trajectories. This system replaces hand-crafted utility heuristics with a global learned policy using a Graph Attention Network (GAT) encoder and a Pointer-Network cross-attention actor-critic architecture trained via Proximal Policy Optimization (PPO).

### 1.2 Core Architectural Paradigm

* **Semi-Markov Decision Process (Semi-MDP):** Each RL decision step $t$ constructs a dynamic topological graph $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t)$ representing the agent pose and all currently active frontier clusters. The policy outputs a discrete action $a_t \in \{1, \dots, K_t\}$ corresponding to a selected frontier node. A deterministic low-level A* planner navigates toward the selected frontier while continuously updating the occupancy belief map via simulated 2D raycasting.


* **Graph Representation:** Node features capture relative geometry, sensor orientation, cluster size, obstacle clearance, and estimated unmapped area. Directed edges connect nodes with geodesic A*/Dijkstra shortest-path distances and kinematic turning angles through known free space.


* **Variable Action-Space Policy:** Because the number of frontiers $K_t$ varies across steps, the model pairs a multi-head `GATv2Conv` encoder with a Pointer-style cross-attention Actor head and a global-pooled Critic head.



### 1.3 Software Stack & Dependencies

* **Language:** Python $\ge 3.10$
* **Core Libraries:**
* `torch>=2.1.0`
* `torch-geometric>=2.4.0`
* `gymnasium>=0.29.0`
* `numpy>=1.24.0`
* `scipy>=1.11.0` (connected components, Euclidean distance transforms, $k$-means)
* `networkx>=3.1` (classical TSP baseline approximation)
* `matplotlib>=3.7.0` (headless `Agg` rendering and `FFMpegWriter` animations)
* `pytest>=7.4.0` (automated unit and integration verification)



---

## 2. Project Scaffolding & Code Organization

Coding agents must adhere strictly to the following directory layout and module boundaries:

```text
gat_frontier_exploration/
├── pyproject.toml
├── SPEC.md
├── config.py                      # Central dataclass configuration
├── envs/
│   ├── __init__.py
│   ├── map_generator.py           # BSP procedural indoor grids + connectivity verification
│   ├── sensor.py                  # 2D Bresenham raycasting & FOV occlusion
│   ├── pathfinding.py             # A* path planner & Multi-Source Dijkstra on grids
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
│   └── classical_agents.py        # NearestFrontier, UtilityGreedy, MyopicTSP
├── visualization/
│   ├── __init__.py
│   ├── dual_view_visualizer.py    # Mode 1: Map exploration + GAT attention graph
│   └── race_visualizer.py         # Mode 2: Synchronized Nearest vs. GAT side-by-side
├── evaluate.py                    # Quantitative benchmark runner across test seeds
└── tests/
    ├── test_map_generator.py
    ├── test_sensor_and_frontiers.py
    ├── test_graph_builder.py
    ├── test_models.py
    ├── test_rl_and_env.py
    └── test_visualizers.py

```

### 2.1 Central Configuration Schema (`config.py`)

All hyperparameters must be centralized in immutable or serializable dataclasses:

```python
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

```

---

## 3. System Architecture & Component Specifications

### 3.1 Environment Generation & Sensor Simulation (`envs/`)

#### 3.1.1 Procedural Map Generator (`envs/map_generator.py`)

Generates 2D discrete occupancy grids dynamically per episode:

* **Grid Encoding:**
* `ground_truth`: `np.ndarray` of shape `(H, W)` with `dtype=np.int8` (`0` = Free, `1` = Obstacle).
* `belief_map`: `np.ndarray` of shape `(H, W)` with `dtype=np.int8` (`-1` = Unknown, `0` = Free, `1` = Obstacle).


* **Generation Algorithm (`generate_bsp_map(height, width, seed)`):**
1. Enclose the outer perimeter (`row 0, H-1` and `col 0, W-1`) with obstacles (`1`).
2. Recursively partition the interior space via Binary Space Partitioning (BSP) to depth $d \in \{3, 4\}$ with minimum leaf dimensions of $18 \times 18$ cells.
3. Build walls of thickness $1$ along partition lines and carve doorway openings of width $4$ to $6$ cells between sibling partitions.
4. Scatter $3$ to $8$ random rectangular obstacles ($2 \times 2$ to $5 \times 5$) inside rooms.
5. **Connectivity Invariant:** Run `scipy.ndimage.label` on `ground_truth == 0`. If multiple disconnected free components exist, retain only the largest component and fill smaller unreachable pockets with `1` (Obstacle) so $100\%$ of free cells are mutually reachable.



#### 3.1.2 2D Raycasting Sensor (`envs/sensor.py`)

* **Class:** `RaycastingSensor(sensor_range: int, fov_deg: float)`
* **Method:** `scan(ground_truth: np.ndarray, belief_map: np.ndarray, pose: tuple[int, int, float]) -> tuple[np.ndarray, int]`
* **Logic:**
1. Cast rays from agent cell $(r_0, c_0)$ to all perimeter cells of a bounding circle of radius `sensor_range` whose angle lies within $[\theta_0 - \frac{\text{FOV}}{2}, \theta_0 + \frac{\text{FOV}}{2}]$.
2. Step along each ray using integer Bresenham line coordinates.
3. Mark visited `Free` cells (`0`) in `belief_map`. If a ray hits an `Obstacle` cell (`1`) in `ground_truth`, mark that cell as `1` in `belief_map` and terminate the ray immediately.
4. Return the updated `belief_map` and `newly_uncovered_free_cells: int` (count of cells that transitioned from `-1` to `0` where `ground_truth == 0`).



#### 3.1.3 Frontier Extraction & Geodesic Pathfinding (`envs/frontier_extractor.py`, `envs/pathfinding.py`)

At every decision step, extract candidate frontiers from `belief_map`:

1. **Frontier Cell Mask:** A cell $(r, c)$ is a frontier cell iff `belief_map[r, c] == 0` and at least one of its 4-connected neighbors $(r \pm 1, c), (r, c \pm 1)$ has `belief_map == -1`.
2. **Connected Components & Splitting:**
* Group frontier edge cells into 8-connected components via `scipy.ndimage.label` with a $3 \times 3$ ones structuring element.
* Discard components with fewer than `min_frontier_size` cells.
* For any component with size $\vert{}C\vert{} > \text{max\_cluster\_size}$, split it into $k = \lceil \vert{}C\vert{} / \text{max\_cluster\_size} \rceil$ sub-clusters using `scipy.cluster.vq.kmeans2`.


3. **Medoid & Inward Normal Calculation:**
* **Medoid Position $\mathbf{p}_i = (r_i, c_i)$:** To guarantee the target lies in valid known free space, select the cell in cluster $C_i$ minimizing Euclidean distance to the cluster's arithmetic centroid:

$$\mathbf{p}_i = \arg\min_{(r, c) \in C_i} \left\Vert{} (r, c) - \frac{1}{\vert{}C_i\vert{}}\sum_{(r', c') \in C_i} (r', c') \right\Vert{}_2$$


* **Canonical Viewing Orientation $\psi_i \in [-\pi, \pi]$:** Extract a $7 \times 7$ local window around $\mathbf{p}_i$ in `belief_map`. Compute the vector from the centroid of `Free` (`0`) pixels in the window to the centroid of `Unknown` (`-1`) pixels in the window, and define $\psi_i = \text{atan2}(\Delta r, \Delta c)$.




4. **Node Feature Extraction Helpers:**
* **Obstacle Clearance:** Compute `edt = scipy.ndimage.distance_transform_edt(belief_map != 1)` and read `edt[r_i, c_i]`.


* **Estimated Unmapped Area:** Count `Unknown` (`-1`) cells within radius `sensor_range` of $\mathbf{p}_i$ (and within sensor FOV aligned with $\psi_i$) that have an unobstructed line-of-sight through non-obstacle cells (`belief_map != 1`), divided by $\pi \cdot \text{sensor\_range}^2$.





```python
from dataclasses import dataclass
import numpy as np

@dataclass
class FrontierNode:
    position: tuple[int, int]          # (row, col) medoid in belief_map == 0
    orientation: float                 # Inward normal angle in radians [-pi, pi]
    cluster_size: int                  # Number of frontier cells in cluster
    obstacle_distance: float           # EDT clearance to nearest known wall
    unmapped_area_est: float           # Normalized [0, 1] visible unknown area
    cells: np.ndarray                  # Shape (M, 2) array of cluster cells

```

#### 3.1.4 Semi-MDP Environment (`envs/exploration_env.py`)

`FrontierExplorationEnv(gymnasium.Env)` manages episodes, macro-steps, and reward calculation:

* **Macro-Step Execution (`step(action_idx: int)`):**
* `action_idx` is an integer in $\{0, \dots, K_t - 1\}$ indexing the selected frontier node $v_{a_t + 1}$ (since node $0$ in $\mathcal{G}_t$ is the Agent Pose Node).


* Compute the shortest 8-connected path $\mathcal{P} = [\mathbf{p}^{(0)}, \mathbf{p}^{(1)}, \dots, \mathbf{p}^{(L)}]$ from the agent's current position $\mathbf{p}^{(0)}$ to the target frontier medoid via A* over known free cells (`belief_map == 0`).
* Record the initial heading rotation $\Delta \theta_{\text{init}} = \vert{}\text{wrap}(\text{atan2}(r^{(1)} - r^{(0)}, c^{(1)} - c^{(0)}) - \theta_0)\vert{}$.


* Step the agent sequentially along $\mathcal{P}$ for at most `macro_step_horizon` steps. At each primitive step:
* Accumulate step distance ($1.0$ for cardinal moves, $\sqrt{2}$ for diagonal moves).
* Perform a `RaycastingSensor.scan()` at the new pose.
* **Early Termination of Macro-Step:** Stop stepping along $\mathcal{P}$ if (a) the agent reaches the target medoid, (b) all cells in the target frontier's `cells` array have been uncovered (`belief_map != -1`), or (c) the overall map coverage reaches `coverage_target`.


* If the agent reaches the target medoid, set the agent's final orientation $\theta_0$ to the frontier's viewing orientation $\psi_{a_t}$ and perform one final sensor scan.


* **Reward Calculation:**

$$r_t = w_{\text{cov}} \frac{\Delta N_{\text{uncovered}}}{N_{\text{total\_free}}} - w_{\text{dist}} \frac{\Delta d_{\text{traveled}}}{D_{\text{diag}}} - w_{\text{turn}} \frac{\Delta \theta_{\text{init}}}{\pi} + r_{\text{terminal}}$$



where $D_{\text{diag}} = \sqrt{H^2 + W^2}$.



---

### 3.2 Dynamic Topological Graph Construction (`envs/graph_builder.py`)

At the end of every macro-step, `GraphBuilder.build_graph(agent_pose, frontiers, belief_map)` constructs a `torch_geometric.data.Data` object representing $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t)$.

#### 3.2.1 Node Ordering Invariant

* Total nodes $N_t = K_t + 1$, where $K_t \ge 1$ is the number of active frontiers.


* **Node $0$ (`index = 0`) is strictly reserved for the Agent Pose Node**.


* **Nodes $1 \dots K_t$ (`indices = 1..K_t`) correspond to Frontier Nodes $0 \dots K_t - 1$**.



#### 3.2.2 Node Feature Matrix ($\mathbf{X} \in \mathbb{R}^{(K_t + 1) \times 8}$)

| Index | Feature Name | Agent Node ($i = 0$) | Frontier Node ($i \in \{1, \dots, K_t\}$) | Value Range |
| --- | --- | --- | --- | --- |
| `0` | `is_agent` | $1.0$ | $0.0$ | $\{0, 1\}$ |
| `1` | `rel_row` | $0.0$ | $(r_i - r_0) / H$ | $[-1, 1]$ |
| `2` | `rel_col` | $0.0$ | $(c_i - c_0) / W$ | $[-1, 1]$ |
| `3` | `cos_orient` | $\cos(\theta_0)$ | $\cos(\psi_i)$ | $[-1, 1]$ |
| `4` | `sin_orient` | $\sin(\theta_0)$ | $\sin(\psi_i)$ | $[-1, 1]$ |
| `5` | `cluster_size` | $0.0$ | $\min(1.0, \vert{}C_i\vert{} / \text{max\_cluster\_size})$ | $[0, 1]$ |
| `6` | `obs_clearance` | $\min(1.0, d_{\text{obs}}(\mathbf{p}_0) / R_{\max})$ | $\min(1.0, d_{\text{obs}}(\mathbf{p}_i) / R_{\max})$ | $[0, 1]$ |
| `7` | `unmapped_est` | $0.0$ | $\text{unmapped\_area\_est}_i$ | $[0, 1]$ |

#### 3.2.3 Edge Connectivity & Edge Attribute Tensor ($\mathbf{E} \in \mathbb{R}^{\vert{}\mathcal{E}_t\vert{} \times 4}$)

* **Topology:** Complete directed graph without self-loops over $\mathcal{V}_t$ (if $K_t + 1 > 16$, restrict frontier-to-frontier edges to the $k=12$ nearest geodesic neighbors while retaining all edges from and to Node $0$).
* **Geodesic Path Computation:** Run 8-connected Dijkstra from each node position $\mathbf{p}_i$ over `belief_map == 0` to obtain exact shortest-path distance $d_{\text{geo}}(i, j)$, initial departure path angle $\phi_{\text{start}}^{(i \to j)}$, and final arrival path angle $\phi_{\text{end}}^{(i \to j)}$.



| Index | Feature Name | Formula | Value Range |
| --- | --- | --- | --- |
| `0` | `geo_dist` | $\min(2.0, d_{\text{geo}}(i, j) / D_{\text{diag}})$ | $[0, 2]$ |
| `1` | `euc_dist` | $\Vert{}\mathbf{p}_j - \mathbf{p}_i\Vert{}_2 / D_{\text{diag}}$ | $[0, 1]$ |
| `2` | `dep_angle` | $\vert{}\text{wrap}(\phi_{\text{start}}^{(i \to j)} - \psi_i)\vert{} / \pi$ | $[0, 1]$ |
| `3` | `arr_angle` | $\vert{}\text{wrap}(\psi_j - \phi_{\text{end}}^{(i \to j)})\vert{} / \pi$ | $[0, 1]$ |

#### 3.2.4 PyG `Data` Contract

Each returned `Data` instance must populate the following fields:

* `data.x`: `torch.FloatTensor` of shape `(K_t + 1, 8)`
* `data.edge_index`: `torch.LongTensor` of shape `(2, E_t)`
* `data.edge_attr`: `torch.FloatTensor` of shape `(E_t, 4)`
* `data.agent_to_frontier_edge_attr`: `torch.FloatTensor` of shape `(K_t, 4)` containing the direct edge features $\mathbf{e}_{0 \to i}$ for $i \in \{1, \dots, K_t\}$
* `data.is_frontier`: `torch.BoolTensor` of shape `(K_t + 1,)` where `data.is_frontier[0] = False` and `data.is_frontier[1:] = True`
* `data.node_positions`: `torch.FloatTensor` of shape `(K_t + 1, 2)` storing grid $(r, c)$ coordinates for visualization.

---

### 3.3 Neural Network Architecture (`models/`)

#### 3.3.1 GATv2 Encoder (`models/gat_encoder.py`)

Processes node and edge features using multi-head attention with edge attributes:

* **Projection Stems:**
* `node_stem = nn.Linear(8, 128)` followed by `nn.ReLU()`
* `edge_stem = nn.Linear(4, 128)` followed by `nn.ReLU()`


* **Message Passing Layers ($L = 3$):**
* Each layer uses `GATv2Conv(in_channels=128, out_channels=32, heads=4, concat=True, edge_dim=128, add_self_loops=True)`.
* Residual connection and layer normalization:

$$\mathbf{h}_i^{(\ell)} = \text{LayerNorm}\left(\mathbf{h}_i^{(\ell - 1)} + \text{ELU}\left(\text{GATv2Conv}^{(\ell)}(\mathbf{h}^{(\ell - 1)}, \text{edge\_index}, \mathbf{u})\right)\right)$$


* Must support `return_attention_weights: bool = False` in `forward()` to return `(edge_index_attn, alpha)` from the final `GATv2Conv` layer for the Dual-View Visualizer.



#### 3.3.2 Pointer Actor-Critic (`models/pointer_actor_critic.py`)

Handles single `Data` graphs and batched `Batch` super-graphs seamlessly:

```python
import math
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax

class PointerActorCritic(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int = 128, logit_clip: float = 10.0):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.logit_clip = logit_clip

        # Actor Pointer Projections
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Critic Head
        self.critic_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data: Data | Batch, return_attn: bool = False):
        # Ensure batch vector exists even for a single unbatched Data object
        batch_vec = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(
            data.x.size(0), dtype=torch.long, device=data.x.device
        )

        if return_attn:
            node_embeds, edge_embeds, attn_tuple = self.encoder(data, return_attention_weights=True)
        else:
            node_embeds, edge_embeds = self.encoder(data, return_attention_weights=False)
            attn_tuple = None

        # Identify Agent Nodes (one per graph in batch) and Frontier Nodes
        is_frontier = data.is_frontier
        is_agent = ~is_frontier

        agent_embeds = node_embeds[is_agent]                          # Shape: (B, d_h)
        mean_graph = global_mean_pool(node_embeds, batch_vec)         # Shape: (B, d_h)
        max_graph = global_max_pool(node_embeds, batch_vec)           # Shape: (B, d_h)

        # Critic Value Prediction
        critic_input = torch.cat([agent_embeds, mean_graph, max_graph], dim=-1)
        values = self.critic_mlp(critic_input).squeeze(-1)            # Shape: (B,)

        # Actor Pointer Cross-Attention over Frontier Nodes only
        queries_per_graph = self.query_proj(torch.cat([agent_embeds, mean_graph], dim=-1)) # (B, d_h)
        frontier_batch_vec = batch_vec[is_frontier]                   # Shape: (Total_Frontiers,)
        expanded_queries = queries_per_graph[frontier_batch_vec]      # Shape: (Total_Frontiers, d_h)

        frontier_embeds = node_embeds[is_frontier]                    # Shape: (Total_Frontiers, d_h)
        direct_edge_embeds = self.encoder.edge_stem(data.agent_to_frontier_edge_attr)
        keys = self.key_proj(torch.cat([frontier_embeds, direct_edge_embeds], dim=-1))

        # Scaled dot-product with tanh clipping
        raw_scores = (expanded_queries * keys).sum(dim=-1) / math.sqrt(self.hidden_dim)
        clipped_logits = self.logit_clip * torch.tanh(raw_scores)     # Shape: (Total_Frontiers,)

        # Per-graph probability distribution over variable K_t frontiers
        probs = pyg_softmax(clipped_logits, frontier_batch_vec)

        return probs, clipped_logits, frontier_batch_vec, values, attn_tuple

```

---

### 3.4 Reinforcement Learning Pipeline (`rl/`)

#### 3.4.1 Heterogeneous Graph Rollout Buffer (`rl/buffer.py`)

Because each transition has a different number of nodes $K_t + 1$, transitions must be stored as a Python list of `Data` objects alongside 1D tensors for scalar RL quantities:

* `graphs: list[Data]` (stored on CPU, moved to device in mini-batches via `Batch.from_data_list`)
* `actions: torch.LongTensor` (local frontier index $a_t \in \{0, \dots, K_t - 1\}$ within each graph)
* `log_probs: torch.FloatTensor`
* `rewards: torch.FloatTensor`
* `dones: torch.BoolTensor`
* `values: torch.FloatTensor`
* **GAE Computation (`compute_returns_and_advantages(last_value, gamma, gae_lambda)`):**
Computes $\hat{A}_t$ and $\hat{R}_t = \hat{A}_t + V_\phi(s_t)$, and normalizes advantages per rollout batch: $\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu_{\hat{A}}}{\sigma_{\hat{A}} + 10^{-8}}$.

#### 3.4.2 PPO Trainer (`rl/ppo_trainer.py`)

* **Evaluating Variable-Length Batched Actions:**
When `Batch.from_data_list(batch_graphs)` collates $B$ graphs, `probs` has shape $(\sum_{b=1}^B K_b,)$. To extract $\log \pi_\theta(a_b \mid s_b)$ and per-graph entropy $H(\pi_\theta(\cdot \mid s_b))$ cleanly in vectorized PyTorch:
1. Compute frontier count offsets `ptr` from `frontier_batch_vec` (`torch.bincount(frontier_batch_vec)` cumulative sum).
2. Convert local action indices $a_b \in \{0, \dots, K_b - 1\}$ into global indices $g_b = \text{ptr}[b] + a_b$.
3. Index `new_log_probs = torch.log(probs[g_b] + 1e-8)`.
4. Compute per-graph entropy via `scatter_add`:

$$H_b = -\sum_{i \in \text{Frontiers}(b)} p_i \log(p_i + 10^{-8})$$




* **Total Loss Function:**

$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}_b\left[\min\left(r_b(\theta)\hat{A}_b, \text{clip}(r_b(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_b\right)\right] + c_v \mathbb{E}_b\left[(V_\phi(s_b) - \hat{R}_b)^2\right] - c_e \mathbb{E}_b[H_b]$$



---

### 3.5 Classical Baselines & Evaluation Suite (`baselines/`, `evaluate.py`)

All agents (learned and classical) must implement a common abstract interface:

```python
from abc import ABC, abstractmethod
from torch_geometric.data import Data

class BaseExplorationAgent(ABC):
    @abstractmethod
    def select_action(self, graph_data: Data, deterministic: bool = True) -> int:
        """Returns local frontier index in {0, ..., K_t - 1}."""
        pass

```

1. **`NearestFrontierAgent`:** Selects $a_t = \arg\min_{i \in \{0, \dots, K_t - 1\}} \text{data.agent\_to\_frontier\_edge\_attr}[i, 0]$ (minimum normalized geodesic distance).


2. **`UtilityGreedyAgent`:** Selects $a_t = \arg\max_{i} \frac{\text{unmapped\_est}_i}{\exp(\lambda \cdot \text{geo\_dist}_i)}$ with $\lambda = 4.0$.
3. **`MyopicTSPAgent`:** Builds a distance matrix over $\mathcal{V}_t$ from `data.edge_attr[:, 0]`, solves a tour starting at Node $0$ via `networkx.approximation.greedy_tsp`, and returns the first frontier visited after Node $0$.

#### Quantitative Benchmark (`evaluate.py`)

Evaluates all agents across $100$ fixed test seeds (`seed = 10000..10099`) and outputs a summary table and JSON report containing:

* **Total Path Length to $95\%$ and $98\%$ Coverage**.


* **Exploration Efficiency:** Mean cells uncovered per unit distance traveled ($\frac{\text{Total Uncovered Free Cells}}{\text{Total Distance Traveled}}$).


* **Backtracking Ratio:** Fraction of primitive path steps spent on grid cells already traversed earlier in the episode.

---

### 3.6 Presentation Visualization Suite (`visualization/`)

#### 3.6.1 Mode 1: Dual-View Graph & Environment Inspector (`visualization/dual_view_visualizer.py`)

Generates a $1 \times 2$ side-by-side figure (`figsize=(14, 6)`, $150$ DPI) at every macro-step and compiles frames into an `.mp4` or `.gif`:

* **Left Subplot (`ax_env` — Environment Exploration):**
* Renders `belief_map` (`-1` = dark gray `#2b2b2b`, `0` = white `#ffffff`, `1` = black `#000000`).
* Plots the agent's cumulative historical trajectory as a continuous blue line (`#1f77b4`) and current pose as a directional triangle.
* Plots active frontier clusters as scatter points colored by cluster index and draws the A* shortest path to the chosen target frontier in dashed crimson (`#d62728`).


* **Right Subplot (`ax_graph` — Topological Graph & GAT Attention):**
* Plots nodes at their exact $(c, r)$ spatial coordinates overlaid on a faint wireframe of the map boundaries.
* Extracts `(edge_index_attn, alpha)` from the final `GATv2Conv` layer (`alpha` averaged across the $4$ heads).
* Renders directed graph edges with line thickness $w_{ij} = 0.5 + 4.0 \cdot \alpha_{ij}$ and opacity $\alpha_{ij} \in [0.15, 1.0]$ so high-attention relationships stand out visually.
* Colors Frontier Nodes using the `plasma` colormap mapped to the Actor's probability distribution $\pi_\theta(a_t = i \mid \mathcal{G}_t)$, and highlights the selected frontier node with a bold gold outer ring.



#### 3.6.2 Mode 2: Synchronized Side-by-Side Race (`visualization/race_visualizer.py`)

Compares `NearestFrontierAgent` vs. `PointerActorCritic` on the exact same map seed:

* **Distance-Synchronized Interpolation:** Roll out both episodes first and record full primitive step histories $[(\mathbf{p}_0, \text{belief}_0, \text{cov}_0), \dots]$ indexed by cumulative distance traveled $d \in [0, D_{\max}]$. Render animation frames at uniform distance increments $\Delta d = 2.0$ grid units so both agents move at the exact same speed in the video.
* **Layout (`matplotlib.gridspec.GridSpec(2, 2, height_ratios=[3, 1])`):**
* **Top-Left:** Nearest Frontier occupancy belief map, historical trajectory, and active frontiers.


* **Top-Right:** Trained GAT Policy occupancy belief map, historical trajectory, and active frontiers.


* **Bottom (Spanning Both Columns):** Live animated line plot of `% Map Covered` (y-axis, $0\%$ to $100\%$) vs. `Distance Traveled` (x-axis), with vertical markers indicating when each agent completes exploration.





---

## 4. Acceptance Criteria & Automated Verification (`tests/`)

Coding agents must implement and pass the following `pytest` verification suite before considering any implementation phase complete. All tests must execute in under $30$ seconds on CPU.

### 4.1 Map Generation & Pathfinding Verification (`tests/test_map_generator.py`)

* **`test_map_determinism_and_bounds`:** Calling `generate_bsp_map(100, 100, seed=42)` twice returns bit-identical arrays; outer borders (`row 0, -1` and `col 0, -1`) are strictly `1` (Obstacle).
* **`test_single_connected_component`:** Across $10$ random seeds, `scipy.ndimage.label(ground_truth == 0)` with a 4-connected structuring element yields exactly `num_features == 1`.
* **`test_astar_and_dijkstra_consistency`:** On a generated map, the A* path length between two random free cells equals the Multi-Source Dijkstra distance within $10^{-5}$, and every cell along the returned A* path has `ground_truth == 0`.

### 4.2 Sensor & Frontier Extraction Verification (`tests/test_sensor_and_frontiers.py`)

* **`test_raycast_wall_occlusion`:** In a synthetic $20 \times 20$ room bisected by an obstacle wall at `col = 10`, scanning from `(10, 5)` never uncovers any cells with `col > 10`.
* **`test_frontier_medoid_in_free_space`:** Across $20$ random partial belief states, every extracted `FrontierNode` satisfies `belief_map[node.position] == 0` (Free) and has at least one 4-connected neighbor with `belief_map == -1` (Unknown).
* **`test_large_frontier_splitting`:** A synthetic straight horizontal frontier of length $45$ cells is automatically split into $\lceil 45 / 20 \rceil = 3$ distinct `FrontierNode` instances, each with `cluster_size <= 20`.

### 4.3 Graph Construction Verification (`tests/test_graph_builder.py`)

* **`test_graph_tensor_shapes_and_bounds`:** Given $K_t = 6$ frontiers, `GraphBuilder.build_graph` returns a `Data` object where:
* `data.x.shape == (7, 8)`
* `data.agent_to_frontier_edge_attr.shape == (6, 4)`
* `torch.isnan(data.x).any() == False` and `torch.isinf(data.edge_attr).any() == False`
* Node $0$ has `data.x[0, 0] == 1.0` (`is_agent`) and Nodes $1 \dots 6$ have `data.x[1:, 0] == 0.0`.
* All node features lie within $[-1.0, 1.0]$ and all edge features lie within $[0.0, 2.0]$.



### 4.4 Neural Network & PyG Batching Verification (`tests/test_models.py`)

* **`test_variable_action_space_forward`:** Pass two synthetic graphs with $K_1 = 3$ and $K_2 = 9$ frontiers through `PointerActorCritic`. Verify that `probs` has shape `(12,)`, ` probs[:3].sum()` equals $1.0 \pm 10^{-5}$, `probs[3:].sum()` equals $1.0 \pm 10^{-5}$, and `values` has shape `(2,)`.
* **`test_batching_invariance`:** Run `PointerActorCritic` on `g1` and `g2` individually, then on `Batch.from_data_list([g1, g2])`. Assert `torch.allclose(probs_individual, probs_batched, atol=1e-5)` and `torch.allclose(values_individual, values_batched, atol=1e-5)`.
* **`test_attention_weights_extraction`:** Calling `model(data, return_attn=True)` returns `attn_tuple = (edge_index_attn, alpha)` where `alpha.shape == (edge_index_attn.size(1), num_heads)` and all $\alpha \ge 0$.
* **`test_gradient_flow`:** Compute a dummy PPO loss on a batched graph, call `loss.backward()`, and assert that `param.grad is not None` and `torch.abs(param.grad).sum() > 0` for every parameter in `encoder`, `query_proj`, `key_proj`, and `critic_mlp`.

### 4.5 Environment Loop & PPO Sanity Verification (`tests/test_rl_and_env.py`)

* **`test_env_episode_completion`:** Run `NearestFrontierAgent` on a $50 \times 50$ map until `terminated or truncated`. Verify that coverage strictly increases on steps where frontiers are reached and no unhandled exceptions occur when $K_t$ drops to $0$.
* **`test_single_graph_ppo_overfit`:** Construct a fixed synthetic graph with $K_t = 4$ frontiers where action `2` is assigned reward $+1.0$ and all other actions receive $-1.0$. Run $30$ PPO update steps on this single state and assert that $\pi_\theta(a = 2 \mid \mathcal{G}) > 0.90$.

### 4.6 Visualizer Smoke Verification (`tests/test_visualizers.py`)

* **`test_dual_view_and_race_headless_render`:** Using `matplotlib.use("Agg")`, run a 3-step rollout on a $40 \times 40$ map, invoke both `DualViewVisualizer.render_frame()` and `RaceVisualizer.render_frame()`, and verify that both produce valid RGB numpy arrays of shape `(H_px, W_px, 3)` without raising runtime or plotting exceptions.