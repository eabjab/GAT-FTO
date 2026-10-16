# GAT-FTO Execution Plan & State Ledger

This file tracks the dependency-ordered execution plan and work package statuses for the **Graph Attention Networks for Frontier Tour Optimization (GAT-FTO)** project as specified in `SPECS.md`.

---

## Dependency Execution Waves

- **Wave 1 (Sequential Foundation):** `WP-1` (Scaffolding & `config.py`)
- **Wave 2 (Parallel Core Modules):** `WP-2` (Map Generator & Pathfinding) || `WP-3` (Raycasting Sensor & Frontier Extractor) || `WP-4` (GATv2 Encoder & Pointer Actor-Critic Models)
- **Wave 3 (Parallel Mid-Tier Modules):** `WP-5` (Topological Graph Builder) || `WP-6` (Classical Exploration Baselines) || `WP-7` (Heterogeneous Rollout Buffer & PPO Trainer)
- **Wave 4 (Sequential Environment & RL Integration):** `WP-8` (Semi-MDP Exploration Environment & RL/Env Integration Suite)
- **Wave 5 (Parallel Benchmark & Visualization Suite):** `WP-9` (Quantitative Evaluation Suite) || `WP-10` (Dual-View & Race Presentation Visualizers)

---

## Work Packages

### WP-1: Project Scaffolding & Central Configuration
- **Wave:** 1 (Sequential)
- **Dependencies:** None
- **Contract Reference:** `SPECS.md` Section 2 & Section 2.1
- **Assigned Files/Directories:**
  - `pyproject.toml`
  - `config.py`
  - `envs/__init__.py`
  - `models/__init__.py`
  - `rl/__init__.py`
  - `baselines/__init__.py`
  - `visualization/__init__.py`
  - `tests/__init__.py`
- **Verification Command:** `.\.venv\Scripts\python.exe -c "from config import EnvConfig, ModelConfig, PPOConfig; e, m, p = EnvConfig(), ModelConfig(), PPOConfig(); assert e.grid_height == 100 and m.hidden_dim == 128 and p.lr == 3e-4; print('WP-1 OK')"`
- **Status:** `DONE`
- **Modified Files:** `pyproject.toml`, `config.py`, `envs/__init__.py`, `models/__init__.py`, `rl/__init__.py`, `baselines/__init__.py`, `visualization/__init__.py`, `tests/__init__.py`, `STATE.md`
- **Verification Command & Output Summary:** `.\.venv\Scripts\python.exe -c "from config import EnvConfig, ModelConfig, PPOConfig; e, m, p = EnvConfig(), ModelConfig(), PPOConfig(); assert e.grid_height == 100 and m.hidden_dim == 128 and p.lr == 3e-4; print('WP-1 OK')"` -> Exit Code 0 (`WP-1 OK`)

---

### WP-2: Procedural Map Generator & Geodesic Pathfinding
- **Wave:** 2 (Parallel)
- **Dependencies:** `WP-1`
- **Contract Reference:** `SPECS.md` Sections 3.1.1, 3.1.3, 3.2.3, 4.1
- **Interface Requirements:**
  - `envs/map_generator.py`: `generate_bsp_map(height: int = 100, width: int = 100, seed: int | None = None) -> np.ndarray` returning `int8` array (`0`=Free, `1`=Obstacle) with single 4-connected free component.
  - `envs/pathfinding.py`: 8-connected `astar_path(grid, start, goal, free_val=0)` returning `(path, cost)` and 8-connected Dijkstra (`dijkstra_distances` / `compute_geodesic_info`) returning shortest-path distances, departure angles, and arrival angles over `grid == free_val`.
- **Assigned Files/Directories:**
  - `envs/map_generator.py`
  - `envs/pathfinding.py`
  - `tests/test_map_generator.py`
- **Verification Command:** `.\.venv\Scripts\pytest.exe tests/test_map_generator.py -v`
- **Status:** `DONE`
- **Modified Files:** `envs/map_generator.py`, `envs/pathfinding.py`, `tests/test_map_generator.py`, `STATE.md`
- **Verification Command & Output Summary:** `.\.venv\Scripts\pytest.exe tests/test_map_generator.py -v` -> Exit Code 0 (`3 passed in 1.49s`)

---

### WP-3: 2D Raycasting Sensor & Frontier Extraction
- **Wave:** 2 (Parallel)
- **Dependencies:** `WP-1`
- **Contract Reference:** `SPECS.md` Sections 3.1.2, 3.1.3, 4.2
- **Interface Requirements:**
  - `envs/sensor.py`: `RaycastingSensor(sensor_range: int = 20, fov_deg: float = 360.0)` with `scan(ground_truth: np.ndarray, belief_map: np.ndarray, pose: tuple[int, int, float]) -> tuple[np.ndarray, int]`.
  - `envs/frontier_extractor.py`: `FrontierNode` dataclass and `FrontierExtractor` / `extract_frontiers(belief_map, min_frontier_size=3, max_cluster_size=20, sensor_range=20, fov_deg=360.0) -> list[FrontierNode]`.
- **Assigned Files/Directories:**
  - `envs/sensor.py`
  - `envs/frontier_extractor.py`
  - `tests/test_sensor_and_frontiers.py`
- **Verification Command:** `.\.venv\Scripts\pytest.exe tests/test_sensor_and_frontiers.py -v`
- **Status:** `DONE`
- **Modified Files:** `envs/sensor.py`, `envs/frontier_extractor.py`, `tests/test_sensor_and_frontiers.py`, `STATE.md`
- **Verification Command & Output Summary:** `.\.venv\Scripts\pytest.exe tests/test_sensor_and_frontiers.py -v` -> Exit Code 0 (`3 passed in 0.51s`)

---

### WP-4: GATv2 Encoder & Pointer Actor-Critic Architecture
- **Wave:** 2 (Parallel)
- **Dependencies:** `WP-1`
- **Contract Reference:** `SPECS.md` Sections 3.3.1, 3.3.2, 4.4
- **Interface Requirements:**
  - `models/gat_encoder.py`: `GATEncoder(nn.Module)` with `node_stem`, `edge_stem`, 3 `GATv2Conv` layers (`in_channels=128, out_channels=32, heads=4, concat=True, edge_dim=128, add_self_loops=True`), residual + `ELU` + `LayerNorm`, supporting `return_attention_weights: bool = False`.
  - `models/pointer_actor_critic.py`: `PointerActorCritic(nn.Module)` matching Section 3.3.2 specification, supporting single `Data` and `Batch` inputs, `return_attn: bool = False`, and `select_action(graph_data: Data, deterministic: bool = True) -> int`.
- **Assigned Files/Directories:**
  - `models/gat_encoder.py`
  - `models/pointer_actor_critic.py`
  - `tests/test_models.py`
- **Verification Command:** `.\.venv\Scripts\pytest.exe tests/test_models.py -v`
- **Status:** `DONE`
- **Modified Files:** `models/gat_encoder.py`, `models/pointer_actor_critic.py`, `tests/test_models.py`, `STATE.md`
- **Verification Command & Output Summary:** `.\.venv\Scripts\pytest.exe tests/test_models.py -v` -> Exit Code 0 (`4 passed in 3.06s`)

---

### WP-5: Dynamic Topological Graph Construction
- **Wave:** 3 (Parallel)
- **Dependencies:** `WP-1`, `WP-2`, `WP-3`
- **Contract Reference:** `SPECS.md` Sections 3.2.1, 3.2.2, 3.2.3, 3.2.4, 4.3
- **Interface Requirements:**
  - `envs/graph_builder.py`: `GraphBuilder` with `build_graph(agent_pose: tuple[int, int, float], frontiers: list[FrontierNode], belief_map: np.ndarray) -> Data` populating `x`, `edge_index`, `edge_attr`, `agent_to_frontier_edge_attr`, `is_frontier`, `node_positions`.
- **Assigned Files/Directories:**
  - `envs/graph_builder.py`
  - `tests/test_graph_builder.py`
- **Verification Command:** `.\.venv\Scripts\pytest.exe tests/test_graph_builder.py -v`
- **Status:** `DONE`
- **Modified Files:** `envs/graph_builder.py`, `tests/test_graph_builder.py`, `STATE.md`
- **Verification Command & Output Summary:** `.\.venv\Scripts\pytest.exe tests/test_graph_builder.py -v` -> Exit Code 0 (`2 passed in 3.33s`)

---

### WP-6: Classical Exploration Baselines
- **Wave:** 3 (Parallel)
- **Dependencies:** `WP-1`, `WP-4`
- **Contract Reference:** `SPECS.md` Section 3.5
- **Interface Requirements:**
  - `baselines/classical_agents.py`: `BaseExplorationAgent(ABC)`, `NearestFrontierAgent`, `UtilityGreedyAgent` ($\lambda = 4.0$), `MyopicTSPAgent` (`networkx.approximation.greedy_tsp`), and `LearnedPolicyAgent` wrapper for `PointerActorCritic`.
- **Assigned Files/Directories:**
  - `baselines/classical_agents.py`
- **Verification Command:** `.\.venv\Scripts\python.exe -c "import torch; from torch_geometric.data import Data; from baselines.classical_agents import BaseExplorationAgent, NearestFrontierAgent, UtilityGreedyAgent, MyopicTSPAgent; x = torch.zeros(4, 8); x[1:, 7] = torch.tensor([0.2, 0.9, 0.1]); ei = torch.tensor([[0,0,0,1,2,3,1,2,1,3,2,3],[1,2,3,0,0,0,2,1,3,1,3,2]], dtype=torch.long); ea = torch.full((12, 4), 0.2); a2f = torch.tensor([[0.5,0,0,0],[0.1,0,0,0],[0.8,0,0,0]], dtype=torch.float32); d = Data(x=x, edge_index=ei, edge_attr=ea, agent_to_frontier_edge_attr=a2f, is_frontier=torch.tensor([False,True,True,True])); assert NearestFrontierAgent().select_action(d) == 1; assert UtilityGreedyAgent().select_action(d) == 1; assert 0 <= MyopicTSPAgent().select_action(d) < 3; print('WP-6 OK')"`
- **Status:** `DONE`
- **Modified Files:** `baselines/classical_agents.py`, `STATE.md`
- **Verification Command & Output Summary:** `.\.venv\Scripts\python.exe -c "import torch; from torch_geometric.data import Data; from baselines.classical_agents import BaseExplorationAgent, NearestFrontierAgent, UtilityGreedyAgent, MyopicTSPAgent; x = torch.zeros(4, 8); x[1:, 7] = torch.tensor([0.2, 0.9, 0.1]); ei = torch.tensor([[0,0,0,1,2,3,1,2,1,3,2,3],[1,2,3,0,0,0,2,1,3,1,3,2]], dtype=torch.long); ea = torch.full((12, 4), 0.2); a2f = torch.tensor([[0.5,0,0,0],[0.1,0,0,0],[0.8,0,0,0]], dtype=torch.float32); d = Data(x=x, edge_index=ei, edge_attr=ea, agent_to_frontier_edge_attr=a2f, is_frontier=torch.tensor([False,True,True,True])); assert NearestFrontierAgent().select_action(d) == 1; assert UtilityGreedyAgent().select_action(d) == 1; assert 0 <= MyopicTSPAgent().select_action(d) < 3; print('WP-6 OK')"` -> Exit Code 0 (`WP-6 OK`)

---

### WP-7: Heterogeneous Graph Rollout Buffer & PPO Trainer
- **Wave:** 3 (Parallel)
- **Dependencies:** `WP-1`, `WP-4`
- **Contract Reference:** `SPECS.md` Sections 3.4.1, 3.4.2
- **Interface Requirements:**
  - `rl/buffer.py`: `RolloutBuffer` storing `graphs: list[Data]`, `actions`, `log_probs`, `rewards`, `dones`, `values`, `returns`, `advantages`, and `compute_returns_and_advantages(last_value, gamma, gae_lambda)`.
  - `rl/ppo_trainer.py`: `PPOTrainer` implementing vectorized variable-length batched action log-probs (`ptr` via `bincount`), per-graph entropy via `scatter_add`, and clipped surrogate PPO update loop.
- **Assigned Files/Directories:**
  - `rl/buffer.py`
  - `rl/ppo_trainer.py`
- **Verification Command:** `.\.venv\Scripts\python.exe -c "import torch; from torch_geometric.data import Data; from config import ModelConfig, PPOConfig; from models.gat_encoder import GATEncoder; from models.pointer_actor_critic import PointerActorCritic; from rl.buffer import RolloutBuffer; from rl.ppo_trainer import PPOTrainer; m = PointerActorCritic(GATEncoder(ModelConfig())); t = PPOTrainer(m, PPOConfig(batch_size=2, ppo_epochs=2)); b = RolloutBuffer(); g = Data(x=torch.randn(4, 8), edge_index=torch.tensor([[0,0,0,1,2,3],[1,2,3,0,0,0]]), edge_attr=torch.rand(6, 4), agent_to_frontier_edge_attr=torch.rand(3, 4), is_frontier=torch.tensor([False,True,True,True])); b.add(g, 1, -1.0, 1.0, False, 0.5); b.add(g, 1, -1.0, 1.0, True, 0.5); b.compute_returns_and_advantages(0.0, 0.99, 0.95); metrics = t.update(b); assert 'policy_loss' in metrics; print('WP-7 OK')"`
- **Status:** `DONE`
- **Modified Files:** `rl/buffer.py`, `rl/ppo_trainer.py`, `STATE.md`
- **Verification Command & Output Summary:** `.\.venv\Scripts\python.exe -c "import torch; from torch_geometric.data import Data; from config import ModelConfig, PPOConfig; from models.gat_encoder import GATEncoder; from models.pointer_actor_critic import PointerActorCritic; from rl.buffer import RolloutBuffer; from rl.ppo_trainer import PPOTrainer; m = PointerActorCritic(GATEncoder(ModelConfig())); t = PPOTrainer(m, PPOConfig(batch_size=2, ppo_epochs=2)); b = RolloutBuffer(); g = Data(x=torch.randn(4, 8), edge_index=torch.tensor([[0,0,0,1,2,3],[1,2,3,0,0,0]]), edge_attr=torch.rand(6, 4), agent_to_frontier_edge_attr=torch.rand(3, 4), is_frontier=torch.tensor([False,True,True,True])); b.add(g, 1, -1.0, 1.0, False, 0.5); b.add(g, 1, -1.0, 1.0, True, 0.5); b.compute_returns_and_advantages(0.0, 0.99, 0.95); metrics = t.update(b); assert 'policy_loss' in metrics; print('WP-7 OK')"` -> Exit Code 0 (`WP-7 OK`)

---

### WP-8: Semi-MDP Exploration Environment & RL/Env Integration Suite
- **Wave:** 4 (Sequential)
- **Dependencies:** `WP-2`, `WP-3`, `WP-4`, `WP-5`, `WP-6`, `WP-7`
- **Contract Reference:** `SPECS.md` Sections 3.1.4, 4.5
- **Interface Requirements:**
  - `envs/exploration_env.py`: `FrontierExplorationEnv(gymnasium.Env)` implementing `reset(seed=None, options=None)` and `step(action_idx: int)` with A* macro-step horizon, early macro-step termination, final orientation alignment + scan, reward calculation, and primitive step trajectory history recording.
  - `tests/test_rl_and_env.py`: `test_env_episode_completion` and `test_single_graph_ppo_overfit`.
- **Assigned Files/Directories:**
  - `envs/exploration_env.py`
  - `tests/test_rl_and_env.py`
- **Verification Command:** `.\.venv\Scripts\pytest.exe tests/test_rl_and_env.py -v`
- **Status:** `DONE`
- **Modified Files:** `envs/exploration_env.py`, `tests/test_rl_and_env.py`, `STATE.md`
- **Verification Command & Output Summary:** `.\.venv\Scripts\pytest.exe tests/test_rl_and_env.py -v` -> Exit Code 0 (`2 passed in 5.56s`)

---

### WP-9: Quantitative Evaluation Benchmark Suite
- **Wave:** 5 (Parallel)
- **Dependencies:** `WP-8`
- **Contract Reference:** `SPECS.md` Section 3.5 (`evaluate.py`)
- **Interface Requirements:**
  - `evaluate.py`: Quantitative benchmark evaluating agents across fixed test seeds (`seed = 10000..10099` by default, configurable via CLI/args) computing Total Path Length to 95% and 98% Coverage, Exploration Efficiency, and Backtracking Ratio, outputting summary table and JSON report.
- **Assigned Files/Directories:**
  - `evaluate.py`
- **Verification Command:** `.\.venv\Scripts\python.exe evaluate.py --num-seeds 2 --max-steps 30`
- **Status:** `DONE`
- **Modified Files:** `evaluate.py`, `STATE.md`
- **Verification Command & Output Summary:** `.\.venv\Scripts\python.exe evaluate.py --num-seeds 2 --max-steps 30` -> Exit Code 0 (Evaluated `NearestFrontierAgent`, `UtilityGreedyAgent`, `MyopicTSPAgent`, and `PointerActorCritic` across seeds `10000..10001`; printed summary table and saved JSON report to `build/evaluation_results.json`)

---

### WP-10: Presentation Visualization Suite (Dual-View & Race Visualizers)
- **Wave:** 5 (Parallel)
- **Dependencies:** `WP-8`
- **Contract Reference:** `SPECS.md` Sections 3.6.1, 3.6.2, 4.6
- **Interface Requirements:**
  - `visualization/dual_view_visualizer.py`: `DualViewVisualizer` with `render_frame(...) -> np.ndarray` (RGB `(H_px, W_px, 3)`) and `.mp4`/`.gif` saving.
  - `visualization/race_visualizer.py`: `RaceVisualizer` with distance-synchronized interpolation (`delta_d = 2.0`), `render_frame(...) -> np.ndarray` (RGB `(H_px, W_px, 3)`), and `.mp4`/`.gif` saving.
  - `tests/test_visualizers.py`: `test_dual_view_and_race_headless_render`.
- **Assigned Files/Directories:**
  - `visualization/dual_view_visualizer.py`
  - `visualization/race_visualizer.py`
  - `tests/test_visualizers.py`
- **Verification Command:** `.\.venv\Scripts\pytest.exe tests/test_visualizers.py -v`
- **Status:** `DONE`
- **Modified Files:** `visualization/dual_view_visualizer.py`, `visualization/race_visualizer.py`, `tests/test_visualizers.py`, `STATE.md`
- **Verification Command & Output Summary:** `.\.venv\Scripts\pytest.exe tests/test_visualizers.py -v` -> Exit Code 0 (`1 passed in 7.22s`)
