"""Unit tests for GATv2 Encoder and Pointer Actor-Critic models (SPECS.md Section 4.4)."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from config import ModelConfig
from models.gat_encoder import GATEncoder
from models.pointer_actor_critic import PointerActorCritic


def _make_synthetic_graph(num_frontiers: int, seed: int = 0) -> Data:
    """Create a synthetic PyG Data graph adhering to the Section 3.2.4 contract."""
    gen = torch.Generator().manual_seed(seed)
    num_nodes = num_frontiers + 1

    # Node features: shape (K_t + 1, 8)
    x = torch.rand((num_nodes, 8), generator=gen, dtype=torch.float32) * 2.0 - 1.0
    x[0, 0] = 1.0   # Agent node
    x[0, 1:3] = 0.0
    x[0, 5] = 0.0
    x[0, 7] = 0.0
    x[1:, 0] = 0.0  # Frontier nodes
    x[1:, 5:] = torch.rand((num_frontiers, 3), generator=gen, dtype=torch.float32)

    # Complete directed graph without self-loops
    src_list: list[int] = []
    dst_list: list[int] = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src_list.append(i)
                dst_list.append(j)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    num_edges = edge_index.size(1)
    edge_attr = torch.rand((num_edges, 4), generator=gen, dtype=torch.float32) + 0.1

    # Extract direct edges 0 -> i for i in 1..K_t
    agent_to_frontier_edge_attr = torch.rand(
        (num_frontiers, 4), generator=gen, dtype=torch.float32
    ) + 0.1

    is_frontier = torch.zeros(num_nodes, dtype=torch.bool)
    is_frontier[1:] = True

    node_positions = torch.rand((num_nodes, 2), generator=gen, dtype=torch.float32) * 100.0

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        agent_to_frontier_edge_attr=agent_to_frontier_edge_attr,
        is_frontier=is_frontier,
        node_positions=node_positions,
    )


def test_variable_action_space_forward() -> None:
    """Pass two synthetic graphs with K_1=3 and K_2=9 frontiers through PointerActorCritic."""
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = PointerActorCritic(GATEncoder(cfg), hidden_dim=cfg.hidden_dim, logit_clip=cfg.logit_clip)
    model.eval()

    g1 = _make_synthetic_graph(num_frontiers=3, seed=101)
    g2 = _make_synthetic_graph(num_frontiers=9, seed=202)
    batch = Batch.from_data_list([g1, g2])

    probs, clipped_logits, frontier_batch_vec, values, attn_tuple = model(batch, return_attn=False)

    assert probs.shape == (12,)
    assert clipped_logits.shape == (12,)
    assert frontier_batch_vec.shape == (12,)
    assert torch.allclose(probs[:3].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(probs[3:].sum(), torch.tensor(1.0), atol=1e-5)
    assert values.shape == (2,)
    assert attn_tuple is None

    # Also verify select_action returns valid local frontier index
    action_det = model.select_action(g1, deterministic=True)
    action_stoch = model.select_action(g1, deterministic=False)
    assert isinstance(action_det, int) and 0 <= action_det < 3
    assert isinstance(action_stoch, int) and 0 <= action_stoch < 3


def test_batching_invariance() -> None:
    """Verify individual forward passes match batched forward pass within 1e-5."""
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = PointerActorCritic(GATEncoder(cfg), hidden_dim=cfg.hidden_dim, logit_clip=cfg.logit_clip)
    model.eval()

    g1 = _make_synthetic_graph(num_frontiers=3, seed=11)
    g2 = _make_synthetic_graph(num_frontiers=9, seed=22)

    probs1, _, _, val1, _ = model(g1, return_attn=False)
    probs2, _, _, val2, _ = model(g2, return_attn=False)
    probs_individual = torch.cat([probs1, probs2], dim=0)
    values_individual = torch.cat([val1, val2], dim=0)

    batch = Batch.from_data_list([g1, g2])
    probs_batched, _, _, values_batched, _ = model(batch, return_attn=False)

    assert torch.allclose(probs_individual, probs_batched, atol=1e-5)
    assert torch.allclose(values_individual, values_batched, atol=1e-5)


def test_attention_weights_extraction() -> None:
    """Verify return_attn=True returns (edge_index_attn, alpha) with valid shapes and non-negative weights."""
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = PointerActorCritic(GATEncoder(cfg), hidden_dim=cfg.hidden_dim, logit_clip=cfg.logit_clip)
    model.eval()

    data = _make_synthetic_graph(num_frontiers=5, seed=33)
    probs, clipped_logits, frontier_batch_vec, values, attn_tuple = model(data, return_attn=True)

    assert attn_tuple is not None
    edge_index_attn, alpha = attn_tuple
    assert edge_index_attn.dim() == 2 and edge_index_attn.size(0) == 2
    assert alpha.shape == (edge_index_attn.size(1), cfg.num_heads)
    assert torch.all(alpha >= 0.0)


def test_gradient_flow() -> None:
    """Compute a dummy PPO loss on a batched graph and verify non-zero gradients on all parameters."""
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = PointerActorCritic(GATEncoder(cfg), hidden_dim=cfg.hidden_dim, logit_clip=cfg.logit_clip)
    model.train()
    model.zero_grad()

    g1 = _make_synthetic_graph(num_frontiers=4, seed=44)
    g2 = _make_synthetic_graph(num_frontiers=6, seed=55)
    batch = Batch.from_data_list([g1, g2])

    probs, clipped_logits, frontier_batch_vec, values, _ = model(batch, return_attn=False)

    # Compute frontier count offsets ptr and select actions a_0 = 1, a_1 = 3
    counts = torch.bincount(frontier_batch_vec)
    ptr = torch.zeros(counts.size(0) + 1, dtype=torch.long, device=probs.device)
    ptr[1:] = torch.cumsum(counts, dim=0)

    local_actions = torch.tensor([1, 3], dtype=torch.long, device=probs.device)
    global_actions = ptr[:-1] + local_actions
    new_log_probs = torch.log(probs[global_actions] + 1e-8)
    old_log_probs = new_log_probs.detach() - 0.1
    advantages = torch.tensor([1.2, -0.8], dtype=torch.float32, device=probs.device)

    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    returns = torch.tensor([2.5, -1.0], dtype=torch.float32, device=probs.device)
    value_loss = ((values - returns) ** 2).mean()

    entropy_terms = -probs * torch.log(probs + 1e-8)
    entropy_per_graph = torch.zeros(2, dtype=torch.float32, device=probs.device)
    entropy_per_graph.scatter_add_(0, frontier_batch_vec, entropy_terms)
    entropy = entropy_per_graph.mean()

    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    loss.backward()

    for module_name, module in [
        ("encoder", model.encoder),
        ("query_proj", model.query_proj),
        ("key_proj", model.key_proj),
        ("critic_mlp", model.critic_mlp),
    ]:
        for param_name, param in module.named_parameters():
            assert param.grad is not None, f"Grad is None for {module_name}.{param_name}"
            grad_abs_sum = torch.abs(param.grad).sum().item()
            assert grad_abs_sum > 0.0, f"Zero grad for {module_name}.{param_name}"
