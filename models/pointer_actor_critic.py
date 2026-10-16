"""Pointer-Network Cross-Attention Actor and Global-Pooled Critic architecture."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

from config import ModelConfig
from models.gat_encoder import GATEncoder


class PointerActorCritic(nn.Module):
    """Cross-attention Pointer Actor and Pooled Critic over variable-size frontier graphs.

    Implements Section 3.3.2 of SPECS.md and provides ``select_action`` for episode rollouts.
    """

    def __init__(
        self,
        encoder: nn.Module | ModelConfig | None = None,
        hidden_dim: int = 128,
        logit_clip: float = 10.0,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        if isinstance(encoder, ModelConfig):
            cfg = encoder
            self.encoder: nn.Module = GATEncoder(cfg)
            self.hidden_dim = cfg.hidden_dim
            self.logit_clip = cfg.logit_clip
        elif encoder is None:
            cfg = config if config is not None else ModelConfig(
                hidden_dim=hidden_dim, logit_clip=logit_clip
            )
            self.encoder = GATEncoder(cfg)
            self.hidden_dim = cfg.hidden_dim
            self.logit_clip = cfg.logit_clip
        else:
            self.encoder = encoder
            if config is not None:
                self.hidden_dim = config.hidden_dim
                self.logit_clip = config.logit_clip
            else:
                self.hidden_dim = hidden_dim
                self.logit_clip = logit_clip

        # Actor Pointer Projections
        self.query_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Critic Head
        self.critic_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self, data: Data | Batch, return_attn: bool = False
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        """Evaluate action distribution and state values for single or batched graphs."""
        # Ensure batch vector exists even for a single unbatched Data object
        batch_vec = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        )

        if return_attn:
            node_embeds, edge_embeds, attn_tuple = self.encoder(
                data, return_attention_weights=True
            )
        else:
            node_embeds, edge_embeds = self.encoder(
                data, return_attention_weights=False
            )
            attn_tuple = None

        # Identify Agent Nodes (one per graph in batch) and Frontier Nodes
        if hasattr(data, "is_frontier") and data.is_frontier is not None:
            is_frontier = data.is_frontier.bool()
        else:
            is_frontier = data.x[:, 0] < 0.5
        is_agent = ~is_frontier

        agent_embeds = node_embeds[is_agent]                          # Shape: (B, d_h)
        mean_graph = global_mean_pool(node_embeds, batch_vec)         # Shape: (B, d_h)
        max_graph = global_max_pool(node_embeds, batch_vec)           # Shape: (B, d_h)

        # Critic Value Prediction
        critic_input = torch.cat([agent_embeds, mean_graph, max_graph], dim=-1)
        values = self.critic_mlp(critic_input).squeeze(-1)            # Shape: (B,)

        # Actor Pointer Cross-Attention over Frontier Nodes only
        queries_per_graph = self.query_proj(
            torch.cat([agent_embeds, mean_graph], dim=-1)
        )                                                             # Shape: (B, d_h)
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

    @torch.no_grad()
    def select_action(self, graph_data: Data, deterministic: bool = True) -> int:
        """Select a local frontier index in {0, ..., K_t - 1} for a single graph."""
        param_device = next(self.parameters()).device
        if graph_data.x.device != param_device:
            graph_data = graph_data.to(param_device)

        probs, _, _, _, _ = self.forward(graph_data, return_attn=False)
        if probs.numel() == 0:
            return 0
        if deterministic:
            return int(torch.argmax(probs).item())
        dist = Categorical(probs=probs)
        return int(dist.sample().item())
