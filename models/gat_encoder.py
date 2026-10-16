"""Multi-head GATv2Conv graph encoder with edge attributes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv

from config import ModelConfig


class GATEncoder(nn.Module):
    """GATv2 backbone processing topological node and edge attributes.

    Implements Section 3.3.1 of SPECS.md:
    - node_stem: Linear(8, 128) -> ReLU()
    - edge_stem: Linear(4, 128) -> ReLU()
    - L=3 message passing layers of GATv2Conv(128, 32, heads=4, concat=True,
      edge_dim=128, add_self_loops=True) with residual connection, ELU, and LayerNorm.
    """

    def __init__(
        self,
        config: ModelConfig | int | None = None,
        edge_feat_dim: int = 4,
        hidden_dim: int = 128,
        num_gat_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
        *,
        node_feat_dim: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(config, ModelConfig):
            self.node_feat_dim = config.node_feat_dim
            self.edge_feat_dim = config.edge_feat_dim
            self.hidden_dim = config.hidden_dim
            self.num_gat_layers = config.num_gat_layers
            self.num_heads = config.num_heads
            self.dropout = config.dropout
        elif isinstance(config, int):
            self.node_feat_dim = config
            self.edge_feat_dim = edge_feat_dim
            self.hidden_dim = hidden_dim
            self.num_gat_layers = num_gat_layers
            self.num_heads = num_heads
            self.dropout = dropout
        else:
            self.node_feat_dim = node_feat_dim if node_feat_dim is not None else 8
            self.edge_feat_dim = edge_feat_dim
            self.hidden_dim = hidden_dim
            self.num_gat_layers = num_gat_layers
            self.num_heads = num_heads
            self.dropout = dropout

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        head_out_channels = self.hidden_dim // self.num_heads

        # Projection Stems
        self.node_stem = nn.Sequential(
            nn.Linear(self.node_feat_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.edge_stem = nn.Sequential(
            nn.Linear(self.edge_feat_dim, self.hidden_dim),
            nn.ReLU(),
        )

        # Message Passing Layers (L = 3)
        self.conv_layers = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=self.hidden_dim,
                    out_channels=head_out_channels,
                    heads=self.num_heads,
                    concat=True,
                    edge_dim=self.hidden_dim,
                    add_self_loops=True,
                    dropout=self.dropout,
                )
                for _ in range(self.num_gat_layers)
            ]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_gat_layers)]
        )
        self.elu = nn.ELU()

    def forward(
        self,
        data_or_x: Data | Batch | torch.Tensor,
        edge_index: torch.Tensor | bool | None = None,
        edge_attr: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
    ):
        """Run forward pass through stems and GATv2 message passing layers.

        Args:
            data_or_x: Either a PyG ``Data``/``Batch`` object or node tensor ``x`` of shape ``(N, 8)``.
            edge_index: Edge index tensor of shape ``(2, E)`` when ``data_or_x`` is a Tensor,
                or optionally ``return_attention_weights`` bool if ``data_or_x`` is ``Data``/``Batch``.
            edge_attr: Edge feature tensor of shape ``(E, 4)`` when ``data_or_x`` is a Tensor.
            return_attention_weights: If True, also returns ``(edge_index_attn, alpha)``
                from the final ``GATv2Conv`` layer.

        Returns:
            ``(node_embeds, edge_embeds)`` when ``return_attention_weights=False``, or
            ``(node_embeds, edge_embeds, (edge_index_attn, alpha))`` when ``True``.
        """
        if isinstance(data_or_x, (Data, Batch)) or (
            hasattr(data_or_x, "x")
            and hasattr(data_or_x, "edge_index")
            and hasattr(data_or_x, "edge_attr")
        ):
            if isinstance(edge_index, bool):
                return_attention_weights = edge_index
            x = data_or_x.x
            ei = data_or_x.edge_index
            ea = data_or_x.edge_attr
        else:
            x = data_or_x
            if edge_index is None or isinstance(edge_index, bool) or edge_attr is None:
                raise ValueError(
                    "edge_index and edge_attr tensors must be provided when passing x directly."
                )
            ei = edge_index
            ea = edge_attr

        h = self.node_stem(x)
        u = self.edge_stem(ea)

        attn_tuple: tuple[torch.Tensor, torch.Tensor] | None = None
        num_layers = len(self.conv_layers)

        for idx, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            is_last = idx == (num_layers - 1)
            if is_last and return_attention_weights:
                conv_out, attn_tuple = conv(
                    h, ei, edge_attr=u, return_attention_weights=True
                )
            else:
                conv_out = conv(h, ei, edge_attr=u)
            h = norm(h + F.elu(conv_out))

        if return_attention_weights:
            assert attn_tuple is not None
            return h, u, attn_tuple
        return h, u
