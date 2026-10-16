"""Classical exploration baseline agents and learned policy wrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Data

from config import ModelConfig
from models.pointer_actor_critic import PointerActorCritic


def _extract_agent_to_frontier_geo_dist(graph_data: Data) -> torch.Tensor:
    """Extract normalized geodesic distances from Node 0 (agent) to each frontier node."""
    if (
        hasattr(graph_data, "agent_to_frontier_edge_attr")
        and graph_data.agent_to_frontier_edge_attr is not None
        and graph_data.agent_to_frontier_edge_attr.numel() > 0
    ):
        return graph_data.agent_to_frontier_edge_attr[:, 0]

    num_nodes = int(graph_data.x.size(0))
    num_frontiers = max(0, num_nodes - 1)
    if num_frontiers == 0:
        return torch.empty(0, dtype=torch.float32, device=graph_data.x.device)

    geo_dists = torch.full(
        (num_frontiers,), 2.0, dtype=torch.float32, device=graph_data.x.device
    )
    if (
        hasattr(graph_data, "edge_index")
        and graph_data.edge_index is not None
        and hasattr(graph_data, "edge_attr")
        and graph_data.edge_attr is not None
        and graph_data.edge_index.numel() > 0
    ):
        src, dst = graph_data.edge_index[0], graph_data.edge_index[1]
        mask = (src == 0) & (dst >= 1) & (dst <= num_frontiers)
        valid_dst = dst[mask] - 1
        geo_dists[valid_dst] = graph_data.edge_attr[mask, 0].to(geo_dists.dtype)
    return geo_dists


class BaseExplorationAgent(ABC):
    """Common abstract interface for classical and learned exploration agents."""

    @abstractmethod
    def select_action(self, graph_data: Data, deterministic: bool = True) -> int:
        """Return local frontier index in {0, ..., K_t - 1}."""
        pass


class NearestFrontierAgent(BaseExplorationAgent):
    """Selects the frontier with the minimum normalized geodesic distance from the agent."""

    def select_action(self, graph_data: Data, deterministic: bool = True) -> int:
        """Return argmin_i data.agent_to_frontier_edge_attr[i, 0]."""
        geo_dists = _extract_agent_to_frontier_geo_dist(graph_data)
        if geo_dists.numel() == 0:
            return 0
        return int(torch.argmin(geo_dists).item())


class UtilityGreedyAgent(BaseExplorationAgent):
    """Selects frontier maximizing unmapped_est_i / exp(lambda * geo_dist_i)."""

    def __init__(self, lam: float = 4.0) -> None:
        self.lam = float(lam)

    def select_action(self, graph_data: Data, deterministic: bool = True) -> int:
        """Return argmax_i unmapped_est_i / exp(lam * geo_dist_i)."""
        geo_dists = _extract_agent_to_frontier_geo_dist(graph_data)
        if geo_dists.numel() == 0:
            return 0

        if hasattr(graph_data, "is_frontier") and graph_data.is_frontier is not None:
            unmapped_est = graph_data.x[graph_data.is_frontier.bool(), 7]
        else:
            unmapped_est = graph_data.x[1:, 7]

        utilities = unmapped_est / torch.exp(self.lam * geo_dists)
        return int(torch.argmax(utilities).item())


class MyopicTSPAgent(BaseExplorationAgent):
    """Solves a greedy TSP tour over V_t and selects the first frontier after Node 0."""

    def select_action(self, graph_data: Data, deterministic: bool = True) -> int:
        """Build distance matrix from edge_attr[:, 0], solve greedy_tsp from Node 0."""
        num_nodes = int(graph_data.x.size(0))
        num_frontiers = num_nodes - 1
        if num_frontiers <= 0:
            return 0
        if num_frontiers == 1:
            return 0

        # Build pairwise distance matrix over V_t = {0, ..., K_t}.
        # Initialize with fallback weight 2.0 (plus Euclidean relative distance) so that
        # sparse k-NN graphs (when K_t + 1 > 16) still form a valid complete graph.
        rel_coords = graph_data.x[:, 1:3].detach().cpu()
        pdist = torch.cdist(rel_coords, rel_coords, p=2.0)
        dist_matrix = 2.0 + pdist
        dist_matrix.fill_diagonal_(0.0)

        if (
            hasattr(graph_data, "edge_index")
            and graph_data.edge_index is not None
            and hasattr(graph_data, "edge_attr")
            and graph_data.edge_attr is not None
            and graph_data.edge_index.numel() > 0
        ):
            edge_index_cpu = graph_data.edge_index.detach().cpu()
            edge_weights_cpu = graph_data.edge_attr[:, 0].detach().cpu().to(torch.float32)
            src = edge_index_cpu[0]
            dst = edge_index_cpu[1]
            dist_matrix[src, dst] = edge_weights_cpu

        # Construct complete directed weighted graph over V_t
        tsp_graph = nx.DiGraph()
        tsp_graph.add_nodes_from(range(num_nodes))
        dist_list = dist_matrix.tolist()
        weighted_edges = [
            (u, v, float(dist_list[u][v]))
            for u in range(num_nodes)
            for v in range(num_nodes)
            if u != v
        ]
        tsp_graph.add_weighted_edges_from(weighted_edges, weight="weight")

        tour = nx.approximation.greedy_tsp(tsp_graph, weight="weight", source=0)
        first_frontier_node = int(tour[1])
        local_idx = first_frontier_node - 1
        return max(0, min(num_frontiers - 1, local_idx))


class LearnedPolicyAgent(BaseExplorationAgent):
    """Exploration agent wrapper around a trained or initialized PointerActorCritic."""

    def __init__(
        self,
        model: PointerActorCritic | nn.Module | None = None,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        config: ModelConfig | None = None,
    ) -> None:
        self.device = torch.device(device)
        if model is None:
            cfg = config if config is not None else ModelConfig()
            self.model: nn.Module = PointerActorCritic(cfg)
        else:
            self.model = model

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"])
            else:
                self.model.load_state_dict(ckpt)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def select_action(self, graph_data: Data, deterministic: bool = True) -> int:
        """Select local frontier index in {0, ..., K_t - 1} using PointerActorCritic."""
        if hasattr(self.model, "select_action"):
            return int(self.model.select_action(graph_data, deterministic=deterministic))

        param_device = next(self.model.parameters()).device
        if graph_data.x.device != param_device:
            graph_data = graph_data.to(param_device)

        probs, _, _, _, _ = self.model(graph_data, return_attn=False)
        if probs.numel() == 0:
            return 0
        if deterministic:
            return int(torch.argmax(probs).item())
        dist = Categorical(probs=probs)
        return int(dist.sample().item())
