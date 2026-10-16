"""Proximal Policy Optimization (PPO) trainer for variable-action-space frontier graphs."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data

from config import PPOConfig
from models.pointer_actor_critic import PointerActorCritic
from rl.buffer import RolloutBuffer


class PPOTrainer:
    """Clipped surrogate PPO trainer over heterogeneous PyG frontier graphs.

    Implements Section 3.4.2 of SPECS.md with vectorized action evaluation via
    ``torch.bincount`` offsets and per-graph entropy via ``scatter_add``.
    """

    def __init__(
        self,
        model: PointerActorCritic,
        config: PPOConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config if config is not None else PPOConfig()
        if device is not None:
            self.device = torch.device(device)
            self.model = model.to(self.device)
        else:
            self.device = next(model.parameters()).device
            self.model = model

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    @torch.no_grad()
    def select_action_and_value(
        self, graph_data: Data, deterministic: bool = False
    ) -> tuple[int, float, float]:
        """Sample an action for a single graph and return (action, log_prob, value)."""
        if graph_data.x.device != self.device:
            graph_data = graph_data.to(self.device)

        probs, _, _, values, _ = self.model(graph_data, return_attn=False)
        if deterministic:
            action_idx = int(torch.argmax(probs).item())
        else:
            dist = Categorical(probs=probs)
            action_idx = int(dist.sample().item())

        log_prob = float(torch.log(probs[action_idx] + 1e-8).item())
        value = float(values.squeeze().item())
        return action_idx, log_prob, value

    def evaluate_actions(
        self, batched_data: Data | Batch, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probabilities, state values, and per-graph entropies for a batch of graphs.

        Follows Section 3.4.2 of SPECS.md:
        1. Compute frontier count offsets ``ptr`` from ``frontier_batch_vec`` via ``torch.bincount`` cumulative sum.
        2. Convert local action indices ``a_b`` into global indices ``g_b = ptr[b] + a_b``.
        3. Index ``new_log_probs = torch.log(probs[g_b] + 1e-8)``.
        4. Compute per-graph entropy via ``scatter_add``.
        """
        probs, _, frontier_batch_vec, values, _ = self.model(
            batched_data, return_attn=False
        )
        num_graphs = int(values.size(0))

        # 1. Frontier count offsets ptr from frontier_batch_vec (bincount cumulative sum)
        counts = torch.bincount(frontier_batch_vec, minlength=num_graphs)
        ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=probs.device)
        ptr[1:] = torch.cumsum(counts, dim=0)

        # 2. Convert local action indices a_b in {0, ..., K_b - 1} to global indices g_b
        global_actions = ptr[:-1] + actions.to(dtype=torch.long, device=probs.device)

        # 3. Extract action log-probabilities
        new_log_probs = torch.log(probs[global_actions] + 1e-8)

        # 4. Per-graph entropy via scatter_add: H_b = -sum_{i in Frontiers(b)} p_i log(p_i + 1e-8)
        elementwise_entropy = -probs * torch.log(probs + 1e-8)
        entropy_per_graph = torch.zeros(
            num_graphs, dtype=probs.dtype, device=probs.device
        ).scatter_add(0, frontier_batch_vec, elementwise_entropy)

        return new_log_probs, values, entropy_per_graph

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO epochs over the rollout buffer and return training metrics."""
        num_samples = len(buffer)
        if num_samples == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "total_loss": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
            }

        if (
            buffer.advantages.numel() != num_samples
            or buffer.returns.numel() != num_samples
        ):
            buffer.compute_returns_and_advantages(
                last_value=0.0,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )

        self.model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss_val = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        num_updates = 0

        batch_size = max(1, min(self.config.batch_size, num_samples))

        for epoch in range(self.config.ppo_epochs):
            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_graphs = [buffer.graphs[i] for i in batch_idx.tolist()]
                batched_data = Batch.from_data_list(batch_graphs).to(self.device)

                batch_actions = buffer.actions[batch_idx].to(self.device)
                batch_old_log_probs = buffer.log_probs[batch_idx].to(self.device)
                batch_returns = buffer.returns[batch_idx].to(self.device)
                batch_advantages = buffer.advantages[batch_idx].to(self.device)

                new_log_probs, values, entropy_per_graph = self.evaluate_actions(
                    batched_data, batch_actions
                )

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_eps,
                        1.0 + self.config.clip_eps,
                    )
                    * batch_advantages
                )

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - batch_returns) ** 2).mean()
                entropy_mean = entropy_per_graph.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_mean
                )

                active_unclipped = (surr1 <= surr2) & (batch_advantages.abs() > 1e-7)
                if epoch == 0 or bool(active_unclipped.any().item()):
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean()
                    clip_frac = (
                        ((ratio - 1.0).abs() > self.config.clip_eps).float().mean()
                    )

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy_mean.item())
                total_loss_val += float(loss.item())
                total_approx_kl += float(approx_kl.item())
                total_clip_frac += float(clip_frac.item())
                num_updates += 1

        denom = max(1, num_updates)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
            "total_loss": total_loss_val / denom,
            "approx_kl": total_approx_kl / denom,
            "clip_fraction": total_clip_frac / denom,
        }
