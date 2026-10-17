"""Heterogeneous graph rollout buffer for variable-size PyG Data transitions."""

from __future__ import annotations

import torch
from torch_geometric.data import Data


class RolloutBuffer:
    """Rollout buffer storing variable-size PyG Data graphs on CPU and RL scalars as 1D tensors.

    Implements Section 3.4.1 of SPECS.md.
    """

    def __init__(self) -> None:
        self.graphs: list[Data] = []
        self._actions: list[int] = []
        self._log_probs: list[float] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._values: list[float] = []

        self._actions_tensor: torch.Tensor | None = None
        self._log_probs_tensor: torch.Tensor | None = None
        self._rewards_tensor: torch.Tensor | None = None
        self._dones_tensor: torch.Tensor | None = None
        self._values_tensor: torch.Tensor | None = None

        self.returns: torch.Tensor = torch.empty(0, dtype=torch.float32)
        self.advantages: torch.Tensor = torch.empty(0, dtype=torch.float32)

    def _invalidate_cache(self) -> None:
        self._actions_tensor = None
        self._log_probs_tensor = None
        self._rewards_tensor = None
        self._dones_tensor = None
        self._values_tensor = None

    @property
    def actions(self) -> torch.Tensor:
        if self._actions_tensor is None:
            self._actions_tensor = torch.tensor(self._actions, dtype=torch.long)
        return self._actions_tensor

    @actions.setter
    def actions(self, value: torch.Tensor) -> None:
        self._actions_tensor = value.detach().cpu().to(dtype=torch.long)
        self._actions = self._actions_tensor.tolist()

    @property
    def log_probs(self) -> torch.Tensor:
        if self._log_probs_tensor is None:
            self._log_probs_tensor = torch.tensor(self._log_probs, dtype=torch.float32)
        return self._log_probs_tensor

    @log_probs.setter
    def log_probs(self, value: torch.Tensor) -> None:
        self._log_probs_tensor = value.detach().cpu().to(dtype=torch.float32)
        self._log_probs = self._log_probs_tensor.tolist()

    @property
    def rewards(self) -> torch.Tensor:
        if self._rewards_tensor is None:
            self._rewards_tensor = torch.tensor(self._rewards, dtype=torch.float32)
        return self._rewards_tensor

    @rewards.setter
    def rewards(self, value: torch.Tensor) -> None:
        self._rewards_tensor = value.detach().cpu().to(dtype=torch.float32)
        self._rewards = self._rewards_tensor.tolist()

    @property
    def dones(self) -> torch.Tensor:
        if self._dones_tensor is None:
            self._dones_tensor = torch.tensor(self._dones, dtype=torch.bool)
        return self._dones_tensor

    @dones.setter
    def dones(self, value: torch.Tensor) -> None:
        self._dones_tensor = value.detach().cpu().to(dtype=torch.bool)
        self._dones = self._dones_tensor.tolist()

    @property
    def values(self) -> torch.Tensor:
        if self._values_tensor is None:
            self._values_tensor = torch.tensor(self._values, dtype=torch.float32)
        return self._values_tensor

    @values.setter
    def values(self, value: torch.Tensor) -> None:
        self._values_tensor = value.detach().cpu().to(dtype=torch.float32)
        self._values = self._values_tensor.tolist()

    def add(
        self,
        graph: Data,
        action: int | torch.Tensor,
        log_prob: float | torch.Tensor,
        reward: float | torch.Tensor,
        done: bool | torch.Tensor,
        value: float | torch.Tensor,
    ) -> None:
        """Append a single transition to the rollout buffer (storing graph on CPU)."""
        if hasattr(graph, "is_frontier") and int(graph.is_frontier.sum().item()) == 0:
            raise ValueError("Cannot add a terminal graph with 0 frontiers to RolloutBuffer.")
        self.graphs.append(graph.detach().cpu())
        self._actions.append(
            int(action.item()) if isinstance(action, torch.Tensor) else int(action)
        )
        self._log_probs.append(
            float(log_prob.item())
            if isinstance(log_prob, torch.Tensor)
            else float(log_prob)
        )
        self._rewards.append(
            float(reward.item()) if isinstance(reward, torch.Tensor) else float(reward)
        )
        self._dones.append(
            bool(done.item()) if isinstance(done, torch.Tensor) else bool(done)
        )
        self._values.append(
            float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        )
        self._invalidate_cache()

    def clear(self) -> None:
        """Reset all stored transitions and computed returns/advantages."""
        self.graphs.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._rewards.clear()
        self._dones.clear()
        self._values.clear()
        self._invalidate_cache()
        self.returns = torch.empty(0, dtype=torch.float32)
        self.advantages = torch.empty(0, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.graphs)

    def compute_returns_and_advantages(
        self,
        last_value: float | torch.Tensor = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns, normalizing advantages across the rollout batch."""
        num_steps = len(self.graphs)
        if num_steps == 0:
            self.returns = torch.empty(0, dtype=torch.float32)
            self.advantages = torch.empty(0, dtype=torch.float32)
            return self.returns, self.advantages

        last_val = (
            float(last_value.item())
            if isinstance(last_value, torch.Tensor)
            else float(last_value)
        )
        rewards = self.rewards
        dones = self.dones
        values = self.values

        advantages = torch.zeros(num_steps, dtype=torch.float32)
        last_gae_lam = 0.0

        for t in reversed(range(num_steps)):
            next_non_terminal = 0.0 if bool(dones[t].item()) else 1.0
            next_value = last_val if t == num_steps - 1 else float(values[t + 1].item())
            delta = (
                float(rewards[t].item())
                + gamma * next_value * next_non_terminal
                - float(values[t].item())
            )
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        self.returns = advantages + values

        if num_steps > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-5 and (advantages.max() - advantages.min()) > 1e-5:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        self.advantages = advantages
        return self.returns, self.advantages
