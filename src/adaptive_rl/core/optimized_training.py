"""Optimized training utilities for PPO.

Fixes performance anti-patterns:
- Vectorized batch processing
- Pre-allocated tensors
- Efficient data shuffling
- Memory-efficient rollout collection
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class RolloutBatch:
    """Efficient rollout data structure with pre-allocated tensors."""
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    next_observations: torch.Tensor

    def to_device(self, device: torch.device) -> "RolloutBatch":
        """Move all tensors to specified device."""
        return RolloutBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            values=self.values.to(device),
            log_probs=self.log_probs.to(device),
            next_observations=self.next_observations.to(device),
        )


class OptimizedRolloutCollector:
    """Memory-efficient rollout collection with pre-allocated buffers."""

    def __init__(self, n_steps: int, n_envs: int, obs_shape: Tuple[int, ...], device: torch.device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device

        # Pre-allocate all tensors
        self.observations = torch.zeros((n_steps, n_envs, *obs_shape), device=device)
        self.actions = torch.zeros((n_steps, n_envs), dtype=torch.long, device=device)
        self.rewards = torch.zeros((n_steps, n_envs), device=device)
        self.dones = torch.zeros((n_steps, n_envs), dtype=torch.bool, device=device)
        self.values = torch.zeros((n_steps, n_envs), device=device)
        self.log_probs = torch.zeros((n_steps, n_envs), device=device)

        self.step_idx = 0

    def reset(self):
        """Reset collector for new rollout."""
        self.step_idx = 0

    def add_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        """Add a single step to the rollout (in-place)."""
        if self.step_idx >= self.n_steps:
            raise ValueError(f"Rollout full: {self.step_idx}/{self.n_steps}")

        self.observations[self.step_idx] = obs
        self.actions[self.step_idx] = action.squeeze() if action.dim() > 1 else action
        self.rewards[self.step_idx] = reward
        self.dones[self.step_idx] = done
        self.values[self.step_idx] = value.squeeze() if value.dim() > 1 else value
        self.log_probs[self.step_idx] = log_prob

        self.step_idx += 1

    def get_batch(self, next_obs: torch.Tensor) -> RolloutBatch:
        """Get the completed rollout batch."""
        if self.step_idx != self.n_steps:
            raise ValueError(f"Incomplete rollout: {self.step_idx}/{self.n_steps}")

        return RolloutBatch(
            observations=self.observations.clone(),
            actions=self.actions.clone(),
            rewards=self.rewards.clone(),
            dones=self.dones.clone(),
            values=self.values.clone(),
            log_probs=self.log_probs.clone(),
            next_observations=next_obs,
        )


class OptimizedBatchSampler:
    """Efficient batch sampling without redundant shuffling."""

    def __init__(self, dataset_size: int, batch_size: int, n_epochs: int, device: torch.device):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device

        # Pre-generate all indices for all epochs
        self.all_indices = self._generate_all_indices()

    def _generate_all_indices(self) -> torch.Tensor:
        """Pre-generate shuffled indices for all epochs."""
        indices_per_epoch = []

        for _ in range(self.n_epochs):
            epoch_indices = torch.randperm(self.dataset_size, device=self.device)
            indices_per_epoch.append(epoch_indices)

        return torch.stack(indices_per_epoch)  # (n_epochs, dataset_size)

    def get_batches(self, epoch: int) -> List[torch.Tensor]:
        """Get all batch indices for a specific epoch."""
        epoch_indices = self.all_indices[epoch]

        batches = []
        for start_idx in range(0, self.dataset_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.dataset_size)
            batch_indices = epoch_indices[start_idx:end_idx]
            batches.append(batch_indices)

        return batches


def optimize_tensor_operations(
    observations: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    batch_indices: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Vectorized tensor indexing - much faster than individual indexing."""
    return (
        observations[batch_indices],
        actions[batch_indices],
        advantages[batch_indices],
        returns[batch_indices],
        old_values[batch_indices],
    )


class PerformanceMonitor:
    """Monitor training performance and detect issues."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: Dict[str, List[float]] = {}

    def record(self, metrics: Dict[str, float]):
        """Record metrics and detect anomalies."""
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []

            self.metrics_history[key].append(value)

            # Keep only recent history
            if len(self.metrics_history[key]) > self.window_size:
                self.metrics_history[key] = self.metrics_history[key][-self.window_size:]

        self._check_anomalies(metrics)

    def _check_anomalies(self, current_metrics: Dict[str, float]):
        """Check for training anomalies."""
        for key, value in current_metrics.items():
            if key not in self.metrics_history:
                continue

            history = self.metrics_history[key]
            if len(history) < 10:  # Need some history
                continue

            # Check for NaN/Inf
            if not torch.isfinite(torch.tensor(value)):
                logger.error(f"Non-finite value detected in {key}: {value}")

            # Check for extreme values
            if len(history) > 20:
                recent_mean = np.mean(history[-20:])
                if abs(value - recent_mean) > 10 * np.std(history[-20:]):
                    logger.warning(f"Anomalous value in {key}: {value} (recent mean: {recent_mean:.3f})")

    def get_smoothed_metrics(self, window: int = 10) -> Dict[str, float]:
        """Get smoothed recent metrics."""
        smoothed = {}
        for key, history in self.metrics_history.items():
            if len(history) >= window:
                smoothed[f"{key}_smooth"] = np.mean(history[-window:])
        return smoothed


def validate_tensor_shapes(tensors: Dict[str, torch.Tensor], expected_shapes: Dict[str, Tuple[int, ...]]):
    """Validate tensor shapes to catch bugs early."""
    for name, tensor in tensors.items():
        if name in expected_shapes:
            expected = expected_shapes[name]
            actual = tensor.shape

            # Allow flexible batch dimension (first dim)
            if len(expected) > 1 and len(actual) > 1:
                if actual[1:] != expected[1:]:
                    raise ValueError(f"{name} shape mismatch: expected {expected}, got {actual}")
            elif actual != expected:
                raise ValueError(f"{name} shape mismatch: expected {expected}, got {actual}")


class SafeTensorOps:
    """Safe tensor operations with error handling."""

    @staticmethod
    def safe_stack(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Stack tensors with shape validation."""
        if not tensors:
            raise ValueError("Cannot stack empty tensor list")

        shapes = [t.shape for t in tensors]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(f"Cannot stack tensors with different shapes: {shapes}")

        try:
            return torch.stack(tensors, dim=dim)
        except RuntimeError as e:
            logger.error(f"Failed to stack tensors: {e}")
            logger.error(f"Tensor shapes: {shapes}")
            raise

    @staticmethod
    def safe_cat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Concatenate tensors with validation."""
        if not tensors:
            raise ValueError("Cannot concatenate empty tensor list")

        try:
            return torch.cat(tensors, dim=dim)
        except RuntimeError as e:
            shapes = [t.shape for t in tensors]
            logger.error(f"Failed to concatenate tensors: {e}")
            logger.error(f"Tensor shapes: {shapes}")
            raise

    @staticmethod
    def safe_normalize(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize tensor with numerical stability."""
        mean = tensor.mean()
        std = tensor.std()

        if std < eps:
            logger.warning(f"Low standard deviation in normalization: {std}")
            return tensor - mean

        return (tensor - mean) / (std + eps)