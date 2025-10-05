"""Behavioral Cloning (BC) implementation.

Critical baseline for teacher-guided RL research.
Pure imitation learning from teacher demonstrations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger

from .base import Algorithm, AlgorithmRegistry


@AlgorithmRegistry.register("bc")
class BehavioralCloning(Algorithm):
    """Behavioral Cloning algorithm for pure imitation learning."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 10,
        network: nn.Module = None,
        device: str = "cuda",
        seed: int = None,
        # BC-specific parameters
        label_smoothing: float = 0.0,
        l2_reg: float = 0.0,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, device, seed)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.label_smoothing = label_smoothing
        self.l2_reg = l2_reg

        # Handle action space
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n
            self.discrete_actions = True
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.action_dim = action_space.nvec[0]  # Assume same for all envs
            self.discrete_actions = True
        else:
            self.action_dim = np.prod(action_space.shape)
            self.discrete_actions = False

        # Handle observation space
        if len(observation_space.shape) == 2:
            self.obs_dim = observation_space.shape[1]
        else:
            self.obs_dim = np.prod(observation_space.shape)

        # Create policy network
        if network is None:
            from adaptive_rl.networks.mlp import MLPNetwork
            self.policy = MLPNetwork(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_sizes=[256, 256],  # Larger for BC
                activation="relu",
                discrete=self.discrete_actions,
                dropout_rate=dropout_rate,
            ).to(device)
        else:
            self.policy = network.to(device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=l2_reg,
        )

        # Track training statistics
        self.training_stats = {
            "total_samples": 0,
            "total_updates": 0,
            "best_loss": float("inf"),
        }

    def collect_demonstrations(
        self, teacher, env, n_episodes: int = 100, max_steps: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect teacher demonstrations for training."""
        logger.info(f"Collecting {n_episodes} demonstrations from teacher")

        observations = []
        actions = []

        for episode in range(n_episodes):
            obs = env.reset()[0]
            episode_obs = []
            episode_actions = []

            for step in range(max_steps):
                # Get teacher action
                teacher_action = teacher.act(obs)

                episode_obs.append(obs.copy())
                episode_actions.append(teacher_action.copy())

                # Environment step
                obs, reward, terminated, truncated, info = env.step(teacher_action)
                done = terminated.any() if hasattr(terminated, 'any') else terminated

                if done:
                    break

            observations.extend(episode_obs)
            actions.extend(episode_actions)

            if (episode + 1) % 20 == 0:
                logger.info(f"Collected {episode + 1}/{n_episodes} episodes")

        observations = np.array(observations)
        actions = np.array(actions)

        logger.info(f"Collected {len(observations)} demonstration samples")
        return observations, actions

    def train_from_demonstrations(
        self, observations: np.ndarray, actions: np.ndarray
    ) -> Dict[str, float]:
        """Train BC policy from collected demonstrations."""
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)

        # Handle action dimensions
        if self.discrete_actions and actions.dim() > 1:
            actions = actions.squeeze()

        dataset_size = len(observations)
        n_batches = (dataset_size + self.batch_size - 1) // self.batch_size

        total_losses = []
        accuracy_scores = []

        for epoch in range(self.n_epochs):
            epoch_losses = []
            epoch_accuracies = []

            # Shuffle data
            indices = torch.randperm(dataset_size, device=self.device)

            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]

                # Forward pass
                if self.discrete_actions:
                    logits = self.policy(batch_obs)

                    # Compute loss with optional label smoothing
                    if self.label_smoothing > 0:
                        loss = self._label_smoothed_cross_entropy(logits, batch_actions)
                    else:
                        loss = F.cross_entropy(logits, batch_actions)

                    # Compute accuracy
                    predicted = torch.argmax(logits, dim=1)
                    accuracy = (predicted == batch_actions).float().mean()
                    epoch_accuracies.append(accuracy.item())

                else:
                    # Continuous actions - MSE loss
                    predicted_actions = self.policy(batch_obs)
                    loss = F.mse_loss(predicted_actions, batch_actions)

                    # Compute "accuracy" as fraction within tolerance
                    tolerance = 0.1  # 10% tolerance
                    accuracy = (
                        torch.abs(predicted_actions - batch_actions) < tolerance
                    ).float().mean()
                    epoch_accuracies.append(accuracy.item())

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            total_losses.append(avg_loss)
            accuracy_scores.append(avg_accuracy)

            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}: Loss={avg_loss:.4f}, Acc={avg_accuracy:.3f}")

        self.training_stats["total_samples"] += dataset_size * self.n_epochs
        self.training_stats["total_updates"] += n_batches * self.n_epochs
        self.training_stats["best_loss"] = min(self.training_stats["best_loss"], min(total_losses))

        return {
            "bc/loss": np.mean(total_losses),
            "bc/final_loss": total_losses[-1],
            "bc/accuracy": np.mean(accuracy_scores),
            "bc/final_accuracy": accuracy_scores[-1],
            "bc/total_samples": self.training_stats["total_samples"],
        }

    def _label_smoothed_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross entropy."""
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # One-hot encode targets
        targets_one_hot = torch.zeros_like(logits)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Apply label smoothing
        smooth_targets = (1 - self.label_smoothing) * targets_one_hot + \
                        self.label_smoothing / n_classes

        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss

    def predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = True,  # BC is always deterministic
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Predict action using trained BC policy."""
        with torch.no_grad():
            if self.discrete_actions:
                logits = self.policy(observation)
                action = torch.argmax(logits, dim=-1)
                # Fake value and log_prob for compatibility
                value = torch.zeros(observation.size(0), 1, device=self.device)
                log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)
            else:
                action = self.policy(observation)
                value = torch.zeros(observation.size(0), 1, device=self.device)
                log_prob = None  # Not applicable for continuous actions

        return action, value, log_prob

    def train_step(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train step for compatibility - BC doesn't use rollout data."""
        # BC training happens in train_from_demonstrations
        # This is just for interface compatibility
        return {
            "bc/info": 0.0,  # No training in this step
            "bc/total_samples": self.training_stats["total_samples"],
        }

    def evaluate_teacher_imitation(
        self, teacher, env, n_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate how well BC policy imitates teacher."""
        total_agreement = 0
        total_steps = 0

        for episode in range(n_episodes):
            obs = env.reset()[0]
            episode_steps = 0
            episode_agreement = 0

            while episode_steps < 1000:  # Max episode length
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

                # Get BC action
                bc_action, _, _ = self.predict(obs_tensor)
                bc_action_np = bc_action.cpu().numpy()

                # Get teacher action
                teacher_action = teacher.act(obs)

                # Check agreement
                if self.discrete_actions:
                    agreement = np.mean(bc_action_np == teacher_action)
                else:
                    # For continuous, check if within 10% tolerance
                    tolerance = 0.1
                    agreement = np.mean(np.abs(bc_action_np - teacher_action) < tolerance)

                episode_agreement += agreement
                episode_steps += 1

                # Step environment with teacher action (for consistency)
                obs, _, terminated, truncated, _ = env.step(teacher_action)
                done = terminated.any() if hasattr(terminated, 'any') else terminated

                if done:
                    break

            total_agreement += episode_agreement
            total_steps += episode_steps

        avg_agreement = total_agreement / total_steps
        return {
            "bc/teacher_agreement": avg_agreement,
            "bc/evaluation_episodes": n_episodes,
        }

    def save(self, path: str) -> None:
        """Save BC model."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "label_smoothing": self.label_smoothing,
                "l2_reg": self.l2_reg,
                "discrete_actions": self.discrete_actions,
            },
        }, path)

    def load(self, path: str) -> None:
        """Load BC model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]