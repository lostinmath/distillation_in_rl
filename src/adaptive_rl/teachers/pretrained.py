"""Pretrained teacher policy that loads a saved PPO model."""

import numpy as np
import torch
from torch import nn

from .base import TeacherPolicy


class PretrainedPPOTeacher(TeacherPolicy):
    """Teacher policy that uses a pretrained PPO model.

    Loads a saved PPO checkpoint and uses it to generate actions.
    """

    def __init__(
        self,
        checkpoint_path: str,
        action_space=None,
        observation_space=None,
        device="cpu",
        deterministic: bool = True,
    ):
        """Initialize pretrained PPO teacher.

        Args:
            checkpoint_path: Path to saved PPO model checkpoint
            action_space: Gym action space
            observation_space: Gym observation space
            device: Device to run on
            deterministic: Whether to use deterministic actions (no sampling)
        """
        super().__init__(action_space, observation_space, device)

        self.deterministic = deterministic
        self.checkpoint_path = checkpoint_path

        # Load the model
        self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: str):
        """Load PPO model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Checkpoint contains state dict and metadata
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            # Checkpoint is just the model
            self.model = checkpoint.to(self.device)
            self.model.eval()
            return

        # Create model architecture (will be replaced with actual architecture)
        # This is a placeholder - actual implementation would recreate the PPO network
        obs_dim = (
            state_dict["actor.0.weight"].shape[1]
            if "actor.0.weight" in state_dict
            else 4
        )
        self.model = self._create_placeholder_model(obs_dim, state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _create_placeholder_model(self, obs_dim: int, state_dict: dict) -> nn.Module:
        """Create a placeholder model architecture.

        This should be replaced with the actual PPO architecture used during training.

        Args:
            obs_dim: Observation dimension
            state_dict: State dictionary to infer architecture from

        Returns:
            Neural network model
        """
        # Infer action dimension from state dict
        for key in state_dict:
            if "actor" in key and "weight" in key and len(state_dict[key].shape) == 2:
                action_dim = state_dict[key].shape[0]
                break
        else:
            action_dim = 2  # Default for CartPole

        # Simple MLP (should match training architecture)
        class SimplePPOActor(nn.Module):
            def __init__(self, obs_dim, action_dim):
                super().__init__()
                self.actor = nn.Sequential(
                    nn.Linear(obs_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, action_dim),
                )

            def forward(self, x):
                return self.actor(x)

        return SimplePPOActor(obs_dim, action_dim)

    def act(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Generate actions using the pretrained model.

        Args:
            obs: Observations from environment

        Returns:
            Actions from the pretrained policy
        """
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            return_numpy = True
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
        else:
            return_numpy = False
            obs_tensor = obs.float()

        # Handle both single and batched observations
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False

        # Generate actions
        with torch.no_grad():
            if hasattr(self.model, "get_action"):
                # Model has a get_action method
                actions = self.model.get_action(
                    obs_tensor, deterministic=self.deterministic
                )
            else:
                # Direct forward pass
                logits = self.model(obs_tensor)

                if self.deterministic:
                    # Take argmax for discrete actions
                    if len(logits.shape) > 1 and logits.shape[1] > 1:
                        actions = torch.argmax(logits, dim=-1)
                    else:
                        actions = (logits > 0).long().squeeze(-1)
                # Sample from distribution
                elif len(logits.shape) > 1 and logits.shape[1] > 1:
                    probs = torch.softmax(logits, dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    actions = (
                        (torch.rand_like(logits) < torch.sigmoid(logits))
                        .long()
                        .squeeze(-1)
                    )

        # Handle single observation
        if single_obs:
            actions = actions[0]

        # Convert back to numpy if needed
        if return_numpy:
            actions = actions.cpu().numpy()

        return actions

    def to(self, device):
        """Move teacher to specified device."""
        self.device = device
        if hasattr(self, "model"):
            self.model = self.model.to(device)
        return self
