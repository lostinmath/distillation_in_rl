"""Random teacher policy for baseline comparisons."""

import numpy as np
import torch

from .base import TeacherPolicy


class RandomTeacher(TeacherPolicy):
    """Random action teacher policy.

    Serves as a baseline to compare against more sophisticated teachers.
    """

    def __init__(self, action_space, observation_space=None, device="cpu"):
        """Initialize random teacher.

        Args:
            action_space: Gym action space
            observation_space: Gym observation space (not used)
            device: Device to run on
        """
        super().__init__(action_space, observation_space, device)

        # Determine if discrete or continuous action space
        self.discrete = hasattr(action_space, "n")

        if self.discrete:
            self.n_actions = action_space.n
        else:
            self.action_low = torch.tensor(action_space.low, device=device)
            self.action_high = torch.tensor(action_space.high, device=device)
            self.action_shape = action_space.shape

    def act(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Generate random actions.

        Args:
            obs: Observations (batch_size, obs_dim) or (obs_dim,)

        Returns:
            Random actions
        """
        # Handle both batched and single observations
        if isinstance(obs, np.ndarray):
            return_numpy = True
            obs_tensor = torch.from_numpy(obs).to(self.device)
        else:
            return_numpy = False
            obs_tensor = obs

        # Determine batch size
        if len(obs_tensor.shape) == 1:
            batch_size = 1
            single_obs = True
        else:
            batch_size = obs_tensor.shape[0]
            single_obs = False

        # Generate random actions
        if self.discrete:
            actions = torch.randint(
                0, self.n_actions, (batch_size,), device=self.device
            )
        else:
            # Sample uniformly from action space
            actions = torch.rand((batch_size, *self.action_shape), device=self.device)
            actions = self.action_low + (self.action_high - self.action_low) * actions

        # Handle single observation case
        if single_obs:
            actions = actions[0]

        # Convert back to numpy if needed
        if return_numpy:
            actions = actions.cpu().numpy()

        return actions
