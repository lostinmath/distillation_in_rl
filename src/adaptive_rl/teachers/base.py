"""Base class for teacher policies in teacher-guided RL."""

from abc import ABC, abstractmethod

import numpy as np
import torch


class TeacherPolicy(ABC):
    """Abstract base class for teacher policies.

    Teacher policies provide guidance to student policies during training.
    They can be pre-trained models, hand-coded heuristics, or random policies.
    """

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        """Initialize teacher policy.

        Args:
            action_space: Gym action space
            observation_space: Gym observation space
            device: Device to run on (cpu/cuda)
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.device = device

    @abstractmethod
    def act(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Generate actions for given observations.

        Args:
            obs: Observations from environment (batch_size, obs_dim) or (obs_dim,)

        Returns:
            Actions to take (batch_size, action_dim) or (action_dim,)
        """

    def reset(self):
        """Reset internal state if needed.

        Some teachers might have internal state (e.g., RNNs).
        Override this method if needed.
        """

    def to(self, device):
        """Move teacher to specified device.

        Args:
            device: Device to move to
        """
        self.device = device
        return self
