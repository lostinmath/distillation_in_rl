"""Abstract base class for RL agents."""

from abc import ABC, abstractmethod
from typing import Any
from beartype import beartype
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    @beartype
    @abstractmethod
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """Get action from observation.

        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy

        Returns:
            Action to take
        """
        pass

    @beartype
    @abstractmethod
    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Perform one training step.

        Args:
            batch: Training batch data

        Returns:
            Training metrics
        """
        pass

    @beartype
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent to file."""
        pass

    @beartype
    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent from file."""
        pass