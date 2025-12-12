"""Abstract base class for policy schedulers."""

from abc import ABC, abstractmethod
from beartype import beartype


class BaseScheduler(ABC):
    """Abstract base class for policy scheduling strategies."""

    @beartype
    def __init__(self, num_envs: int = 1):
        self.num_envs = num_envs

    @beartype
    @abstractmethod
    def choose_policy(self, env_idx: int, prev_reward: float) -> str:
        """Choose policy type for given environment and previous reward.

        Args:
            env_idx: Environment index
            prev_reward: Previous step reward (-1 if environment reset)

        Returns:
            Policy type: "student" or "teacher"
        """
        pass

    @beartype
    @abstractmethod
    def reset(self, env_idx: int) -> None:
        """Reset scheduler state for given environment."""
        pass