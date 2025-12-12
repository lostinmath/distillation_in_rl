"""Environment registry with factory pattern."""

from dataclasses import dataclass
from typing import Callable, Any
from beartype import beartype
import gymnasium as gym


@dataclass
class EnvSpec:
    """Environment specification."""
    env_id: str
    env_type: str  # "discrete", "continuous"
    teacher_factory: Callable[[], Any]
    description: str
    max_episode_steps: int = 500


class EnvironmentRegistry:
    """Registry for supported environments and their teachers."""

    def __init__(self):
        self._registry: dict[str, EnvSpec] = {}

    @beartype
    def register(self, name: str, spec: EnvSpec) -> None:
        """Register an environment."""
        self._registry[name] = spec

    @beartype
    def create_env(self, name: str) -> tuple[gym.Env, Any]:
        """Create environment and teacher policy."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown environment: {name}. Available: {available}")

        spec = self._registry[name]
        env = gym.make(spec.env_id, max_episode_steps=spec.max_episode_steps)
        teacher = spec.teacher_factory()

        return env, teacher

    @beartype
    def list_environments(self) -> list[str]:
        """List all registered environments."""
        return list(self._registry.keys())

    @beartype
    def get_spec(self, name: str) -> EnvSpec:
        """Get environment specification."""
        if name not in self._registry:
            raise ValueError(f"Unknown environment: {name}")
        return self._registry[name]


# Global registry instance
registry = EnvironmentRegistry()


def register_env(name: str, spec: EnvSpec) -> None:
    """Register environment in global registry."""
    registry.register(name, spec)