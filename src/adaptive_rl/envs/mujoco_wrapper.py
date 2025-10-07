"""MuJoCo Gym environment wrapper for adaptive RL experiments.

Provides access to classic MuJoCo environments with consistent interface.
These are the standard Gymnasium MuJoCo environments.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Union, Optional
import warnings


class MuJoCoWrapper(gym.Wrapper):
    """Wrapper for MuJoCo Gym environments compatible with our system.

    Provides consistent interface for standard MuJoCo control tasks.
    """

    def __init__(
        self,
        env_name: str,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """Initialize MuJoCo environment wrapper.

        Args:
            env_name: MuJoCo environment name
            render_mode: Rendering mode ('human', 'rgb_array', etc.)
            **kwargs: Additional arguments for environment
        """
        self.env_name = env_name

        try:
            # Create base environment
            base_env = gym.make(env_name, render_mode=render_mode, **kwargs)
        except Exception as e:
            warnings.warn(f"Failed to create MuJoCo environment {env_name}: {e}")
            raise

        # Add episode statistics tracking
        base_env = gym.wrappers.RecordEpisodeStatistics(base_env)

        super().__init__(base_env)

        # Store configuration
        self.config = {
            "env_name": env_name,
            "action_space_type": "continuous" if isinstance(self.action_space, spaces.Box) else "discrete",
            "observation_space_shape": self.observation_space.shape,
            "action_space_shape": getattr(self.action_space, 'shape', (1,))
        }

    def reset(self, **kwargs) -> Tuple[Union[np.ndarray, Dict], Dict]:
        """Reset environment."""
        obs, info = self.env.reset(**kwargs)
        info.update(self.config)
        return obs, info

    def step(self, action) -> Tuple[Union[np.ndarray, Dict], float, bool, bool, Dict]:
        """Step environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(self.config)
        return obs, reward, terminated, truncated, info

    @property
    def spec(self):
        """Environment spec."""
        return gym.envs.registration.EnvSpec(
            id=f"MuJoCo_{self.env_name}-v0",
            entry_point=None
        )


# Supported MuJoCo environments (Gymnasium versions)
MUJOCO_ENVS = {
    # Classic control
    "inverted_pendulum": "InvertedPendulum-v4",
    "inverted_double_pendulum": "InvertedDoublePendulum-v4",
    "pendulum": "Pendulum-v1",

    # Locomotion - bipeds
    "walker2d": "Walker2d-v4",
    "humanoid": "Humanoid-v4",
    "humanoid_standup": "HumanoidStandup-v4",

    # Locomotion - quadrupeds
    "ant": "Ant-v4",
    "halfcheetah": "HalfCheetah-v4",

    # Locomotion - hoppers
    "hopper": "Hopper-v4",

    # Manipulation
    "reacher": "Reacher-v4",
    "pusher": "Pusher-v4",
    "striker": "Striker-v4",
    "thrower": "Thrower-v4",

    # Swimming
    "swimmer": "Swimmer-v4",

    # Additional classic environments
    "mountain_car_continuous": "MountainCarContinuous-v0",
    "acrobot": "Acrobot-v1",
}


def create_mujoco_env(env_name: str, **kwargs) -> MuJoCoWrapper:
    """Create MuJoCo environment by name.

    Args:
        env_name: Environment name from MUJOCO_ENVS
        **kwargs: Additional arguments for MuJoCoWrapper

    Returns:
        MuJoCoWrapper instance
    """
    if env_name not in MUJOCO_ENVS:
        available = list(MUJOCO_ENVS.keys())
        raise ValueError(f"Unknown MuJoCo environment {env_name}. Available: {available}")

    gym_env_name = MUJOCO_ENVS[env_name]
    return MuJoCoWrapper(gym_env_name, **kwargs)


def get_mujoco_env_info(env_name: str) -> Dict[str, Any]:
    """Get information about a MuJoCo environment."""
    if env_name not in MUJOCO_ENVS:
        raise ValueError(f"Unknown environment: {env_name}")

    # Environment categorization
    env_categories = {
        # Control tasks
        "inverted_pendulum": {"category": "control", "complexity": "low", "description": "Balance inverted pendulum"},
        "inverted_double_pendulum": {"category": "control", "complexity": "medium", "description": "Balance double pendulum"},
        "pendulum": {"category": "control", "complexity": "low", "description": "Swing up pendulum"},
        "mountain_car_continuous": {"category": "control", "complexity": "low", "description": "Continuous mountain car"},
        "acrobot": {"category": "control", "complexity": "medium", "description": "Swing up acrobot"},

        # Locomotion - bipeds
        "walker2d": {"category": "locomotion", "complexity": "medium", "robot": "biped", "description": "2D walking"},
        "humanoid": {"category": "locomotion", "complexity": "high", "robot": "humanoid", "description": "3D humanoid locomotion"},
        "humanoid_standup": {"category": "locomotion", "complexity": "high", "robot": "humanoid", "description": "Humanoid standup"},
        "hopper": {"category": "locomotion", "complexity": "medium", "robot": "hopper", "description": "Single-leg hopping"},

        # Locomotion - quadrupeds
        "ant": {"category": "locomotion", "complexity": "medium", "robot": "quadruped", "description": "4-legged locomotion"},
        "halfcheetah": {"category": "locomotion", "complexity": "medium", "robot": "quadruped", "description": "Fast running"},

        # Manipulation
        "reacher": {"category": "manipulation", "complexity": "low", "robot": "arm", "description": "2D reaching"},
        "pusher": {"category": "manipulation", "complexity": "medium", "robot": "arm", "description": "Push object to target"},
        "striker": {"category": "manipulation", "complexity": "medium", "robot": "arm", "description": "Strike ball to target"},
        "thrower": {"category": "manipulation", "complexity": "high", "robot": "arm", "description": "Throw ball to target"},

        # Swimming
        "swimmer": {"category": "locomotion", "complexity": "medium", "robot": "swimmer", "description": "Swimming locomotion"},
    }

    return env_categories.get(env_name, {
        "category": "unknown",
        "complexity": "medium",
        "robot": "unknown",
        "description": f"MuJoCo environment: {env_name}"
    })


def list_mujoco_envs_by_category() -> Dict[str, list]:
    """List MuJoCo environments grouped by category."""
    categories = {}
    for env_name in MUJOCO_ENVS.keys():
        info = get_mujoco_env_info(env_name)
        category = info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(env_name)
    return categories


def get_mujoco_benchmark_envs() -> Dict[str, list]:
    """Get standard MuJoCo benchmark environments for different categories."""
    return {
        "locomotion": ["halfcheetah", "walker2d", "hopper", "ant"],
        "manipulation": ["reacher", "pusher"],
        "control": ["inverted_pendulum", "pendulum"],
        "complex": ["humanoid", "swimmer"]
    }