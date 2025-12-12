"""PyBullet environment wrapper for adaptive RL experiments.

Provides consistent interface for PyBullet robotics environments.
Supports both manipulation and locomotion tasks.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Union, Optional
import warnings

try:
    import pybullet_envs
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    warnings.warn("PyBullet not available. Install with: pip install pybullet")


class PyBulletWrapper(gym.Wrapper):
    """Wrapper for PyBullet environments compatible with our system.

    Provides consistent interface and observation preprocessing.
    """

    def __init__(
        self,
        env_name: str,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """Initialize PyBullet environment wrapper.

        Args:
            env_name: PyBullet environment name
            render_mode: Rendering mode ('human', 'rgb_array', etc.)
            **kwargs: Additional arguments for environment
        """
        if not PYBULLET_AVAILABLE:
            raise ImportError("PyBullet not available. Install with: pip install pybullet")

        self.env_name = env_name

        # Create base environment
        base_env = gym.make(env_name, render_mode=render_mode, **kwargs)

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
            id=f"PyBullet_{self.env_name}-v0",
            entry_point=None
        )


# Supported PyBullet environments
PYBULLET_ENVS = {
    # Manipulation tasks
    "kuka_reach": "KukaCamReach-v0",
    "kuka_grasp": "KukaCamGrasp-v0",
    "kuka_diverse": "KukaDiverseObjectEnv-v0",

    # Locomotion tasks - quadrupeds
    "ant": "AntBulletEnv-v0",
    "halfcheetah": "HalfCheetahBulletEnv-v0",
    "walker2d": "Walker2DBulletEnv-v0",

    # Locomotion tasks - humanoids
    "humanoid": "HumanoidBulletEnv-v0",
    "humanoid_flagrun": "HumanoidFlagrunBulletEnv-v0",
    "humanoid_flagrun_harder": "HumanoidFlagrunHarderBulletEnv-v0",

    # Manipulation - reaching
    "reacher": "ReacherBulletEnv-v0",
    "pusher": "PusherBulletEnv-v0",
    "striker": "StrikerBulletEnv-v0",
    "thrower": "ThrowerBulletEnv-v0",

    # Locomotion - hoppers
    "hopper": "HopperBulletEnv-v0",
    "inverted_pendulum": "InvertedPendulumBulletEnv-v0",
    "inverted_double_pendulum": "InvertedDoublePendulumBulletEnv-v0",

    # Racing/driving
    "racecar": "RacecarBulletEnv-v0",
    "racecar_zed": "RacecarZedBulletEnv-v0",

    # Flying
    "minitaur": "MinitaurBulletEnv-v0",
    "minitaur_duck": "MinitaurBulletDuckEnv-v0",
}


def create_pybullet_env(env_name: str, **kwargs) -> PyBulletWrapper:
    """Create PyBullet environment by name.

    Args:
        env_name: Environment name from PYBULLET_ENVS
        **kwargs: Additional arguments for PyBulletWrapper

    Returns:
        PyBulletWrapper instance
    """
    if not PYBULLET_AVAILABLE:
        raise ImportError("PyBullet not available. Install with: pip install pybullet")

    if env_name not in PYBULLET_ENVS:
        available = list(PYBULLET_ENVS.keys())
        raise ValueError(f"Unknown PyBullet environment {env_name}. Available: {available}")

    gym_env_name = PYBULLET_ENVS[env_name]
    return PyBulletWrapper(gym_env_name, **kwargs)


def get_pybullet_env_info(env_name: str) -> Dict[str, Any]:
    """Get information about a PyBullet environment."""
    if not PYBULLET_AVAILABLE:
        raise ImportError("PyBullet not available")

    if env_name not in PYBULLET_ENVS:
        raise ValueError(f"Unknown environment: {env_name}")

    # Environment categorization
    env_categories = {
        # Manipulation
        "kuka_reach": {"category": "manipulation", "complexity": "medium", "robot": "kuka", "description": "Reach to target"},
        "kuka_grasp": {"category": "manipulation", "complexity": "high", "robot": "kuka", "description": "Grasp objects"},
        "reacher": {"category": "manipulation", "complexity": "low", "robot": "reacher", "description": "Point reaching"},
        "pusher": {"category": "manipulation", "complexity": "medium", "robot": "pusher", "description": "Push object"},

        # Locomotion - quadrupeds
        "ant": {"category": "locomotion", "complexity": "medium", "robot": "quadruped", "description": "4-legged walking"},
        "halfcheetah": {"category": "locomotion", "complexity": "medium", "robot": "quadruped", "description": "Fast running"},
        "walker2d": {"category": "locomotion", "complexity": "medium", "robot": "biped", "description": "2D walking"},

        # Locomotion - humanoids
        "humanoid": {"category": "locomotion", "complexity": "high", "robot": "humanoid", "description": "3D humanoid walking"},
        "hopper": {"category": "locomotion", "complexity": "low", "robot": "hopper", "description": "Single-leg hopping"},

        # Control
        "inverted_pendulum": {"category": "control", "complexity": "low", "robot": "pendulum", "description": "Balance pendulum"},
        "inverted_double_pendulum": {"category": "control", "complexity": "medium", "robot": "pendulum", "description": "Double pendulum"},

        # Specialized
        "racecar": {"category": "driving", "complexity": "medium", "robot": "vehicle", "description": "Car racing"},
        "minitaur": {"category": "locomotion", "complexity": "high", "robot": "quadruped", "description": "Minitaur robot"},
    }

    return env_categories.get(env_name, {
        "category": "unknown",
        "complexity": "medium",
        "robot": "unknown",
        "description": f"PyBullet environment: {env_name}"
    })


def list_pybullet_envs_by_category() -> Dict[str, list]:
    """List PyBullet environments grouped by category."""
    categories = {}
    for env_name in PYBULLET_ENVS.keys():
        info = get_pybullet_env_info(env_name)
        category = info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(env_name)
    return categories