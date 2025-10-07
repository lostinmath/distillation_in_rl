"""dm_control environment wrapper for adaptive RL experiments."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite
from typing import Dict, Any, Tuple, Union


class DMControlWrapper(gym.Env):
    """Wrapper to make dm_control environments compatible with our system.

    Converts dict observations to flat vectors and handles continuous actions
    for compatibility with our PPO implementation.
    """

    def __init__(self, domain_name: str, task_name: str, flatten_obs: bool = True):
        """Initialize dm_control environment wrapper.

        Args:
            domain_name: dm_control domain (e.g., 'cheetah')
            task_name: dm_control task (e.g., 'run')
            flatten_obs: Whether to flatten dict observations to vector
        """
        self.domain_name = domain_name
        self.task_name = task_name
        self.flatten_obs = flatten_obs

        # Create dm_control environment
        self._env = suite.load(domain_name=domain_name, task_name=task_name)

        # Get specs
        self._action_spec = self._env.action_spec()
        self._obs_spec = self._env.observation_spec()

        # Create gymnasium spaces
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()

        # State tracking
        self._step_count = 0
        self._episode_reward = 0.0

    def _create_action_space(self) -> spaces.Box:
        """Create gymnasium action space from dm_control action spec."""
        return spaces.Box(
            low=float(self._action_spec.minimum.min()),
            high=float(self._action_spec.maximum.max()),
            shape=self._action_spec.shape,
            dtype=np.float32
        )

    def _create_observation_space(self) -> Union[spaces.Box, spaces.Dict]:
        """Create gymnasium observation space from dm_control observation spec."""
        if self.flatten_obs:
            # Calculate total flattened size
            total_size = 0
            for key, spec in self._obs_spec.items():
                total_size += int(np.prod(spec.shape))

            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_size,),
                dtype=np.float32
            )
        else:
            # Keep dict structure
            obs_spaces = {}
            for key, spec in self._obs_spec.items():
                obs_spaces[key] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=spec.shape,
                    dtype=np.float32
                )
            return spaces.Dict(obs_spaces)

    def _flatten_observation(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten dict observation to vector."""
        obs_parts = []
        for key in sorted(obs_dict.keys()):  # Ensure consistent ordering
            obs_parts.append(obs_dict[key].flatten())
        return np.concatenate(obs_parts).astype(np.float32)

    def reset(self, seed=None, options=None) -> Tuple[Union[np.ndarray, Dict], Dict]:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        time_step = self._env.reset()
        obs = self._process_observation(time_step.observation)

        self._step_count = 0
        self._episode_reward = 0.0

        info = {
            'discount': time_step.discount,
            'step_count': self._step_count
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Union[np.ndarray, Dict], float, bool, bool, Dict]:
        """Step environment."""
        # Convert action to proper format
        action = np.asarray(action, dtype=np.float32)

        # Clip to action bounds
        action = np.clip(action, self._action_spec.minimum, self._action_spec.maximum)

        # Step environment
        time_step = self._env.step(action)

        # Process observation
        obs = self._process_observation(time_step.observation)

        # Get reward
        reward = float(time_step.reward or 0.0)
        self._episode_reward += reward

        # Check termination
        terminated = time_step.last()
        truncated = False  # dm_control handles episode length internally

        self._step_count += 1

        info = {
            'discount': time_step.discount,
            'step_count': self._step_count,
            'episode_reward': self._episode_reward
        }

        return obs, reward, terminated, truncated, info

    def _process_observation(self, obs_dict: Dict[str, np.ndarray]) -> Union[np.ndarray, Dict]:
        """Process observation based on flatten_obs setting."""
        if self.flatten_obs:
            return self._flatten_observation(obs_dict)
        else:
            # Convert to float32 for consistency
            return {key: value.astype(np.float32) for key, value in obs_dict.items()}

    def render(self, mode='rgb_array', **kwargs):
        """Render environment."""
        return self._env.physics.render(**kwargs)

    def close(self):
        """Close environment."""
        self._env.close()

    @property
    def spec(self):
        """Environment spec."""
        return gym.envs.registration.EnvSpec(
            id=f"DMControl_{self.domain_name}_{self.task_name}-v0",
            entry_point=None
        )


def make_dm_control_env(domain_name: str, task_name: str, **kwargs) -> DMControlWrapper:
    """Factory function to create dm_control environment."""
    return DMControlWrapper(domain_name, task_name, **kwargs)


# Common environment configurations
DM_CONTROL_ENVS = {
    "cheetah_run": ("cheetah", "run"),
    "walker_walk": ("walker", "walk"),
    "walker_stand": ("walker", "stand"),
    "reacher_easy": ("reacher", "easy"),
    "reacher_hard": ("reacher", "hard"),
    "cartpole_balance": ("cartpole", "balance"),
    "cartpole_swingup": ("cartpole", "swingup"),
    "ball_in_cup_catch": ("ball_in_cup", "catch"),
}


def create_dm_control_env(env_name: str, **kwargs) -> DMControlWrapper:
    """Create dm_control environment by name.

    Args:
        env_name: Environment name from DM_CONTROL_ENVS
        **kwargs: Additional arguments for DMControlWrapper

    Returns:
        DMControlWrapper instance
    """
    if env_name not in DM_CONTROL_ENVS:
        available = list(DM_CONTROL_ENVS.keys())
        raise ValueError(f"Unknown environment {env_name}. Available: {available}")

    domain_name, task_name = DM_CONTROL_ENVS[env_name]
    return make_dm_control_env(domain_name, task_name, **kwargs)