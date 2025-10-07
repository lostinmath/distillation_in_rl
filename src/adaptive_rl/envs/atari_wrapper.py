"""Atari environment wrapper for adaptive RL experiments.

Provides consistent interface for Atari games with both visual and RAM observations.
Includes frame stacking, preprocessing, and action space handling.
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStack,
    RecordEpisodeStatistics
)
from typing import Union, Tuple, Dict, Any


class AtariWrapper(gym.Wrapper):
    """Wrapper for Atari environments compatible with our system.

    Handles both visual (84x84 grayscale) and RAM (128D vector) observations.
    Supports frame stacking for temporal information.
    """

    def __init__(
        self,
        env_name: str,
        obs_type: str = "rgb",  # "rgb" or "ram"
        frame_stack: int = 4,
        noop_max: int = 30,
        frameskip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        grayscale_obs: bool = True,
        scale_obs: bool = True
    ):
        """Initialize Atari environment wrapper.

        Args:
            env_name: Atari environment name (e.g., 'BreakoutNoFrameskip-v4')
            obs_type: 'rgb' for visual observations, 'ram' for RAM state
            frame_stack: Number of frames to stack for temporal info
            noop_max: Maximum no-op actions at episode start
            frameskip: Number of frames to skip between actions
            screen_size: Size of preprocessed frames (screen_size x screen_size)
            terminal_on_life_loss: Whether to terminate on life loss
            grayscale_obs: Whether to convert to grayscale
            scale_obs: Whether to scale observations to [0,1]
        """
        self.env_name = env_name
        self.obs_type = obs_type
        self.frame_stack = frame_stack

        # Create base environment
        if obs_type == "ram":
            # Use RAM observations (128D vector)
            base_env_name = env_name.replace("NoFrameskip", "")
            if not base_env_name.endswith("-ram"):
                base_env_name = base_env_name.replace("-v4", "-ram-v4")
            env = gym.make(base_env_name)
        else:
            # Use visual observations
            env = gym.make(env_name)

        # Add episode statistics tracking
        env = RecordEpisodeStatistics(env)

        if obs_type == "rgb":
            # Apply Atari preprocessing for visual observations
            env = AtariPreprocessing(
                env,
                noop_max=noop_max,
                frame_skip=frameskip,
                screen_size=screen_size,
                terminal_on_life_loss=terminal_on_life_loss,
                grayscale_obs=grayscale_obs,
                scale_obs=scale_obs
            )

            # Add frame stacking for temporal information
            if frame_stack > 1:
                env = FrameStack(env, num_stack=frame_stack)

        super().__init__(env)

        # Store configuration
        self.config = {
            "env_name": env_name,
            "obs_type": obs_type,
            "frame_stack": frame_stack,
            "screen_size": screen_size if obs_type == "rgb" else None,
            "action_space_size": self.action_space.n
        }

    def reset(self, **kwargs) -> Tuple[Union[np.ndarray, Dict], Dict]:
        """Reset environment."""
        obs, info = self.env.reset(**kwargs)
        info.update(self.config)
        return obs, info

    def step(self, action: int) -> Tuple[Union[np.ndarray, Dict], float, bool, bool, Dict]:
        """Step environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(self.config)
        return obs, reward, terminated, truncated, info

    @property
    def spec(self):
        """Environment spec."""
        return gym.envs.registration.EnvSpec(
            id=f"Atari_{self.env_name}_{self.obs_type}-v0",
            entry_point=None
        )


# Supported Atari environments
ATARI_ENVS = {
    # Classic games - good for teacher demonstration
    "breakout": "BreakoutNoFrameskip-v4",
    "pong": "PongNoFrameskip-v4",
    "pacman": "MsPacmanNoFrameskip-v4",
    "space_invaders": "SpaceInvadersNoFrameskip-v4",

    # More complex games
    "asterix": "AsterixNoFrameskip-v4",
    "beam_rider": "BeamRiderNoFrameskip-v4",
    "enduro": "EnduroNoFrameskip-v4",
    "qbert": "QbertNoFrameskip-v4",

    # Exploration-heavy games
    "freeway": "FreewayNoFrameskip-v4",
    "seaquest": "SeaquestNoFrameskip-v4",
}


def create_atari_env(
    env_name: str,
    obs_type: str = "rgb",
    **kwargs
) -> AtariWrapper:
    """Create Atari environment by name.

    Args:
        env_name: Environment name from ATARI_ENVS
        obs_type: 'rgb' for visual observations, 'ram' for RAM state
        **kwargs: Additional arguments for AtariWrapper

    Returns:
        AtariWrapper instance
    """
    if env_name not in ATARI_ENVS:
        available = list(ATARI_ENVS.keys())
        raise ValueError(f"Unknown Atari environment {env_name}. Available: {available}")

    gym_env_name = ATARI_ENVS[env_name]
    return AtariWrapper(gym_env_name, obs_type=obs_type, **kwargs)


def get_atari_action_meanings(env_name: str) -> Dict[int, str]:
    """Get action meanings for Atari environment.

    Useful for implementing rule-based teachers.
    """
    if env_name not in ATARI_ENVS:
        raise ValueError(f"Unknown environment: {env_name}")

    gym_env_name = ATARI_ENVS[env_name]
    temp_env = gym.make(gym_env_name)
    meanings = temp_env.get_action_meanings()
    temp_env.close()

    return {i: meaning for i, meaning in enumerate(meanings)}