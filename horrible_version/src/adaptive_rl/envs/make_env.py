"""Environment factory for creating Gymnasium environments.

Supports both single and vectorized environments with
optional video recording and monitoring.
"""

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np


def make_env(
    env_id: str,
    seed: int = 0,
    idx: int = 0,
    capture_video: bool = False,
    run_name: str = "",
    video_dir: str = "videos",
) -> Callable:
    """Create a single environment wrapped with monitoring.

    Args:
        env_id: Gymnasium environment ID
        seed: Random seed
        idx: Environment index (for video recording)
        capture_video: Whether to record videos
        run_name: Name for the run (used in video path)
        video_dir: Directory for videos

    Returns:
        A thunk (callable) that creates the environment
    """

    def thunk():
        # Create base environment
        env = gym.make(env_id, render_mode="rgb_array" if capture_video else None)

        # Add episode statistics wrapper
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Add video recording for first environment
        if capture_video and idx == 0:
            video_path = Path(video_dir) / run_name
            video_path.mkdir(parents=True, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                str(video_path),
                episode_trigger=lambda e: e % 100 == 0,  # Record every 100 episodes
                name_prefix=f"env_{idx}",
            )

        # Set seeds
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk


def make_vec_env(
    env_id: str,
    num_envs: int,
    seed: int = 0,
    capture_video: bool = False,
    run_name: str = "",
    video_dir: str = "videos",
    vec_env_cls: str = "sync",
) -> gym.vector.VectorEnv:
    """Create vectorized environments.

    Args:
        env_id: Gymnasium environment ID
        num_envs: Number of parallel environments
        seed: Base random seed
        capture_video: Whether to record videos
        run_name: Name for the run
        video_dir: Directory for videos
        vec_env_cls: Type of vectorization ("sync" or "async")

    Returns:
        Vectorized environment
    """
    # Create environment thunks
    env_fns = [
        make_env(env_id, seed + i, i, capture_video, run_name, video_dir)
        for i in range(num_envs)
    ]

    # Create vectorized environment
    if vec_env_cls == "async":
        envs = gym.vector.AsyncVectorEnv(env_fns)
    else:
        envs = gym.vector.SyncVectorEnv(env_fns)

    return envs


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to [-1, 1] range.

    Useful for environments with unbounded observations.
    """

    def __init__(self, env, epsilon=1e-8):
        """Initialize normalizer.

        Args:
            env: Environment to wrap
            epsilon: Small value to avoid division by zero
        """
        super().__init__(env)
        self.epsilon = epsilon
        self.obs_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.obs_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.count = 0

    def observation(self, obs):
        """Normalize observation.

        Args:
            obs: Raw observation

        Returns:
            Normalized observation
        """
        # Update running statistics
        self.count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.count
        self.obs_var += delta * (obs - self.obs_mean)

        # Compute standard deviation
        std = np.sqrt(self.obs_var / max(1, self.count - 1))
        std = np.maximum(std, self.epsilon)

        # Normalize
        return (obs - self.obs_mean) / std


class RewardScaler(gym.RewardWrapper):
    """Scale rewards by a constant factor.

    Useful for adjusting reward magnitudes.
    """

    def __init__(self, env, scale=1.0):
        """Initialize reward scaler.

        Args:
            env: Environment to wrap
            scale: Scaling factor for rewards
        """
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        """Scale reward.

        Args:
            reward: Raw reward

        Returns:
            Scaled reward
        """
        return reward * self.scale


def create_env_with_wrappers(
    env_id: str,
    seed: int = 0,
    normalize_obs: bool = False,
    reward_scale: float = 1.0,
    capture_video: bool = False,
    run_name: str = "",
    video_dir: str = "videos",
) -> gym.Env:
    """Create a single environment with optional wrappers.

    Args:
        env_id: Gymnasium environment ID
        seed: Random seed
        normalize_obs: Whether to normalize observations
        reward_scale: Reward scaling factor
        capture_video: Whether to record videos
        run_name: Name for the run
        video_dir: Directory for videos

    Returns:
        Wrapped environment
    """
    # Create base environment
    env = gym.make(env_id, render_mode="rgb_array" if capture_video else None)

    # Add wrappers
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if normalize_obs:
        env = NormalizeObservation(env)

    if reward_scale != 1.0:
        env = RewardScaler(env, reward_scale)

    if capture_video:
        video_path = Path(video_dir) / run_name
        video_path.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, str(video_path), episode_trigger=lambda e: e % 100 == 0
        )

    # Set seeds
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
