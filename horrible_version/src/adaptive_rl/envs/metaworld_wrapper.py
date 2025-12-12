"""MetaWorld environment wrapper for adaptive RL experiments.

Provides consistent interface for MetaWorld manipulation tasks.
Handles observation preprocessing and action space normalization.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Union, Optional
import warnings

try:
    import metaworld
    METAWORLD_AVAILABLE = True
except ImportError:
    METAWORLD_AVAILABLE = False
    warnings.warn("MetaWorld not available. Install with: pip install metaworld")


class MetaWorldWrapper(gym.Env):
    """Wrapper to make MetaWorld environments compatible with our system.

    Handles observation preprocessing and provides consistent interface.
    """

    def __init__(
        self,
        task_name: str,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """Initialize MetaWorld environment wrapper.

        Args:
            task_name: MetaWorld task name (e.g., 'reach-v2')
            seed: Random seed
            render_mode: Rendering mode
        """
        if not METAWORLD_AVAILABLE:
            raise ImportError("MetaWorld not available. Install with: pip install metaworld")

        self.task_name = task_name
        self.seed_value = seed

        # Create MetaWorld environment
        if task_name in metaworld.ML1.ENV_NAMES:
            # ML1 (single task)
            ml1 = metaworld.ML1(task_name, seed=seed)
            self._env = ml1.train_classes[task_name]()
            tasks = ml1.train_tasks
            self._env.set_task(tasks[0])
        else:
            raise ValueError(f"Unknown MetaWorld task: {task_name}")

        # Set render mode
        if render_mode:
            self._env.render_mode = render_mode

        # Create gymnasium spaces
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()

        # Episode tracking
        self._step_count = 0
        self._episode_reward = 0.0
        self._max_episode_steps = getattr(self._env, 'max_path_length', 500)

    def _create_action_space(self) -> spaces.Box:
        """Create gymnasium action space from MetaWorld action space."""
        # MetaWorld uses 4D actions: [x, y, z, gripper]
        # Actions are typically in [-1, 1] range
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

    def _create_observation_space(self) -> spaces.Box:
        """Create gymnasium observation space from MetaWorld observation space."""
        # Get sample observation to determine shape
        sample_obs = self._env.reset()
        obs_shape = sample_obs.shape

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        if seed is not None:
            self.seed_value = seed
            np.random.seed(seed)

        obs = self._env.reset()
        obs = np.array(obs, dtype=np.float32)

        self._step_count = 0
        self._episode_reward = 0.0

        info = {
            'task_name': self.task_name,
            'step_count': self._step_count,
            'max_episode_steps': self._max_episode_steps
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment."""
        # Ensure action is properly formatted
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # Step environment
        obs, reward, done, info = self._env.step(action)
        obs = np.array(obs, dtype=np.float32)

        self._step_count += 1
        self._episode_reward += reward

        # MetaWorld doesn't distinguish terminated vs truncated
        terminated = done
        truncated = self._step_count >= self._max_episode_steps

        info.update({
            'task_name': self.task_name,
            'step_count': self._step_count,
            'episode_reward': self._episode_reward,
            'success': info.get('success', False)
        })

        return obs, float(reward), terminated, truncated, info

    def render(self, mode='rgb_array', **kwargs):
        """Render environment."""
        return self._env.render(mode=mode, **kwargs)

    def close(self):
        """Close environment."""
        if hasattr(self._env, 'close'):
            self._env.close()

    @property
    def spec(self):
        """Environment spec."""
        return gym.envs.registration.EnvSpec(
            id=f"MetaWorld_{self.task_name}-v0",
            entry_point=None,
            max_episode_steps=self._max_episode_steps
        )


# Supported MetaWorld tasks (subset of ML1 for key manipulation skills)
METAWORLD_TASKS = {
    # Basic manipulation
    "reach": "reach-v2",
    "push": "push-v2",
    "pick_place": "pick-place-v2",

    # Object interaction
    "door_open": "door-open-v2",
    "drawer_open": "drawer-open-v2",
    "drawer_close": "drawer-close-v2",

    # Precision tasks
    "button_press": "button-press-v2",
    "button_press_topdown": "button-press-topdown-v2",

    # Complex manipulation
    "assembly": "assembly-v2",
    "peg_insert_side": "peg-insert-side-v2",

    # Tool use
    "hammer": "hammer-v2",
    "lever_pull": "lever-pull-v2",

    # Diverse skills
    "window_open": "window-open-v2",
    "window_close": "window-close-v2",
    "sweep_into": "sweep-into-v2",
}


def create_metaworld_env(task_name: str, **kwargs) -> MetaWorldWrapper:
    """Create MetaWorld environment by name.

    Args:
        task_name: Task name from METAWORLD_TASKS
        **kwargs: Additional arguments for MetaWorldWrapper

    Returns:
        MetaWorldWrapper instance
    """
    if not METAWORLD_AVAILABLE:
        raise ImportError("MetaWorld not available. Install with: pip install metaworld")

    if task_name not in METAWORLD_TASKS:
        available = list(METAWORLD_TASKS.keys())
        raise ValueError(f"Unknown MetaWorld task {task_name}. Available: {available}")

    metaworld_task_name = METAWORLD_TASKS[task_name]
    return MetaWorldWrapper(metaworld_task_name, **kwargs)


def get_metaworld_task_info(task_name: str) -> Dict[str, Any]:
    """Get information about a MetaWorld task."""
    if not METAWORLD_AVAILABLE:
        raise ImportError("MetaWorld not available")

    if task_name not in METAWORLD_TASKS:
        raise ValueError(f"Unknown task: {task_name}")

    # Basic task categorization
    task_categories = {
        # Basic skills
        "reach": {"category": "basic", "complexity": "low", "description": "Reach to target position"},
        "push": {"category": "contact", "complexity": "low", "description": "Push object to target"},
        "pick_place": {"category": "grasping", "complexity": "medium", "description": "Pick and place object"},

        # Object interaction
        "door_open": {"category": "interaction", "complexity": "medium", "description": "Open door"},
        "drawer_open": {"category": "interaction", "complexity": "medium", "description": "Open drawer"},
        "button_press": {"category": "precision", "complexity": "low", "description": "Press button"},

        # Complex tasks
        "assembly": {"category": "complex", "complexity": "high", "description": "Assemble objects"},
        "hammer": {"category": "tool_use", "complexity": "high", "description": "Use hammer"},
    }

    return task_categories.get(task_name, {
        "category": "unknown",
        "complexity": "medium",
        "description": f"MetaWorld task: {task_name}"
    })