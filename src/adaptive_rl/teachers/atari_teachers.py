"""Teacher policies for Atari environments.

Implements rule-based and heuristic teachers for various Atari games.
These provide reasonable baselines for teacher-student scheduling experiments.
"""

import numpy as np
import torch
from typing import Union, Dict, Any, List
import random

from .base import TeacherPolicy


class BreakoutTeacher(TeacherPolicy):
    """Rule-based teacher for Breakout.

    Strategy: Track ball position and move paddle accordingly.
    For visual observations, uses simple heuristics.
    """

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.action_meanings = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}
        self.prev_ball_x = None
        self.paddle_x = None

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for Breakout.

        For visual observations: Use simple heuristics based on screen analysis.
        For RAM observations: Could use ball/paddle positions directly.
        """
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()

        # Handle batch dimension
        if len(observations.shape) == 1:
            batch_size = 1
            obs = observations.reshape(1, -1)
        else:
            batch_size = observations.shape[0]
            obs = observations

        actions = []

        for i in range(batch_size):
            obs_i = obs[i]

            # Simple heuristic teacher for Breakout
            if len(obs_i.shape) > 1:  # Visual observations
                action = self._visual_breakout_policy(obs_i)
            else:  # RAM observations
                action = self._ram_breakout_policy(obs_i)

            actions.append(action)

        result = np.array(actions)

        if batch_size == 1:
            result = result[0]

        return result

    def _visual_breakout_policy(self, obs: np.ndarray) -> int:
        """Simple visual policy for Breakout."""
        # Very basic heuristic - mostly moves right with some randomness
        # In practice, could analyze pixel positions for ball/paddle
        rand = random.random()
        if rand < 0.4:
            return 2  # RIGHT
        elif rand < 0.7:
            return 3  # LEFT
        elif rand < 0.8:
            return 1  # FIRE
        else:
            return 0  # NOOP

    def _ram_breakout_policy(self, obs: np.ndarray) -> int:
        """RAM-based policy for Breakout (could be more sophisticated)."""
        # With RAM observations, we could track exact ball/paddle positions
        # For now, use similar heuristic
        return self._visual_breakout_policy(obs)


class PongTeacher(TeacherPolicy):
    """Rule-based teacher for Pong.

    Strategy: Track ball movement and position paddle accordingly.
    """

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.action_meanings = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"}

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for Pong."""
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()

        # Handle batch dimension
        if len(observations.shape) == 1:
            batch_size = 1
            obs = observations.reshape(1, -1)
        else:
            batch_size = observations.shape[0]
            obs = observations

        actions = []

        for i in range(batch_size):
            # Simple oscillating strategy - moves paddle up and down
            # Could be improved with ball tracking
            action = self._pong_policy()
            actions.append(action)

        result = np.array(actions)

        if batch_size == 1:
            result = result[0]

        return result

    def _pong_policy(self) -> int:
        """Simple Pong policy."""
        # Oscillating movement with some randomness
        rand = random.random()
        if rand < 0.45:
            return 2  # UP
        elif rand < 0.9:
            return 3  # DOWN
        else:
            return 0  # NOOP


class PacmanTeacher(TeacherPolicy):
    """Rule-based teacher for Ms. Pacman.

    Strategy: Avoid ghosts, collect dots, go for power pellets.
    """

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.action_meanings = {
            0: "NOOP", 1: "UP", 2: "RIGHT", 3: "LEFT", 4: "DOWN",
            5: "UPRIGHT", 6: "UPLEFT", 7: "DOWNRIGHT", 8: "DOWNLEFT"
        }

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for Ms. Pacman."""
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()

        # Handle batch dimension
        if len(observations.shape) == 1:
            batch_size = 1
            obs = observations.reshape(1, -1)
        else:
            batch_size = observations.shape[0]
            obs = observations

        actions = []

        for i in range(batch_size):
            # Random but biased toward movement
            action = self._pacman_policy()
            actions.append(action)

        result = np.array(actions)

        if batch_size == 1:
            result = result[0]

        return result

    def _pacman_policy(self) -> int:
        """Simple Pacman policy."""
        # Prefer movement actions over NOOP
        actions = [1, 2, 3, 4]  # UP, RIGHT, LEFT, DOWN
        weights = [0.25, 0.25, 0.25, 0.25]
        return np.random.choice(actions, p=weights)


class SpaceInvadersTeacher(TeacherPolicy):
    """Rule-based teacher for Space Invaders.

    Strategy: Move to avoid shots, fire frequently.
    """

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.action_meanings = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"}

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for Space Invaders."""
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()

        # Handle batch dimension
        if len(observations.shape) == 1:
            batch_size = 1
            obs = observations.reshape(1, -1)
        else:
            batch_size = observations.shape[0]
            obs = observations

        actions = []

        for i in range(batch_size):
            action = self._space_invaders_policy()
            actions.append(action)

        result = np.array(actions)

        if batch_size == 1:
            result = result[0]

        return result

    def _space_invaders_policy(self) -> int:
        """Simple Space Invaders policy."""
        # Frequently fire while moving
        rand = random.random()
        if rand < 0.3:
            return 1  # FIRE
        elif rand < 0.5:
            return 4  # RIGHTFIRE
        elif rand < 0.7:
            return 5  # LEFTFIRE
        elif rand < 0.85:
            return 2  # RIGHT
        else:
            return 3  # LEFT


class RandomAtariTeacher(TeacherPolicy):
    """Random baseline teacher for any Atari game."""

    def __init__(self, action_space=None, observation_space=None, device="cpu", num_actions=4):
        super().__init__(action_space, observation_space, device)
        self.num_actions = num_actions

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate random actions."""
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()

        # Handle batch dimension
        if len(observations.shape) == 1:
            batch_size = 1
        else:
            batch_size = observations.shape[0]

        actions = np.random.randint(0, self.num_actions, size=batch_size)

        if batch_size == 1:
            actions = actions[0]

        return actions


# Registry of Atari teachers
ATARI_TEACHERS = {
    "breakout": BreakoutTeacher,
    "pong": PongTeacher,
    "pacman": PacmanTeacher,
    "space_invaders": SpaceInvadersTeacher,
    "asterix": RandomAtariTeacher,
    "beam_rider": RandomAtariTeacher,
    "enduro": RandomAtariTeacher,
    "qbert": RandomAtariTeacher,
    "freeway": RandomAtariTeacher,
    "seaquest": RandomAtariTeacher,
}


def create_atari_teacher(env_name: str, **kwargs) -> TeacherPolicy:
    """Create teacher for Atari environment.

    Args:
        env_name: Environment name (e.g., 'breakout')
        **kwargs: Additional arguments for teacher

    Returns:
        TeacherPolicy instance
    """
    if env_name not in ATARI_TEACHERS:
        available = list(ATARI_TEACHERS.keys())
        raise ValueError(f"No teacher available for {env_name}. Available: {available}")

    teacher_class = ATARI_TEACHERS[env_name]
    return teacher_class(**kwargs)