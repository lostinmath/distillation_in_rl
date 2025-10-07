"""Teacher policies for MetaWorld manipulation environments.

Implements scripted and heuristic teachers for various MetaWorld tasks.
These provide structured guidance for manipulation learning.
"""

import numpy as np
import torch
from typing import Union, Dict, Any, Tuple
import warnings

from .base import TeacherPolicy


class ReachTeacher(TeacherPolicy):
    """Teacher for reach task - move end effector to target position."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.target_threshold = 0.05
        self.action_scale = 1.0

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for reach task.

        MetaWorld observation typically includes:
        - End effector position (3D)
        - Target position (3D)
        - Gripper state
        - Object positions
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
            action = self._reach_policy(obs_i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        return result

    def _reach_policy(self, obs: np.ndarray) -> np.ndarray:
        """Simple reach policy: move toward target."""
        # Extract end effector and target positions
        # This assumes standard MetaWorld observation format
        if len(obs) >= 6:
            ee_pos = obs[:3]  # End effector position
            target_pos = obs[3:6]  # Target position
        else:
            # Fallback for different observation formats
            ee_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
            target_pos = np.array([0.5, 0.0, 0.2])  # Default target

        # Calculate direction to target
        direction = target_pos - ee_pos
        distance = np.linalg.norm(direction)

        if distance > self.target_threshold:
            # Move toward target
            direction_normalized = direction / (distance + 1e-8)
            action_xyz = direction_normalized * self.action_scale
        else:
            # Close to target, small movements
            action_xyz = direction * 0.1

        # Clip to action bounds
        action_xyz = np.clip(action_xyz, -1.0, 1.0)

        # Add gripper action (keep open for reach task)
        action_gripper = np.array([0.0])  # Open gripper

        return np.concatenate([action_xyz, action_gripper])


class PickPlaceTeacher(TeacherPolicy):
    """Teacher for pick-and-place task with staged approach."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.grasp_threshold = 0.03
        self.lift_height = 0.1
        self.stages = ["approach", "grasp", "lift", "move", "place"]
        self.current_stage = "approach"

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for pick-place task with state machine."""
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
            action = self._pick_place_policy(obs_i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        return result

    def _pick_place_policy(self, obs: np.ndarray) -> np.ndarray:
        """Staged pick-and-place policy."""
        # Extract positions (assuming standard MetaWorld format)
        if len(obs) >= 9:
            ee_pos = obs[:3]
            obj_pos = obs[3:6]
            target_pos = obs[6:9]
        else:
            # Fallback
            ee_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
            obj_pos = obs[3:6] if len(obs) >= 6 else np.array([0.0, 0.0, 0.0])
            target_pos = obs[6:9] if len(obs) >= 9 else np.array([0.2, 0.2, 0.1])

        # Simple state machine
        ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        obj_height = obj_pos[2] if len(obj_pos) > 2 else 0.0

        if ee_to_obj > self.grasp_threshold:
            # Approach object
            direction = obj_pos - ee_pos
            action_xyz = direction * 2.0  # Move toward object
            action_gripper = 1.0  # Open gripper
        elif obj_height < 0.05:
            # Grasp object
            action_xyz = np.array([0.0, 0.0, -0.1])  # Move down slightly
            action_gripper = -1.0  # Close gripper
        else:
            # Move to target
            direction = target_pos - ee_pos
            action_xyz = direction * 1.5
            action_gripper = -1.0 if np.linalg.norm(direction) > 0.1 else 1.0

        # Clip actions
        action_xyz = np.clip(action_xyz, -1.0, 1.0)
        action_gripper = np.clip(action_gripper, -1.0, 1.0)

        return np.concatenate([action_xyz, [action_gripper]])


class DoorOpenTeacher(TeacherPolicy):
    """Teacher for door opening task."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for door opening."""
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
            action = self._door_open_policy(obs_i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        return result

    def _door_open_policy(self, obs: np.ndarray) -> np.ndarray:
        """Simple door opening policy."""
        # Move toward door handle and pull
        action_xyz = np.array([0.2, -0.3, 0.0])  # Pull motion
        action_gripper = -1.0  # Closed gripper

        action_xyz = np.clip(action_xyz, -1.0, 1.0)
        return np.concatenate([action_xyz, [action_gripper]])


class ButtonPressTeacher(TeacherPolicy):
    """Teacher for button pressing task."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for button pressing."""
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
            action = self._button_press_policy(obs_i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        return result

    def _button_press_policy(self, obs: np.ndarray) -> np.ndarray:
        """Simple button pressing policy."""
        # Move toward button and press down
        action_xyz = np.array([0.0, 0.0, -0.5])  # Downward motion
        action_gripper = 0.0  # Neutral gripper

        action_xyz = np.clip(action_xyz, -1.0, 1.0)
        return np.concatenate([action_xyz, [action_gripper]])


class RandomMetaWorldTeacher(TeacherPolicy):
    """Random baseline teacher for MetaWorld tasks."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate random actions for MetaWorld."""
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()

        # Handle batch dimension
        if len(observations.shape) == 1:
            batch_size = 1
        else:
            batch_size = observations.shape[0]

        # Random actions in [-1, 1] for 4D action space
        actions = np.random.uniform(-1.0, 1.0, size=(batch_size, 4))

        if batch_size == 1:
            actions = actions[0]

        return actions.astype(np.float32)


# Registry of MetaWorld teachers
METAWORLD_TEACHERS = {
    "reach": ReachTeacher,
    "push": ReachTeacher,  # Similar to reach
    "pick_place": PickPlaceTeacher,
    "door_open": DoorOpenTeacher,
    "drawer_open": DoorOpenTeacher,  # Similar pulling motion
    "drawer_close": DoorOpenTeacher,  # Similar pushing motion
    "button_press": ButtonPressTeacher,
    "button_press_topdown": ButtonPressTeacher,
    "assembly": PickPlaceTeacher,  # Multi-stage like pick-place
    "peg_insert_side": PickPlaceTeacher,  # Precision placement
    "hammer": RandomMetaWorldTeacher,  # Complex, use random
    "lever_pull": DoorOpenTeacher,  # Similar pulling motion
    "window_open": DoorOpenTeacher,  # Similar to door
    "window_close": DoorOpenTeacher,
    "sweep_into": RandomMetaWorldTeacher,  # Complex sweeping motion
}


def create_metaworld_teacher(task_name: str, **kwargs) -> TeacherPolicy:
    """Create teacher for MetaWorld task.

    Args:
        task_name: Task name (e.g., 'reach')
        **kwargs: Additional arguments for teacher

    Returns:
        TeacherPolicy instance
    """
    if task_name not in METAWORLD_TEACHERS:
        available = list(METAWORLD_TEACHERS.keys())
        warnings.warn(f"No specific teacher for {task_name}, using random teacher")
        return RandomMetaWorldTeacher(**kwargs)

    teacher_class = METAWORLD_TEACHERS[task_name]
    return teacher_class(**kwargs)