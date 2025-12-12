"""Teacher policies for PyBullet robotics environments.

Implements control-theoretic and heuristic teachers for PyBullet tasks.
Covers both manipulation and locomotion scenarios.
"""

import numpy as np
import torch
from typing import Union, Dict, Any
import warnings

from .base import TeacherPolicy


class KukaReachTeacher(TeacherPolicy):
    """Teacher for Kuka reaching tasks using inverse kinematics approach."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.target_threshold = 0.05
        self.action_scale = 2.0

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for Kuka reaching."""
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
            action = self._kuka_reach_policy(obs_i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        return result

    def _kuka_reach_policy(self, obs: np.ndarray) -> np.ndarray:
        """Simple Kuka reaching policy."""
        # Extract end effector and target positions from observation
        # Kuka observations typically include joint positions, end effector pose, target

        # Simplified: assume first 3 elements are end effector position,
        # next 3 are target position
        if len(obs) >= 6:
            ee_pos = obs[:3]
            target_pos = obs[3:6]
        else:
            # Fallback for different observation formats
            ee_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
            target_pos = np.array([0.5, 0.0, 0.5])

        # Calculate direction and distance
        direction = target_pos - ee_pos
        distance = np.linalg.norm(direction)

        if distance > self.target_threshold:
            # Proportional control toward target
            action = direction * self.action_scale
        else:
            # Small corrections when close
            action = direction * 0.1

        # Determine action dimension based on action space
        action_dim = getattr(self.action_space, 'shape', (7,))[0] if self.action_space else 7

        if len(action) < action_dim:
            # Pad with zeros for additional DOF
            action = np.pad(action, (0, action_dim - len(action)), mode='constant')
        elif len(action) > action_dim:
            # Truncate if too long
            action = action[:action_dim]

        return np.clip(action, -1.0, 1.0)


class AntTeacher(TeacherPolicy):
    """Teacher for Ant locomotion using gait patterns."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.gait_frequency = 2.0
        self.step_counter = 0

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for Ant locomotion."""
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
            action = self._ant_gait_policy(obs_i, i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        self.step_counter += 1
        return result

    def _ant_gait_policy(self, obs: np.ndarray, env_idx: int) -> np.ndarray:
        """Ant gait policy using sinusoidal patterns."""
        # Create phase for gait
        phase = (self.step_counter + env_idx) * 0.1

        # Generate coordinated leg movements
        # Ant typically has 8 actuators (2 per leg)
        action_dim = 8

        # Alternating gait pattern
        actions = np.zeros(action_dim)

        for i in range(4):  # 4 legs
            leg_phase = phase + i * np.pi / 2  # Phase offset for each leg
            # Two joints per leg
            actions[i*2] = 0.5 * np.sin(leg_phase)  # Hip
            actions[i*2 + 1] = 0.3 * np.cos(leg_phase)  # Knee/ankle

        return np.clip(actions, -1.0, 1.0)


class HumanoidTeacher(TeacherPolicy):
    """Teacher for humanoid locomotion."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.step_counter = 0
        self.gait_frequency = 1.5

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for humanoid walking."""
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
            action = self._humanoid_walk_policy(obs_i, i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        self.step_counter += 1
        return result

    def _humanoid_walk_policy(self, obs: np.ndarray, env_idx: int) -> np.ndarray:
        """Humanoid walking policy."""
        # Phase for walking gait
        phase = (self.step_counter + env_idx) * 0.05

        # Humanoid typically has 17 actuators
        action_dim = 17
        actions = np.zeros(action_dim)

        # Simple walking pattern
        # Hip joints (alternating)
        actions[0] = 0.3 * np.sin(phase)  # Right hip
        actions[3] = 0.3 * np.sin(phase + np.pi)  # Left hip

        # Knee joints
        actions[1] = 0.2 * np.abs(np.sin(phase))  # Right knee
        actions[4] = 0.2 * np.abs(np.sin(phase + np.pi))  # Left knee

        # Ankle joints
        actions[2] = 0.1 * np.sin(phase)  # Right ankle
        actions[5] = 0.1 * np.sin(phase + np.pi)  # Left ankle

        # Arm swinging (opposite to legs)
        actions[6] = 0.2 * np.sin(phase + np.pi)  # Right arm
        actions[9] = 0.2 * np.sin(phase)  # Left arm

        return np.clip(actions, -1.0, 1.0)


class HalfCheetahTeacher(TeacherPolicy):
    """Teacher for HalfCheetah running."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.step_counter = 0

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for HalfCheetah running."""
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
            action = self._cheetah_run_policy(obs_i, i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        self.step_counter += 1
        return result

    def _cheetah_run_policy(self, obs: np.ndarray, env_idx: int) -> np.ndarray:
        """HalfCheetah running policy."""
        # Running gait phase
        phase = (self.step_counter + env_idx) * 0.15

        # HalfCheetah has 6 actuators
        action_dim = 6
        actions = np.zeros(action_dim)

        # Running gait for back and front legs
        actions[0] = 0.4 * np.sin(phase)  # Back thigh
        actions[1] = 0.6 * np.cos(phase)  # Back shin
        actions[2] = 0.3 * np.sin(phase + np.pi/4)  # Back foot

        actions[3] = 0.4 * np.sin(phase + np.pi)  # Front thigh
        actions[4] = 0.6 * np.cos(phase + np.pi)  # Front shin
        actions[5] = 0.3 * np.sin(phase + np.pi + np.pi/4)  # Front foot

        return np.clip(actions, -1.0, 1.0)


class ReacherTeacher(TeacherPolicy):
    """Teacher for Reacher tasks."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for reaching task."""
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
            action = self._reacher_policy(obs_i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        return result

    def _reacher_policy(self, obs: np.ndarray) -> np.ndarray:
        """Simple reacher policy."""
        # Extract end effector and target positions
        # Reacher obs typically includes joint angles, velocities, and target info
        if len(obs) >= 4:
            # Simple proportional control based on observation
            # Last elements often contain target information
            target_info = obs[-2:]  # Last 2 elements might be target relative position
            action = target_info * 2.0  # Proportional control
        else:
            action = np.array([0.1, 0.1])  # Default small movement

        return np.clip(action, -1.0, 1.0)


class RandomPyBulletTeacher(TeacherPolicy):
    """Random baseline teacher for PyBullet environments."""

    def __init__(self, action_space=None, observation_space=None, device="cpu", action_dim=8):
        super().__init__(action_space, observation_space, device)
        self.action_dim = action_dim
        if action_space and hasattr(action_space, 'shape'):
            self.action_dim = action_space.shape[0]

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate random actions."""
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()

        # Handle batch dimension
        if len(observations.shape) == 1:
            batch_size = 1
        else:
            batch_size = observations.shape[0]

        # Random actions in [-1, 1]
        actions = np.random.uniform(-1.0, 1.0, size=(batch_size, self.action_dim))

        if batch_size == 1:
            actions = actions[0]

        return actions.astype(np.float32)


# Registry of PyBullet teachers
PYBULLET_TEACHERS = {
    # Manipulation
    "kuka_reach": KukaReachTeacher,
    "kuka_grasp": KukaReachTeacher,  # Similar reaching motion
    "kuka_diverse": KukaReachTeacher,
    "reacher": ReacherTeacher,
    "pusher": ReacherTeacher,  # Similar arm movement
    "striker": ReacherTeacher,
    "thrower": ReacherTeacher,

    # Locomotion - quadrupeds
    "ant": AntTeacher,
    "halfcheetah": HalfCheetahTeacher,
    "walker2d": lambda **kwargs: RandomPyBulletTeacher(action_dim=6, **kwargs),

    # Locomotion - humanoids
    "humanoid": HumanoidTeacher,
    "humanoid_flagrun": HumanoidTeacher,
    "humanoid_flagrun_harder": HumanoidTeacher,

    # Locomotion - hoppers
    "hopper": lambda **kwargs: RandomPyBulletTeacher(action_dim=3, **kwargs),
    "inverted_pendulum": lambda **kwargs: RandomPyBulletTeacher(action_dim=1, **kwargs),
    "inverted_double_pendulum": lambda **kwargs: RandomPyBulletTeacher(action_dim=1, **kwargs),

    # Specialized
    "racecar": lambda **kwargs: RandomPyBulletTeacher(action_dim=2, **kwargs),
    "racecar_zed": lambda **kwargs: RandomPyBulletTeacher(action_dim=2, **kwargs),
    "minitaur": lambda **kwargs: RandomPyBulletTeacher(action_dim=8, **kwargs),
    "minitaur_duck": lambda **kwargs: RandomPyBulletTeacher(action_dim=8, **kwargs),
}


def create_pybullet_teacher(env_name: str, **kwargs) -> TeacherPolicy:
    """Create teacher for PyBullet environment.

    Args:
        env_name: Environment name (e.g., 'ant')
        **kwargs: Additional arguments for teacher

    Returns:
        TeacherPolicy instance
    """
    if env_name not in PYBULLET_TEACHERS:
        available = list(PYBULLET_TEACHERS.keys())
        warnings.warn(f"No specific teacher for {env_name}, using random teacher")
        return RandomPyBulletTeacher(**kwargs)

    teacher_factory = PYBULLET_TEACHERS[env_name]
    return teacher_factory(**kwargs)