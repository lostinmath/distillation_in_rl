"""Teacher policies for MuJoCo Gym environments.

Implements control-based and heuristic teachers for classic MuJoCo tasks.
These provide strong baselines for various locomotion and manipulation tasks.
"""

import numpy as np
import torch
from typing import Union, Dict, Any
import warnings

from .base import TeacherPolicy


class PendulumTeacher(TeacherPolicy):
    """Teacher for pendulum swing-up using energy-based control."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.target_energy = 1.0  # Target energy for upright position

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for pendulum swing-up."""
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
            action = self._pendulum_policy(obs_i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        return result

    def _pendulum_policy(self, obs: np.ndarray) -> np.ndarray:
        """Energy-based pendulum control."""
        # Pendulum observation: [cos(theta), sin(theta), angular_velocity]
        if len(obs) >= 3:
            cos_theta = obs[0]
            sin_theta = obs[1]
            angular_vel = obs[2]

            # Calculate current energy
            theta = np.arctan2(sin_theta, cos_theta)
            energy = 0.5 * angular_vel**2 + (1 - cos_theta)

            # Energy-based control
            if abs(theta) < 0.5 and abs(angular_vel) < 2.0:
                # Near upright: stabilizing control
                action = -10.0 * theta - 3.0 * angular_vel
            else:
                # Swing up: energy control
                energy_error = self.target_energy - energy
                action = np.sign(angular_vel * cos_theta) * energy_error * 5.0

            action = np.clip(action, -2.0, 2.0)
        else:
            action = 0.0

        return np.array([action])


class ReacherTeacher(TeacherPolicy):
    """Teacher for 2D reacher using inverse kinematics."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for reacher task."""
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
        """Simple reacher policy using target direction."""
        # Reacher obs: [joint_angles, joint_velocities, end_effector_pos, target_pos]
        if len(obs) >= 8:
            # Extract target relative position (usually last 2 elements)
            target_rel = obs[-2:]
            # Simple proportional control
            action = target_rel * 5.0
        else:
            action = np.array([0.1, 0.1])

        return np.clip(action, -1.0, 1.0)


class HalfCheetahTeacher(TeacherPolicy):
    """Teacher for HalfCheetah using running gait patterns."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.step_counter = 0
        self.gait_frequency = 2.0

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
            action = self._cheetah_gait_policy(obs_i, i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        self.step_counter += 1
        return result

    def _cheetah_gait_policy(self, obs: np.ndarray, env_idx: int) -> np.ndarray:
        """HalfCheetah running gait."""
        # Phase for gait timing
        phase = (self.step_counter + env_idx) * 0.15

        # HalfCheetah has 6 actuators
        actions = np.zeros(6)

        # Coordinated running gait
        # Back leg
        actions[0] = 0.5 * np.sin(phase)  # Back thigh
        actions[1] = 0.8 * (np.cos(phase) - 0.5)  # Back shin
        actions[2] = 0.3 * np.sin(phase + np.pi/4)  # Back foot

        # Front leg (opposite phase)
        actions[3] = 0.5 * np.sin(phase + np.pi)  # Front thigh
        actions[4] = 0.8 * (np.cos(phase + np.pi) - 0.5)  # Front shin
        actions[5] = 0.3 * np.sin(phase + np.pi + np.pi/4)  # Front foot

        return np.clip(actions, -1.0, 1.0)


class AntTeacher(TeacherPolicy):
    """Teacher for Ant using quadruped gait."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
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
        """Ant quadruped gait."""
        # Phase for gait
        phase = (self.step_counter + env_idx) * 0.1

        # Ant has 8 actuators (2 per leg)
        actions = np.zeros(8)

        # Quadruped trot gait
        for leg in range(4):
            leg_phase = phase + leg * np.pi/2
            # Hip joint
            actions[leg*2] = 0.4 * np.sin(leg_phase)
            # Knee joint
            actions[leg*2 + 1] = 0.3 * np.cos(leg_phase)

        return np.clip(actions, -1.0, 1.0)


class Walker2dTeacher(TeacherPolicy):
    """Teacher for Walker2d using bipedal walking pattern."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.step_counter = 0

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for Walker2d."""
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
            action = self._walker_gait_policy(obs_i, i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        self.step_counter += 1
        return result

    def _walker_gait_policy(self, obs: np.ndarray, env_idx: int) -> np.ndarray:
        """Walker2d bipedal gait."""
        # Walking phase
        phase = (self.step_counter + env_idx) * 0.08

        # Walker2d has 6 actuators
        actions = np.zeros(6)

        # Bipedal walking pattern
        # Right leg
        actions[0] = 0.3 * np.sin(phase)  # Right thigh
        actions[1] = 0.4 * np.abs(np.sin(phase))  # Right shin
        actions[2] = 0.2 * np.sin(phase)  # Right foot

        # Left leg (opposite phase)
        actions[3] = 0.3 * np.sin(phase + np.pi)  # Left thigh
        actions[4] = 0.4 * np.abs(np.sin(phase + np.pi))  # Left shin
        actions[5] = 0.2 * np.sin(phase + np.pi)  # Left foot

        return np.clip(actions, -1.0, 1.0)


class HopperTeacher(TeacherPolicy):
    """Teacher for Hopper using hopping pattern."""

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        super().__init__(action_space, observation_space, device)
        self.step_counter = 0

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for Hopper."""
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
            action = self._hopper_policy(obs_i, i)
            actions.append(action)

        result = np.array(actions, dtype=np.float32)

        if batch_size == 1:
            result = result[0]

        self.step_counter += 1
        return result

    def _hopper_policy(self, obs: np.ndarray, env_idx: int) -> np.ndarray:
        """Hopper hopping policy."""
        # Hopping phase
        phase = (self.step_counter + env_idx) * 0.12

        # Hopper has 3 actuators
        actions = np.zeros(3)

        # Hopping pattern
        hop_cycle = np.sin(phase)
        actions[0] = 0.4 * hop_cycle  # Thigh
        actions[1] = 0.6 * np.abs(hop_cycle)  # Leg
        actions[2] = 0.3 * hop_cycle  # Foot

        return np.clip(actions, -1.0, 1.0)


class RandomMuJoCoTeacher(TeacherPolicy):
    """Random baseline teacher for MuJoCo environments."""

    def __init__(self, action_space=None, observation_space=None, device="cpu", action_dim=6):
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


# Registry of MuJoCo teachers
MUJOCO_TEACHERS = {
    # Control
    "inverted_pendulum": lambda **kwargs: RandomMuJoCoTeacher(action_dim=1, **kwargs),
    "inverted_double_pendulum": lambda **kwargs: RandomMuJoCoTeacher(action_dim=1, **kwargs),
    "pendulum": PendulumTeacher,
    "mountain_car_continuous": lambda **kwargs: RandomMuJoCoTeacher(action_dim=1, **kwargs),
    "acrobot": lambda **kwargs: RandomMuJoCoTeacher(action_dim=1, **kwargs),

    # Locomotion
    "walker2d": Walker2dTeacher,
    "humanoid": lambda **kwargs: RandomMuJoCoTeacher(action_dim=17, **kwargs),
    "humanoid_standup": lambda **kwargs: RandomMuJoCoTeacher(action_dim=17, **kwargs),
    "ant": AntTeacher,
    "halfcheetah": HalfCheetahTeacher,
    "hopper": HopperTeacher,
    "swimmer": lambda **kwargs: RandomMuJoCoTeacher(action_dim=2, **kwargs),

    # Manipulation
    "reacher": ReacherTeacher,
    "pusher": lambda **kwargs: RandomMuJoCoTeacher(action_dim=7, **kwargs),
    "striker": lambda **kwargs: RandomMuJoCoTeacher(action_dim=7, **kwargs),
    "thrower": lambda **kwargs: RandomMuJoCoTeacher(action_dim=7, **kwargs),
}


def create_mujoco_teacher(env_name: str, **kwargs) -> TeacherPolicy:
    """Create teacher for MuJoCo environment.

    Args:
        env_name: Environment name (e.g., 'halfcheetah')
        **kwargs: Additional arguments for teacher

    Returns:
        TeacherPolicy instance
    """
    if env_name not in MUJOCO_TEACHERS:
        available = list(MUJOCO_TEACHERS.keys())
        warnings.warn(f"No specific teacher for {env_name}, using random teacher")
        return RandomMuJoCoTeacher(**kwargs)

    teacher_factory = MUJOCO_TEACHERS[env_name]
    return teacher_factory(**kwargs)