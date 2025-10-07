"""Teacher policies for dm_control environments."""

import numpy as np
import torch
from typing import Union, Dict, Any

from .base import TeacherPolicy


class CheetahRunTeacher(TeacherPolicy):
    """Hand-coded teacher for cheetah_run task.

    Strategy: Maintain forward momentum with energy-efficient gait.
    Based on observations of optimal cheetah locomotion.
    """

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        """Initialize cheetah run teacher."""
        super().__init__(action_space, observation_space, device)

        # Cheetah-specific parameters
        self.target_velocity = 10.0  # Desired forward velocity
        self.energy_efficiency = 0.8  # Balance speed vs energy
        self.stability_factor = 0.5  # Maintain stability

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for cheetah running.

        Observation space (flattened):
        - 0-7: joint positions (rootx, rooty, rotz, bthigh, bshin, bfoot, fthigh, fshin, ffoot)
        - 8-16: joint velocities

        Action space (6D continuous):
        - Back thigh, back shin, back foot, front thigh, front shin, front foot
        """
        # Convert to tensor if needed
        if isinstance(observations, np.ndarray):
            return_numpy = True
            obs_tensor = torch.from_numpy(observations).float().to(self.device)
        else:
            return_numpy = False
            obs_tensor = observations.float()

        # Handle batch dimension
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False

        batch_size = obs_tensor.shape[0]
        actions = torch.zeros(batch_size, 6, device=self.device)

        for i in range(batch_size):
            obs = obs_tensor[i]

            # Extract relevant state information
            # Positions (assuming flattened dm_control observation)
            if len(obs) >= 17:  # Full observation
                rootx_pos = obs[0]
                joint_positions = obs[3:8]  # bthigh, bshin, bfoot, fthigh, fshin
                joint_velocities = obs[8:13] if len(obs) >= 13 else torch.zeros(5)
                forward_velocity = obs[8] if len(obs) >= 9 else 0.0
            else:
                # Simplified observation handling
                joint_positions = obs[:5] if len(obs) >= 5 else obs
                joint_velocities = obs[5:10] if len(obs) >= 10 else torch.zeros(5)
                forward_velocity = joint_velocities[0] if len(joint_velocities) > 0 else 0.0

            # Simple running gait controller
            # Create a phase-based gait pattern
            time_phase = (i * 0.1) % (2 * np.pi)  # Simple time-based phase

            # Target joint positions for running gait
            # Back leg (indices 0-2: thigh, shin, foot)
            back_thigh_target = 0.3 * np.sin(time_phase)
            back_shin_target = -0.8 + 0.4 * np.cos(time_phase)
            back_foot_target = 0.2 * np.sin(time_phase + np.pi/4)

            # Front leg (indices 3-5: thigh, shin, foot)
            front_thigh_target = 0.3 * np.sin(time_phase + np.pi)
            front_shin_target = -0.8 + 0.4 * np.cos(time_phase + np.pi)
            front_foot_target = 0.2 * np.sin(time_phase + np.pi + np.pi/4)

            # PD controller to reach target positions
            kp = 10.0  # Proportional gain
            kd = 1.0   # Derivative gain

            # Back leg actions
            actions[i, 0] = kp * (back_thigh_target - joint_positions[0]) - kd * joint_velocities[0]
            actions[i, 1] = kp * (back_shin_target - joint_positions[1]) - kd * joint_velocities[1]
            actions[i, 2] = kp * (back_foot_target - joint_positions[2]) - kd * joint_velocities[2]

            # Front leg actions
            actions[i, 3] = kp * (front_thigh_target - joint_positions[3]) - kd * joint_velocities[3]
            actions[i, 4] = kp * (front_shin_target - joint_positions[4]) - kd * joint_velocities[4]
            actions[i, 5] = front_foot_target * 0.5  # Simpler foot control

            # Velocity feedback to maintain target speed
            velocity_error = self.target_velocity - forward_velocity
            velocity_correction = 0.1 * velocity_error

            # Apply velocity correction to driving joints
            actions[i, 0] += velocity_correction
            actions[i, 3] += velocity_correction

        # Clip actions to valid range [-1, 1]
        actions = torch.clamp(actions, -1.0, 1.0)

        # Handle single observation case
        if single_obs:
            actions = actions[0]

        # Convert back to numpy if needed
        if return_numpy:
            actions = actions.cpu().numpy()

        return actions


class WalkerWalkTeacher(TeacherPolicy):
    """Hand-coded teacher for walker_walk task.

    Strategy: Stable bipedal walking with balance control.
    """

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        """Initialize walker teacher."""
        super().__init__(action_space, observation_space, device)

        self.target_velocity = 1.0
        self.balance_gain = 5.0

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for walker walking."""
        # Convert to tensor if needed
        if isinstance(observations, np.ndarray):
            return_numpy = True
            obs_tensor = torch.from_numpy(observations).float().to(self.device)
        else:
            return_numpy = False
            obs_tensor = observations.float()

        # Handle batch dimension
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False

        batch_size = obs_tensor.shape[0]
        action_dim = 6  # Walker has 6 actuators

        # Simple walking controller
        actions = torch.zeros(batch_size, action_dim, device=self.device)

        for i in range(batch_size):
            # Simple oscillating pattern for walking
            time_phase = (i * 0.05) % (2 * np.pi)

            # Hip, knee, ankle for both legs
            actions[i, 0] = 0.3 * np.sin(time_phase)      # Right hip
            actions[i, 1] = 0.5 * np.cos(time_phase)      # Right knee
            actions[i, 2] = 0.2 * np.sin(time_phase)      # Right ankle
            actions[i, 3] = 0.3 * np.sin(time_phase + np.pi)  # Left hip
            actions[i, 4] = 0.5 * np.cos(time_phase + np.pi)  # Left knee
            actions[i, 5] = 0.2 * np.sin(time_phase + np.pi)  # Left ankle

        # Clip actions
        actions = torch.clamp(actions, -1.0, 1.0)

        if single_obs:
            actions = actions[0]

        if return_numpy:
            actions = actions.cpu().numpy()

        return actions


class ReacherTeacher(TeacherPolicy):
    """Hand-coded teacher for reacher tasks.

    Strategy: Direct path to target with smooth movements.
    """

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        """Initialize reacher teacher."""
        super().__init__(action_space, observation_space, device)

    def act(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Generate actions for reaching task."""
        # Convert to tensor if needed
        if isinstance(observations, np.ndarray):
            return_numpy = True
            obs_tensor = torch.from_numpy(observations).float().to(self.device)
        else:
            return_numpy = False
            obs_tensor = observations.float()

        # Handle batch dimension
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False

        batch_size = obs_tensor.shape[0]
        action_dim = 2  # Reacher has 2 actuators

        actions = torch.zeros(batch_size, action_dim, device=self.device)

        # Simple proportional controller towards target
        # This assumes target position is in the observation
        # (implementation would depend on exact observation structure)

        for i in range(batch_size):
            # Placeholder: simple sinusoidal movement
            actions[i, 0] = 0.5 * np.sin(i * 0.1)
            actions[i, 1] = 0.5 * np.cos(i * 0.1)

        actions = torch.clamp(actions, -1.0, 1.0)

        if single_obs:
            actions = actions[0]

        if return_numpy:
            actions = actions.cpu().numpy()

        return actions


# Registry of dm_control teachers
DM_CONTROL_TEACHERS = {
    "cheetah_run": CheetahRunTeacher,
    "walker_walk": WalkerWalkTeacher,
    "walker_stand": WalkerWalkTeacher,  # Can reuse for standing
    "reacher_easy": ReacherTeacher,
    "reacher_hard": ReacherTeacher,
}


def create_dm_control_teacher(env_name: str, **kwargs) -> TeacherPolicy:
    """Create teacher for dm_control environment.

    Args:
        env_name: Environment name (e.g., 'cheetah_run')
        **kwargs: Additional arguments for teacher

    Returns:
        TeacherPolicy instance
    """
    if env_name not in DM_CONTROL_TEACHERS:
        available = list(DM_CONTROL_TEACHERS.keys())
        raise ValueError(f"No teacher available for {env_name}. Available: {available}")

    teacher_class = DM_CONTROL_TEACHERS[env_name]
    return teacher_class(**kwargs)