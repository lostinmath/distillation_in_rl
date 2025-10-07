"""Hand-coded optimal teacher policies for specific environments.

These serve as upper-bound baselines for teacher quality.
"""

import numpy as np
import torch

from .base import TeacherPolicy


class CartPoleOptimalTeacher(TeacherPolicy):
    """Hand-coded optimal policy for CartPole-v1.

    Uses a simple but effective heuristic based on pole angle and velocity.
    """

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        """Initialize CartPole optimal teacher."""
        super().__init__(action_space, observation_space, device)

    def act(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Generate optimal actions for CartPole.

        Strategy: Move cart in direction to keep pole upright,
        considering both angle and angular velocity.

        Args:
            obs: CartPole observations [cart_pos, cart_vel, pole_angle, pole_vel]

        Returns:
            Actions (0=left, 1=right)
        """
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            return_numpy = True
            obs_tensor = torch.from_numpy(obs).to(self.device)
        else:
            return_numpy = False
            obs_tensor = obs

        # Handle both single and batched observations
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False

        # Extract relevant features
        pole_angle = obs_tensor[:, 2]
        pole_velocity = obs_tensor[:, 3]

        # Simple but effective heuristic:
        # Consider both current angle and predicted future angle
        angle_threshold = 0.0
        future_angle = pole_angle + 0.5 * pole_velocity

        # Move in direction to correct the pole
        actions = (future_angle > angle_threshold).long()

        # Handle single observation
        if single_obs:
            actions = actions[0]

        # Convert back to numpy if needed
        if return_numpy:
            actions = actions.cpu().numpy()

        return actions


class AcrobotOptimalTeacher(TeacherPolicy):
    """Energy-based policy for Acrobot-v1.

    Uses a simple but effective strategy based on energy pumping.
    """

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        """Initialize Acrobot optimal teacher."""
        super().__init__(action_space, observation_space, device)

    def act(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Generate actions using energy-based strategy for Acrobot.

        Strategy: Pump energy when pendulum is moving in right direction.

        Args:
            obs: Acrobot observations [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]

        Returns:
            Actions (0=negative torque, 1=no torque, 2=positive torque)
        """
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            return_numpy = True
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
        else:
            return_numpy = False
            obs_tensor = obs.float()

        # Handle both single and batched observations
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False

        # Extract state variables
        cos_theta1 = obs_tensor[:, 0]
        sin_theta1 = obs_tensor[:, 1]
        cos_theta2 = obs_tensor[:, 2]
        sin_theta2 = obs_tensor[:, 3]
        theta1_dot = obs_tensor[:, 4]
        theta2_dot = obs_tensor[:, 5]

        # Compute angles from cos/sin
        theta1 = torch.atan2(sin_theta1, cos_theta1)
        theta2 = torch.atan2(sin_theta2, cos_theta2)

        # Simple energy-based strategy
        # When the second link is swinging up and has positive velocity, apply torque
        # in the direction of motion to pump energy
        actions = torch.ones(obs_tensor.size(0), dtype=torch.long, device=self.device)  # Default: no torque

        # Apply positive torque when the tip is moving upward and in the right direction
        tip_velocity_y = -1 * sin_theta1 * theta1_dot + torch.cos(theta1 + theta2) * (theta1_dot + theta2_dot)

        # Pump energy when moving in beneficial direction
        pump_condition = (tip_velocity_y > 0) & (torch.cos(theta1 + theta2) > -0.8)
        actions[pump_condition] = 2  # Positive torque

        # Apply negative torque in opposite conditions
        brake_condition = (tip_velocity_y < 0) & (torch.cos(theta1 + theta2) > -0.8)
        actions[brake_condition] = 0  # Negative torque

        # Handle single observation
        if single_obs:
            actions = actions[0]

        # Convert back to numpy if needed
        if return_numpy:
            actions = actions.cpu().numpy()

        return actions


class LunarLanderOptimalTeacher(TeacherPolicy):
    """Heuristic policy for LunarLander-v2.

    Based on the OpenAI Gym heuristic demonstration agent.
    """

    def __init__(self, action_space=None, observation_space=None, device="cpu"):
        """Initialize LunarLander optimal teacher."""
        super().__init__(action_space, observation_space, device)

    def act(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Generate actions using heuristic for LunarLander.

        Uses position, velocity, and angle to determine thrust.

        Args:
            obs: LunarLander observations [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]

        Returns:
            Actions (0=nothing, 1=left engine, 2=main engine, 3=right engine)
        """
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            return_numpy = True
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
        else:
            return_numpy = False
            obs_tensor = obs.float()

        # Handle both single and batched observations
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False

        batch_size = obs_tensor.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Extract features
        x = obs_tensor[:, 0]
        y = obs_tensor[:, 1]
        vx = obs_tensor[:, 2]
        vy = obs_tensor[:, 3]
        angle = obs_tensor[:, 4]
        angular_vel = obs_tensor[:, 5]
        left_leg = obs_tensor[:, 6]
        right_leg = obs_tensor[:, 7]

        # Heuristic control logic
        for i in range(batch_size):
            # Target hover position
            target_x = 0.0
            target_y = 0.0

            # Angle control - try to stay upright
            angle_todo = angle[i] * 0.5 + angular_vel[i] * 1.0

            # Horizontal control
            hover_todo = (target_x - x[i]) * 0.5 - vx[i] * 0.5

            # Combined control
            if x[i] < -0.4:
                angle_todo = -0.5
            elif x[i] > 0.4:
                angle_todo = 0.5

            # Vertical control
            if y[i] > 0.3:
                # High altitude - control horizontal position
                if abs(angle_todo) > 0.4:
                    # Fire side engines to control angle
                    if angle_todo > 0:
                        actions[i] = 3  # Right engine
                    else:
                        actions[i] = 1  # Left engine
                elif abs(hover_todo) > 0.3:
                    # Control horizontal movement
                    if hover_todo > 0:
                        actions[i] = 3  # Right engine
                    else:
                        actions[i] = 1  # Left engine
                else:
                    actions[i] = 0  # Do nothing
            # Low altitude - focus on landing
            elif left_leg[i] or right_leg[i]:
                # Touching ground - minimal thrust
                actions[i] = 0
            elif vy[i] < -0.3:
                # Falling too fast - main engine
                actions[i] = 2
            elif abs(angle[i]) > 0.3:
                # Too tilted - correct angle
                if angle[i] > 0:
                    actions[i] = 3
                else:
                    actions[i] = 1
            else:
                actions[i] = 2 if vy[i] < -0.1 else 0

        # Handle single observation
        if single_obs:
            actions = actions[0]

        # Convert back to numpy if needed
        if return_numpy:
            actions = actions.cpu().numpy()

        return actions


# Registry of optimal teachers for different environments
OPTIMAL_TEACHERS = {
    "CartPole-v1": CartPoleOptimalTeacher,
    "CartPole-v0": CartPoleOptimalTeacher,
    "Acrobot-v1": AcrobotOptimalTeacher,
    "LunarLander-v2": LunarLanderOptimalTeacher,
    "LunarLander-v3": LunarLanderOptimalTeacher,
}


def create_optimal_teacher(env_id: str, **kwargs) -> TeacherPolicy:
    """Create an optimal teacher for the specified environment.

    Args:
        env_id: Environment ID
        **kwargs: Additional arguments for the teacher

    Returns:
        Optimal teacher instance

    Raises:
        ValueError: If no optimal teacher exists for the environment
    """
    if env_id not in OPTIMAL_TEACHERS:
        raise ValueError(
            f"No optimal teacher available for {env_id}. "
            f"Available environments: {list(OPTIMAL_TEACHERS.keys())}"
        )

    teacher_class = OPTIMAL_TEACHERS[env_id]
    # Filter kwargs to only pass what the teacher constructor accepts
    valid_kwargs = {k: v for k, v in kwargs.items()
                   if k in ['action_space', 'observation_space', 'device']}
    return teacher_class(**valid_kwargs)
