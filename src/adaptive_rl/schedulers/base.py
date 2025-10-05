"""Base class for policy scheduling strategies in teacher-guided RL.

This module defines the abstract interface for scheduling when to use
teacher vs student policies during training.
"""

from abc import ABC, abstractmethod

import torch


class PolicyScheduler(ABC):
    """Abstract base class for policy scheduling strategies.

    Determines when to use teacher guidance vs student exploration
    during reinforcement learning training.
    """

    def __init__(
        self,
        student_policy=None,
        teacher_policy=None,
        num_envs: int = 1,
        trust_length: int = 5,
        device: str = "cpu",
        log_dir: str | None = None,
        **kwargs,
    ):
        """Initialize the policy scheduler.

        Args:
            student_policy: Student policy (can be None)
            teacher_policy: Teacher policy (can be None)
            num_envs: Number of parallel environments
            trust_length: Number of steps to trust a policy before re-evaluating
            device: Device to run on (cpu/cuda)
            log_dir: Directory for logging policy history
            **kwargs: Additional strategy-specific parameters
        """
        self.student_policy = student_policy
        self.teacher_policy = teacher_policy
        self.num_envs = num_envs
        self.trust_length = trust_length
        self.device = device
        self.log_dir = log_dir

        # Track which policy was last used for each environment
        self.last_used_policy = ["teacher"] * num_envs
        self.steps_taken_on_last_policy = [0] * num_envs

        # For reward-based strategies
        self.prev_prev_reward = [-1] * num_envs

        # Statistics tracking
        self.teacher_usage_count = 0
        self.student_usage_count = 0
        self.total_steps = 0
        self.switch_count = 0

    def reset(self):
        """Reset internal state for new episode."""
        self.last_used_policy = ["teacher"] * self.num_envs
        self.steps_taken_on_last_policy = [0] * self.num_envs
        self.prev_prev_reward = [-1] * self.num_envs
        self.teacher_usage_count = 0
        self.student_usage_count = 0
        self.switch_count = 0

    @abstractmethod
    def choose_policy_type(
        self,
        iteration: int,
        global_step: int,
        steps_since_reset: torch.Tensor,
        prev_reward: torch.Tensor,
    ) -> list[str]:
        """Choose which policy to use for each environment.

        Args:
            iteration: Current training iteration
            global_step: Total number of environment steps taken
            steps_since_reset: Steps since last reset for each env
            prev_reward: Previous reward for each env (-1 if reset)

        Returns:
            List of policy names ("teacher" or "student") for each env
        """

    def update_statistics(self, policy_types: list[str]):
        """Update usage statistics based on chosen policies.

        Args:
            policy_types: List of chosen policy types
        """
        for i, policy in enumerate(policy_types):
            # Track switches
            if policy != self.last_used_policy[i]:
                self.switch_count += 1

            # Count usage
            if policy == "teacher":
                self.teacher_usage_count += 1
            else:
                self.student_usage_count += 1

        self.total_steps += len(policy_types)

    def get_statistics(self) -> dict[str, float]:
        """Get current scheduling statistics.

        Returns:
            Dictionary of statistics
        """
        total = max(1, self.teacher_usage_count + self.student_usage_count)
        return {
            "teacher_usage_ratio": self.teacher_usage_count / total,
            "student_usage_ratio": self.student_usage_count / total,
            "switch_frequency": self.switch_count / max(1, self.total_steps),
            "total_steps": self.total_steps,
            "total_switches": self.switch_count,
        }

    def get_metrics(self) -> dict[str, float]:
        """Get current scheduler metrics (alias for get_statistics)."""
        stats = self.get_statistics()
        return {f"scheduler/{key}": value for key, value in stats.items()}

    def should_switch_policy(
        self, env_idx: int, current_policy: str, reward: float, steps_on_policy: int
    ) -> bool:
        """Helper method to determine if policy should be switched.

        Can be overridden by subclasses for specific logic.

        Args:
            env_idx: Environment index
            current_policy: Currently used policy
            reward: Current reward
            steps_on_policy: Steps taken on current policy

        Returns:
            Whether to switch policies
        """
        return False
