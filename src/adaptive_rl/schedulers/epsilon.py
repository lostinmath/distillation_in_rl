"""Epsilon-based scheduling strategies for teacher-guided RL.

Includes both fixed epsilon and linearly decreasing epsilon strategies.
"""

import random

import torch

from .base import PolicyScheduler


class EpsilonScheduler(PolicyScheduler):
    """Fixed epsilon scheduling strategy.

    Switches between teacher and student with fixed probability epsilon
    after a trust period expires.
    """

    def __init__(
        self,
        num_envs: int,
        epsilon: float = 0.1,
        trust_length: int = 5,
        device: str = "cpu",
        log_dir: str = None,
        **kwargs,
    ):
        """Initialize epsilon scheduler.

        Args:
            num_envs: Number of parallel environments
            epsilon: Probability of using teacher (fixed)
            trust_length: Steps before considering switch
            device: Device to run on
            log_dir: Directory for logging
        """
        super().__init__(
            num_envs=num_envs,
            trust_length=trust_length,
            device=device,
            log_dir=log_dir,
            **kwargs
        )
        self.epsilon = epsilon

    def choose_policy_type(
        self,
        iteration: int,
        global_step: int,
        steps_since_reset: torch.Tensor,
        prev_reward: torch.Tensor,
    ) -> list[str]:
        """Choose policy with fixed epsilon probability.

        Args:
            iteration: Current training iteration
            global_step: Total environment steps
            steps_since_reset: Steps since env reset
            prev_reward: Previous step reward (-1 if reset)

        Returns:
            List of policy types for each environment
        """
        list_of_policies = []

        for i in range(self.num_envs):
            # Environment was reset - start with teacher
            if prev_reward[i] == -1:
                list_of_policies.append("teacher")
                self.steps_taken_on_last_policy[i] = 0

            # Check if trust period has expired
            elif self.steps_taken_on_last_policy[i] >= self.trust_length:
                sampled_epsilon = random.random()

                # Switch from student to teacher with probability epsilon
                if (
                    self.last_used_policy[i] == "student"
                    and sampled_epsilon < self.epsilon
                ):
                    list_of_policies.append("teacher")
                    self.last_used_policy[i] = "teacher"
                    self.steps_taken_on_last_policy[i] = 0

                # Switch from teacher to student with probability 1-epsilon
                elif (
                    self.last_used_policy[i] == "teacher"
                    and sampled_epsilon < 1 - self.epsilon
                ):
                    list_of_policies.append("student")
                    self.last_used_policy[i] = "student"
                    self.steps_taken_on_last_policy[i] = 0

                # Keep current policy
                else:
                    list_of_policies.append(self.last_used_policy[i])
            else:
                # Still in trust period - keep current policy
                list_of_policies.append(self.last_used_policy[i])

            self.steps_taken_on_last_policy[i] += 1

        # Update statistics
        self.update_statistics(list_of_policies)
        return list_of_policies


class EpsilonDecreasingScheduler(PolicyScheduler):
    """Linearly decreasing epsilon scheduling strategy.

    Starts with high probability of using teacher and linearly decreases
    to encourage more student exploration over time.
    """

    def __init__(
        self,
        num_envs: int,
        trust_length: int = 5,
        decrease_until_global_step: int = 500_000,
        device: str = "cpu",
        log_dir: str = None,
        **kwargs,
    ):
        """Initialize decreasing epsilon scheduler.

        Args:
            num_envs: Number of parallel environments
            trust_length: Steps before considering switch
            decrease_until_global_step: Steps over which to decrease epsilon from 1 to 0
            device: Device to run on
            log_dir: Directory for logging
        """
        super().__init__(
            num_envs=num_envs,
            trust_length=trust_length,
            device=device,
            log_dir=log_dir,
            **kwargs
        )
        self.decrease_until_global_step = decrease_until_global_step
        self.current_epsilon = 1.0

    def calculate_epsilon(self, global_step: int) -> float:
        """Calculate current epsilon value based on global step.

        Linearly decreases from 1.0 to 0.0 over decrease_until_global_step steps.

        Args:
            global_step: Current global training step

        Returns:
            Current epsilon value
        """
        if self.decrease_until_global_step <= 0:
            return 0.0

        self.current_epsilon = max(
            0.0, 1.0 - (global_step / self.decrease_until_global_step)
        )
        return self.current_epsilon

    def choose_policy_type(
        self,
        iteration: int,
        global_step: int,
        steps_since_reset: torch.Tensor,
        prev_reward: torch.Tensor,
    ) -> list[str]:
        """Choose policy with decreasing epsilon probability.

        Args:
            iteration: Current training iteration
            global_step: Total environment steps
            steps_since_reset: Steps since env reset
            prev_reward: Previous step reward (-1 if reset)

        Returns:
            List of policy types for each environment
        """
        list_of_policies = []

        # Calculate current epsilon
        cur_epsilon = self.calculate_epsilon(global_step)

        # If epsilon has decreased to 0, always use student
        if global_step >= self.decrease_until_global_step:
            list_of_policies = ["student"] * self.num_envs
        else:
            for i in range(self.num_envs):
                # Environment was reset - start with teacher
                if prev_reward[i] == -1:
                    list_of_policies.append("teacher")
                    self.last_used_policy[i] = "teacher"
                    self.steps_taken_on_last_policy[i] = 0

                # Check if trust period has expired
                elif self.steps_taken_on_last_policy[i] >= self.trust_length:
                    sampled_epsilon = random.random()

                    # Switch from student to teacher with probability epsilon
                    if (
                        self.last_used_policy[i] == "student"
                        and sampled_epsilon < cur_epsilon
                    ):
                        list_of_policies.append("teacher")
                        self.last_used_policy[i] = "teacher"
                        self.steps_taken_on_last_policy[i] = 0

                    # Switch from teacher to student with probability 1-epsilon
                    elif (
                        self.last_used_policy[i] == "teacher"
                        and sampled_epsilon < 1 - cur_epsilon
                    ):
                        list_of_policies.append("student")
                        self.last_used_policy[i] = "student"
                        self.steps_taken_on_last_policy[i] = 0

                    # Keep current policy
                    else:
                        list_of_policies.append(self.last_used_policy[i])
                else:
                    # Still in trust period - keep current policy
                    list_of_policies.append(self.last_used_policy[i])

                self.steps_taken_on_last_policy[i] += 1

        # Update statistics
        self.update_statistics(list_of_policies)
        return list_of_policies

    def get_statistics(self) -> dict:
        """Get epsilon scheduling statistics.

        Returns:
            Dictionary with scheduling statistics including current epsilon
        """
        stats = super().get_statistics()
        stats["current_epsilon"] = self.current_epsilon
        return stats
