"""Reward-based scheduling strategy for teacher-guided RL.

MAIN CONTRIBUTION: Adaptive switching between teacher and student policies
based on reward trends. If performance decreases, the strategy switches to
the alternative policy after a trust period.
"""

import torch

from src.adaptive_rl.schedulers.base import PolicyScheduler


class RewardBasedScheduler(PolicyScheduler):
    """Switches between teacher and student policies based on reward trends.

    This is the main contribution of the research: an adaptive scheduling
    strategy that monitors reward progression and switches policies when
    performance degrades.

    Algorithm:
    1. Give each policy a "trust period" of K steps
    2. After trust period, compare current reward with previous reward
    3. If reward decreased, switch to alternative policy
    4. If reward increased or stayed same, keep current policy
    5. Reset trust period after each switch
    """

    def __init__(
        self,
        student_policy=None,
        teacher_policy=None,
        num_envs: int = 1,
        trust_period: int = 5,
        policy_trust_threshold: float = 0.6,
        internal_policy_warmup_length: int = 5,
        initial_policy: str = "teacher",
        device: str = "cpu",
        log_dir: str = None,
        **kwargs,
    ):
        """Initialize reward-based scheduler.

        Args:
            student_policy: Student policy (can be None)
            teacher_policy: Teacher policy (can be None)
            num_envs: Number of parallel environments
            trust_period: Number of steps before evaluating policy performance
            policy_trust_threshold: Not used in current implementation but kept for compatibility
            internal_policy_warmup_length: Initial warmup period for student policy
            initial_policy: Which policy to start with ("teacher" or "student")
            device: Device to run on
            log_dir: Directory for logging
        """
        # Handle trust_length parameter (backwards compatibility)
        trust_length = kwargs.pop('trust_length', trust_period)

        super().__init__(
            student_policy=student_policy,
            teacher_policy=teacher_policy,
            num_envs=num_envs,
            trust_length=trust_length,
            device=device,
            log_dir=log_dir,
            **kwargs,
        )

        self.policy_trust_threshold = policy_trust_threshold
        self.internal_policy_warmup_length = internal_policy_warmup_length
        self.initial_policy = initial_policy

        # Track reward history for comparison
        self.prev_prev_reward = [-1.0] * num_envs

        # Statistics specific to reward-based scheduling
        self.reward_triggered_switches = 0
        self.performance_improvements = 0
        self.performance_degradations = 0

    def choose_policy_type(
        self,
        iteration: int,
        global_step: int,
        steps_since_reset: torch.Tensor,
        prev_reward: torch.Tensor,
    ) -> list[str]:
        """Choose policy based on reward trends.

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
                self.last_used_policy[i] = "teacher"
                self.steps_taken_on_last_policy[i] = 0
                self.prev_prev_reward[i] = -1.0

            # Check if we should evaluate switching
            elif self.steps_taken_on_last_policy[i] >= self.trust_length:
                # CORE LOGIC: Switch if reward decreased
                if prev_reward[i] < self.prev_prev_reward[i]:
                    # Switch to alternative policy
                    new_policy = (
                        "student"
                        if self.last_used_policy[i] == "teacher"
                        else "teacher"
                    )
                    list_of_policies.append(new_policy)

                    # Update tracking
                    self.last_used_policy[i] = new_policy
                    self.steps_taken_on_last_policy[i] = 0
                    self.reward_triggered_switches += 1
                    self.performance_degradations += 1
                else:
                    # Keep current policy (reward improved or stayed same)
                    list_of_policies.append(self.last_used_policy[i])
                    if prev_reward[i] > self.prev_prev_reward[i]:
                        self.performance_improvements += 1
            else:
                # Still in trust period - keep current policy
                list_of_policies.append(self.last_used_policy[i])

            # Update reward history and step counter
            self.prev_prev_reward[i] = (
                prev_reward[i].item()
                if torch.is_tensor(prev_reward[i])
                else prev_reward[i]
            )
            self.steps_taken_on_last_policy[i] += 1

        # Update general statistics
        self.update_statistics(list_of_policies)

        return list_of_policies

    def get_statistics(self) -> dict:
        """Get reward-based scheduling statistics.

        Returns:
            Dictionary with scheduling statistics
        """
        stats = super().get_statistics()

        # Add reward-based specific stats
        stats.update(
            {
                "reward_triggered_switches": self.reward_triggered_switches,
                "performance_improvements": self.performance_improvements,
                "performance_degradations": self.performance_degradations,
                "avg_steps_before_switch": self.total_steps / max(1, self.switch_count),
            }
        )

        return stats

    def reset(self):
        """Reset scheduler state for new training run."""
        super().reset()
        self.prev_prev_reward = [-1.0] * self.num_envs
        self.reward_triggered_switches = 0
        self.performance_improvements = 0
        self.performance_degradations = 0
