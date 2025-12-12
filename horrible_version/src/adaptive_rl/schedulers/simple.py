"""Simple scheduling strategies for teacher-guided RL.

Includes baseline strategies like always teacher, always student,
and alternating between them.
"""

import torch

from .base import PolicyScheduler


class StudentOnlyScheduler(PolicyScheduler):
    """Always use the student policy (standard RL without teacher).

    This serves as a baseline to compare against teacher-guided strategies.
    """

    def __init__(self, student_policy=None, teacher_policy=None, num_envs: int = 1, **kwargs):
        super().__init__(student_policy, teacher_policy, num_envs, **kwargs)

    def choose_policy_type(
        self,
        iteration: int,
        global_step: int,
        steps_since_reset: torch.Tensor,
        prev_reward: torch.Tensor,
    ) -> list[str]:
        """Always return student policy."""
        policies = ["student"] * self.num_envs
        self.update_statistics(policies)
        return policies


class TeacherOnlyScheduler(PolicyScheduler):
    """Always use the teacher policy (pure imitation).

    This serves as an upper bound baseline when the teacher is optimal.
    """

    def __init__(self, student_policy=None, teacher_policy=None, num_envs: int = 1, **kwargs):
        super().__init__(student_policy, teacher_policy, num_envs, **kwargs)

    def choose_policy_type(
        self,
        iteration: int,
        global_step: int,
        steps_since_reset: torch.Tensor,
        prev_reward: torch.Tensor,
    ) -> list[str]:
        """Always return teacher policy."""
        policies = ["teacher"] * self.num_envs
        self.update_statistics(policies)
        return policies


class AlternatingScheduler(PolicyScheduler):
    """Alternate between teacher and student policies every iteration.

    Simple strategy that gives equal time to both policies.
    """

    def __init__(self, student_policy=None, teacher_policy=None, num_envs: int = 1, **kwargs):
        super().__init__(student_policy, teacher_policy, num_envs, **kwargs)

    def choose_policy_type(
        self,
        iteration: int,
        global_step: int,
        steps_since_reset: torch.Tensor,
        prev_reward: torch.Tensor,
    ) -> list[str]:
        """Alternate based on iteration number."""
        if iteration % 2 == 0:
            policies = ["student"] * self.num_envs
        else:
            policies = ["teacher"] * self.num_envs

        self.update_statistics(policies)
        return policies


class TeacherThenStudentScheduler(PolicyScheduler):
    """Use teacher for initial steps/iterations, then switch to student.

    Useful for bootstrapping the student with initial teacher demonstrations.
    """

    def __init__(
        self,
        student_policy=None,
        teacher_policy=None,
        num_envs: int = 1,
        iteration_to_switch: int = 50,
        step_to_switch: int = 15,
        **kwargs,
    ):
        """Initialize teacher-then-student scheduler.

        Args:
            student_policy: Student policy (optional)
            teacher_policy: Teacher policy (optional)
            num_envs: Number of parallel environments
            iteration_to_switch: Iteration after which to use only student
            step_to_switch: Steps within episode to use teacher before switching
        """
        super().__init__(student_policy, teacher_policy, num_envs, **kwargs)
        self.iteration_to_switch = iteration_to_switch
        self.step_to_switch = step_to_switch

    def choose_policy_type(
        self,
        iteration: int,
        global_step: int,
        steps_since_reset: torch.Tensor,
        prev_reward: torch.Tensor,
    ) -> list[str]:
        """Use teacher initially, then switch to student.

        Args:
            iteration: Current training iteration
            global_step: Total environment steps
            steps_since_reset: Steps since env reset
            prev_reward: Previous step reward

        Returns:
            List of policy types for each environment
        """
        if iteration < self.iteration_to_switch:
            # Before switch iteration: use teacher for first K steps of each episode
            list_of_policies = [
                "teacher" if steps < self.step_to_switch else "student"
                for steps in steps_since_reset
            ]
        else:
            # After switch iteration: always use student
            list_of_policies = ["student"] * self.num_envs

        self.update_statistics(list_of_policies)
        return list_of_policies
