"""Policy scheduling strategies for teacher-guided reinforcement learning.

Available schedulers:
- RewardBasedScheduler: Switches based on reward trends (main contribution)
- EpsilonScheduler: Fixed probability switching
- EpsilonDecreasingScheduler: Linearly decreasing probability
- StudentOnlyScheduler: Baseline - no teacher
- TeacherOnlyScheduler: Baseline - pure imitation
- AlternatingScheduler: Alternate every iteration
- TeacherThenStudentScheduler: Bootstrap with teacher then switch
"""

from .base import PolicyScheduler
from .epsilon import EpsilonDecreasingScheduler, EpsilonScheduler
from .reward_based import RewardBasedScheduler
from .simple import (
    AlternatingScheduler,
    StudentOnlyScheduler,
    TeacherOnlyScheduler,
    TeacherThenStudentScheduler,
)

# Registry of all available schedulers
SCHEDULERS = {
    "reward_based": RewardBasedScheduler,
    "epsilon": EpsilonScheduler,
    "epsilon_decreasing": EpsilonDecreasingScheduler,
    "student_only": StudentOnlyScheduler,
    "teacher_only": TeacherOnlyScheduler,
    "alternating": AlternatingScheduler,
    "teacher_then_student": TeacherThenStudentScheduler,
    # Legacy names for compatibility
    "internal_only": StudentOnlyScheduler,
    "octo_only": TeacherOnlyScheduler,
    "octo_epsilon": EpsilonScheduler,
    "octo_epsilon_decreasing": EpsilonDecreasingScheduler,
    "internal_octo_interchangeably": AlternatingScheduler,
    "octo_than_internal": TeacherThenStudentScheduler,
    "octo_reward_based": RewardBasedScheduler,
}


def create_scheduler(strategy: str, **kwargs) -> PolicyScheduler:
    """Factory function to create a scheduler by name.

    Args:
        strategy: Name of the scheduling strategy
        **kwargs: Arguments to pass to the scheduler

    Returns:
        Initialized scheduler instance

    Raises:
        ValueError: If strategy name is not recognized
    """
    if strategy not in SCHEDULERS:
        raise ValueError(
            f"Unknown scheduling strategy: {strategy}. "
            f"Available strategies: {list(SCHEDULERS.keys())}"
        )

    scheduler_class = SCHEDULERS[strategy]
    return scheduler_class(**kwargs)


__all__ = [
    "SCHEDULERS",
    "AlternatingScheduler",
    "EpsilonDecreasingScheduler",
    "EpsilonScheduler",
    "PolicyScheduler",
    "RewardBasedScheduler",
    "StudentOnlyScheduler",
    "TeacherOnlyScheduler",
    "TeacherThenStudentScheduler",
    "create_scheduler",
]
