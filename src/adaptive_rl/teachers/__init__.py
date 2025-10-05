"""Teacher policies for guided reinforcement learning.

Available teachers:
- RandomTeacher: Random actions (baseline)
- CartPoleOptimalTeacher: Hand-coded optimal policy for CartPole
- LunarLanderOptimalTeacher: Heuristic policy for LunarLander
- PretrainedPPOTeacher: Load a pretrained PPO model
"""

from .base import TeacherPolicy
from .optimal import (
    OPTIMAL_TEACHERS,
    CartPoleOptimalTeacher,
    LunarLanderOptimalTeacher,
    create_optimal_teacher,
)
from .pretrained import PretrainedPPOTeacher
from .random import RandomTeacher

# Registry of all teacher types
TEACHER_TYPES = {
    "random": RandomTeacher,
    "optimal": create_optimal_teacher,  # Factory function
    "pretrained": PretrainedPPOTeacher,
}


def create_teacher(
    teacher_type: str,
    env_id: str = None,
    action_space=None,
    observation_space=None,
    device="cpu",
    **kwargs,
) -> TeacherPolicy:
    """Factory function to create a teacher by type.

    Args:
        teacher_type: Type of teacher ("random", "optimal", "pretrained")
        env_id: Environment ID (needed for optimal teachers)
        action_space: Gym action space
        observation_space: Gym observation space
        device: Device to run on
        **kwargs: Additional teacher-specific arguments

    Returns:
        Teacher policy instance

    Raises:
        ValueError: If teacher type is not recognized
    """
    if teacher_type not in TEACHER_TYPES:
        raise ValueError(
            f"Unknown teacher type: {teacher_type}. "
            f"Available types: {list(TEACHER_TYPES.keys())}"
        )

    if teacher_type == "optimal":
        if env_id is None:
            raise ValueError("env_id must be specified for optimal teachers")
        return create_optimal_teacher(
            env_id,
            action_space=action_space,
            observation_space=observation_space,
            device=device,
            **kwargs,
        )
    teacher_class = TEACHER_TYPES[teacher_type]
    return teacher_class(
        action_space=action_space,
        observation_space=observation_space,
        device=device,
        **kwargs,
    )


__all__ = [
    "TEACHER_TYPES",
    "CartPoleOptimalTeacher",
    "LunarLanderOptimalTeacher",
    "PretrainedPPOTeacher",
    "RandomTeacher",
    "TeacherPolicy",
    "create_optimal_teacher",
    "create_teacher",
]
