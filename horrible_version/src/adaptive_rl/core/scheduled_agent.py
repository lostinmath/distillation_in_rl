"""Algorithm-agnostic scheduled agent wrapper.

This wrapper intercepts actions from any RL algorithm and applies teacher-student
scheduling without modifying the underlying algorithm implementation.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import numpy as np
from abc import ABC, abstractmethod

from src.adaptive_rl.schedulers.base import PolicyScheduler
from src.adaptive_rl.teachers.base import TeacherPolicy


class BaseAgent(ABC):
    """Abstract base for any RL agent."""

    @abstractmethod
    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action and value from agent."""
        pass

    @abstractmethod
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate from agent."""
        pass


class CleanRLAgentWrapper(BaseAgent):
    """Wrapper for CleanRL agents to match BaseAgent interface."""

    def __init__(self, cleanrl_agent):
        self.agent = cleanrl_agent

    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.agent.get_action_and_value(obs, action)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.agent.get_value(obs)


class ScheduledAgent:
    """Algorithm-agnostic scheduled agent.

    Wraps any RL agent and applies teacher-student scheduling without
    modifying the underlying algorithm. This enables clean integration
    with existing PPO/SAC/TD3 implementations.

    Key Design:
    - Agent wrapper pattern: zero modification to base algorithms
    - Action interception: schedule at environment interface
    - Metrics tracking: comprehensive scheduling analytics
    - Algorithm agnostic: works with any policy representation
    """

    def __init__(
        self,
        student_agent: BaseAgent,
        teacher_policy: Optional[TeacherPolicy],
        scheduler: PolicyScheduler,
        device: torch.device = torch.device("cpu"),
        track_metrics: bool = True
    ):
        self.student = student_agent
        self.teacher = teacher_policy
        self.scheduler = scheduler
        self.device = device
        self.track_metrics = track_metrics

        # Metrics tracking
        self.reset_metrics()

    def reset_metrics(self):
        """Reset scheduling metrics."""
        self.metrics = {
            "teacher_actions": 0,
            "student_actions": 0,
            "policy_switches": 0,
            "last_policy": None,
            "episode_teacher_ratio": [],
            "scheduling_decisions": []
        }

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        iteration: int = 0,
        global_step: int = 0,
        steps_since_reset: Optional[torch.Tensor] = None,
        prev_reward: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Get scheduled action and value.

        Returns:
            action: Selected action (teacher or student)
            logprob: Student's log probability (for PPO loss)
            entropy: Student's entropy (for PPO loss)
            value: Student's value estimate (for PPO loss)
            scheduling_info: Metrics and policy selection info
        """
        batch_size = obs.shape[0]

        # Always get student's action and value for training
        student_action, student_logprob, student_entropy, student_value = self.student.get_action_and_value(obs, action)

        # If no teacher or scheduler, return student action
        if self.teacher is None:
            return student_action, student_logprob, student_entropy, student_value, {"policy": "student"}

        # Get teacher actions
        teacher_actions = self.teacher.act(obs.cpu().numpy())
        if isinstance(teacher_actions, np.ndarray):
            teacher_actions = torch.tensor(teacher_actions).to(self.device)
        else:
            teacher_actions = teacher_actions.to(self.device)

        # Use existing scheduler interface
        if steps_since_reset is None:
            steps_since_reset = torch.zeros(batch_size)
        if prev_reward is None:
            prev_reward = torch.zeros(batch_size)

        # Get policy decisions from scheduler
        policy_choices = self.scheduler.choose_policy_type(
            iteration=iteration,
            global_step=global_step,
            steps_since_reset=steps_since_reset,
            prev_reward=prev_reward
        )

        # Ensure policy_choices matches batch_size
        if len(policy_choices) != batch_size:
            print(f"Warning: policy_choices length {len(policy_choices)} != batch_size {batch_size}")
            # Extend or truncate to match batch_size
            if len(policy_choices) < batch_size:
                policy_choices = policy_choices + [policy_choices[-1]] * (batch_size - len(policy_choices))
            else:
                policy_choices = policy_choices[:batch_size]

        # Apply scheduling decisions
        final_actions = student_action.clone()
        scheduling_info = {"policies": policy_choices}

        for env_idx in range(batch_size):
            if policy_choices[env_idx] == "teacher":
                final_actions[env_idx] = teacher_actions[env_idx]
                if self.track_metrics:
                    self.metrics["teacher_actions"] += 1
            else:
                if self.track_metrics:
                    self.metrics["student_actions"] += 1

        if self.track_metrics:
            teacher_ratio = self.metrics["teacher_actions"] / max(1, self.metrics["teacher_actions"] + self.metrics["student_actions"])
            scheduling_info["teacher_ratio"] = teacher_ratio

        # CRITICAL: Return student's logprob/entropy/value for PPO training
        return final_actions, student_logprob, student_entropy, student_value, scheduling_info

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate (always from student for consistency)."""
        return self.student.get_value(obs)

    def get_scheduling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scheduling metrics."""
        if not self.track_metrics:
            return {}

        total_actions = self.metrics["teacher_actions"] + self.metrics["student_actions"]
        if total_actions == 0:
            return {"teacher_ratio": 0.0, "student_ratio": 1.0, "policy_switches": 0}

        return {
            "teacher_ratio": self.metrics["teacher_actions"] / total_actions,
            "student_ratio": self.metrics["student_actions"] / total_actions,
            "policy_switches": self.metrics["policy_switches"],
            "total_actions": total_actions
        }