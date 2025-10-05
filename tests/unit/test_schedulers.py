"""Unit tests for scheduling strategies."""

import pytest
import torch

from adaptive_rl.schedulers import (
    AlternatingScheduler,
    EpsilonDecreasingScheduler,
    EpsilonScheduler,
    RewardBasedScheduler,
    TeacherOnlyScheduler,
    TeacherThenStudentScheduler,
    create_scheduler,
)


class TestRewardBasedScheduler:
    """Test reward-based scheduling strategy (main contribution)."""

    def test_initialization(self, num_envs, trust_length, device):
        """Test scheduler initialization."""
        scheduler = RewardBasedScheduler(
            num_envs=num_envs, trust_length=trust_length, device=device
        )

        assert scheduler.num_envs == num_envs
        assert scheduler.trust_length == trust_length
        assert scheduler.device == device
        assert len(scheduler.last_used_policy) == num_envs
        assert len(scheduler.steps_taken_on_last_policy) == num_envs
        assert len(scheduler.prev_prev_reward) == num_envs

    def test_reset_behavior(
        self, reward_based_scheduler, reset_rewards, steps_since_reset
    ):
        """Test that reset rewards (-1) always choose teacher."""
        policies = reward_based_scheduler.choose_policy_type(
            iteration=0,
            global_step=100,
            steps_since_reset=steps_since_reset,
            prev_reward=reset_rewards,
        )

        assert all(policy == "teacher" for policy in policies)
        # After reset, step counter should be 1 (reset to 0 then incremented)
        assert all(
            steps == 1 for steps in reward_based_scheduler.steps_taken_on_last_policy
        )

    def test_trust_period_enforcement(self, reward_based_scheduler, sample_rewards):
        """Test that trust period is enforced before switching."""
        num_envs = reward_based_scheduler.num_envs
        trust_length = reward_based_scheduler.trust_length

        # Set steps below trust length
        reward_based_scheduler.steps_taken_on_last_policy = [
            trust_length - 1
        ] * num_envs
        reward_based_scheduler.prev_prev_reward = [
            1.0
        ] * num_envs  # Higher than current

        # Create rewards lower than previous (should trigger switch if trust period expired)
        low_rewards = torch.zeros(num_envs)

        policies = reward_based_scheduler.choose_policy_type(
            iteration=1,
            global_step=100,
            steps_since_reset=torch.ones(num_envs) * 10,
            prev_reward=low_rewards,
        )

        # Should NOT switch because trust period hasn't expired
        assert all(policy == "teacher" for policy in policies)  # Default start policy

    def test_reward_based_switching(self, reward_based_scheduler):
        """Test the core reward-based switching logic."""
        num_envs = reward_based_scheduler.num_envs
        trust_length = reward_based_scheduler.trust_length

        # Set up scenario: trust period expired, reward decreased
        reward_based_scheduler.steps_taken_on_last_policy = [trust_length] * num_envs
        reward_based_scheduler.last_used_policy = ["teacher"] * num_envs
        reward_based_scheduler.prev_prev_reward = [1.0] * num_envs  # Previous reward

        # Current reward is lower - should trigger switch
        current_rewards = torch.zeros(num_envs)  # Lower than 1.0

        policies = reward_based_scheduler.choose_policy_type(
            iteration=1,
            global_step=100,
            steps_since_reset=torch.ones(num_envs) * 10,
            prev_reward=current_rewards,
        )

        # Should switch to student
        assert all(policy == "student" for policy in policies)
        assert all(
            policy == "student" for policy in reward_based_scheduler.last_used_policy
        )
        # After switching, step counter should be 1 (reset to 0 then incremented)
        assert all(
            steps == 1 for steps in reward_based_scheduler.steps_taken_on_last_policy
        )

    def test_no_switch_on_improvement(self, reward_based_scheduler):
        """Test that no switch occurs when reward improves."""
        num_envs = reward_based_scheduler.num_envs
        trust_length = reward_based_scheduler.trust_length

        # Set up scenario: trust period expired, reward improved
        reward_based_scheduler.steps_taken_on_last_policy = [trust_length] * num_envs
        reward_based_scheduler.last_used_policy = ["teacher"] * num_envs
        reward_based_scheduler.prev_prev_reward = [0.5] * num_envs  # Previous reward

        # Current reward is higher - should NOT trigger switch
        current_rewards = torch.ones(num_envs)  # Higher than 0.5

        policies = reward_based_scheduler.choose_policy_type(
            iteration=1,
            global_step=100,
            steps_since_reset=torch.ones(num_envs) * 10,
            prev_reward=current_rewards,
        )

        # Should keep teacher
        assert all(policy == "teacher" for policy in policies)

    def test_statistics_tracking(self, reward_based_scheduler):
        """Test that statistics are tracked correctly."""
        initial_stats = reward_based_scheduler.get_statistics()
        assert initial_stats["reward_triggered_switches"] == 0
        assert initial_stats["performance_improvements"] == 0
        assert initial_stats["performance_degradations"] == 0

    def test_alternating_switches(self, reward_based_scheduler):
        """Test that scheduler can switch back and forth."""
        # Start with teacher, switch to student
        reward_based_scheduler.steps_taken_on_last_policy = [5] * 4
        reward_based_scheduler.last_used_policy = ["teacher"] * 4
        reward_based_scheduler.prev_prev_reward = [1.0] * 4

        # First switch: teacher -> student (reward decreased)
        policies1 = reward_based_scheduler.choose_policy_type(
            iteration=1,
            global_step=100,
            steps_since_reset=torch.ones(4) * 10,
            prev_reward=torch.zeros(4),
        )
        assert all(p == "student" for p in policies1)

        # Second switch: student -> teacher (reward decreased again)
        reward_based_scheduler.steps_taken_on_last_policy = [5] * 4
        reward_based_scheduler.prev_prev_reward = [1.0] * 4

        policies2 = reward_based_scheduler.choose_policy_type(
            iteration=2,
            global_step=200,
            steps_since_reset=torch.ones(4) * 10,
            prev_reward=torch.zeros(4),
        )
        assert all(p == "teacher" for p in policies2)


class TestEpsilonScheduler:
    """Test epsilon-based scheduling strategies."""

    def test_epsilon_scheduler_initialization(self, num_envs, device):
        """Test epsilon scheduler initialization."""
        epsilon = 0.3
        scheduler = EpsilonScheduler(
            num_envs=num_envs, epsilon=epsilon, trust_length=5, device=device
        )

        assert scheduler.epsilon == epsilon
        assert scheduler.num_envs == num_envs

    def test_reset_behavior_epsilon(
        self, epsilon_scheduler, reset_rewards, steps_since_reset
    ):
        """Test epsilon scheduler with reset rewards."""
        policies = epsilon_scheduler.choose_policy_type(
            iteration=0,
            global_step=100,
            steps_since_reset=steps_since_reset,
            prev_reward=reset_rewards,
        )

        # Should always choose teacher on reset
        assert all(policy == "teacher" for policy in policies)

    def test_trust_period_epsilon(self, epsilon_scheduler, sample_rewards):
        """Test that epsilon scheduler respects trust period."""
        # Set steps below trust length
        epsilon_scheduler.steps_taken_on_last_policy = [
            2,
            2,
            2,
            2,
        ]  # Below trust_length=5

        policies = epsilon_scheduler.choose_policy_type(
            iteration=1,
            global_step=100,
            steps_since_reset=torch.ones(4) * 10,
            prev_reward=sample_rewards,
        )

        # Should keep current policy (teacher by default)
        assert all(policy == "teacher" for policy in policies)


class TestEpsilonDecreasingScheduler:
    """Test epsilon decreasing scheduler."""

    def test_epsilon_calculation(self, num_envs, device):
        """Test epsilon calculation over time."""
        scheduler = EpsilonDecreasingScheduler(
            num_envs=num_envs,
            trust_length=5,
            decrease_until_global_step=1000,
            device=device,
        )

        # At step 0, epsilon should be 1.0
        epsilon_0 = scheduler.calculate_epsilon(0)
        assert epsilon_0 == 1.0

        # At step 500, epsilon should be 0.5
        epsilon_500 = scheduler.calculate_epsilon(500)
        assert abs(epsilon_500 - 0.5) < 1e-6

        # At step 1000+, epsilon should be 0.0
        epsilon_1000 = scheduler.calculate_epsilon(1000)
        assert epsilon_1000 == 0.0

    def test_final_phase_student_only(self, num_envs, device):
        """Test that after decay period, only student is used."""
        scheduler = EpsilonDecreasingScheduler(
            num_envs=num_envs,
            trust_length=5,
            decrease_until_global_step=100,
            device=device,
        )

        policies = scheduler.choose_policy_type(
            iteration=50,
            global_step=200,  # Beyond decrease_until_global_step
            steps_since_reset=torch.ones(num_envs) * 10,
            prev_reward=torch.ones(num_envs),
        )

        assert all(policy == "student" for policy in policies)


class TestSimpleSchedulers:
    """Test simple baseline schedulers."""

    def test_student_only_scheduler(
        self, student_only_scheduler, sample_rewards, steps_since_reset
    ):
        """Test student-only scheduler always returns student."""
        policies = student_only_scheduler.choose_policy_type(
            iteration=5,
            global_step=100,
            steps_since_reset=steps_since_reset,
            prev_reward=sample_rewards,
        )

        assert all(policy == "student" for policy in policies)

    def test_teacher_only_scheduler(self, num_envs, device):
        """Test teacher-only scheduler always returns teacher."""
        scheduler = TeacherOnlyScheduler(num_envs=num_envs, device=device)

        policies = scheduler.choose_policy_type(
            iteration=5,
            global_step=100,
            steps_since_reset=torch.ones(num_envs) * 5,
            prev_reward=torch.ones(num_envs),
        )

        assert all(policy == "teacher" for policy in policies)

    def test_alternating_scheduler(self, num_envs, device):
        """Test alternating scheduler switches every iteration."""
        scheduler = AlternatingScheduler(num_envs=num_envs, device=device)

        # Even iteration -> student
        policies_even = scheduler.choose_policy_type(
            iteration=2,
            global_step=100,
            steps_since_reset=torch.ones(num_envs) * 5,
            prev_reward=torch.ones(num_envs),
        )
        assert all(policy == "student" for policy in policies_even)

        # Odd iteration -> teacher
        policies_odd = scheduler.choose_policy_type(
            iteration=3,
            global_step=150,
            steps_since_reset=torch.ones(num_envs) * 5,
            prev_reward=torch.ones(num_envs),
        )
        assert all(policy == "teacher" for policy in policies_odd)

    def test_teacher_then_student_scheduler(self, num_envs, device):
        """Test teacher-then-student scheduler."""
        scheduler = TeacherThenStudentScheduler(
            num_envs=num_envs, iteration_to_switch=10, step_to_switch=5, device=device
        )

        # Before iteration_to_switch, use teacher for first steps
        policies_early = scheduler.choose_policy_type(
            iteration=5,  # < iteration_to_switch
            global_step=100,
            steps_since_reset=torch.tensor([3, 7, 2, 8]),  # Mixed steps
            prev_reward=torch.ones(num_envs),
        )
        expected = [
            "teacher",
            "student",
            "teacher",
            "student",
        ]  # Based on step_to_switch=5
        assert policies_early == expected

        # After iteration_to_switch, always use student
        policies_late = scheduler.choose_policy_type(
            iteration=15,  # > iteration_to_switch
            global_step=300,
            steps_since_reset=torch.tensor([3, 7, 2, 8]),
            prev_reward=torch.ones(num_envs),
        )
        assert all(policy == "student" for policy in policies_late)


class TestSchedulerFactory:
    """Test scheduler factory function."""

    def test_create_scheduler_reward_based(self, num_envs, device):
        """Test creating reward-based scheduler via factory."""
        scheduler = create_scheduler(
            strategy="reward_based", num_envs=num_envs, trust_length=3, device=device
        )

        assert isinstance(scheduler, RewardBasedScheduler)
        assert scheduler.trust_length == 3

    def test_create_scheduler_epsilon(self, num_envs, device):
        """Test creating epsilon scheduler via factory."""
        scheduler = create_scheduler(
            strategy="epsilon", num_envs=num_envs, epsilon=0.4, device=device
        )

        assert isinstance(scheduler, EpsilonScheduler)
        assert scheduler.epsilon == 0.4

    def test_create_scheduler_legacy_names(self, num_envs, device):
        """Test that legacy names still work."""
        # Test legacy name mapping
        scheduler = create_scheduler(
            strategy="octo_reward_based",  # Legacy name
            num_envs=num_envs,
            device=device,
        )

        assert isinstance(scheduler, RewardBasedScheduler)

    def test_create_scheduler_invalid_strategy(self, num_envs, device):
        """Test error handling for invalid strategy."""
        with pytest.raises(ValueError, match="Unknown scheduling strategy"):
            create_scheduler(
                strategy="invalid_strategy", num_envs=num_envs, device=device
            )


class TestSchedulerStatistics:
    """Test scheduler statistics tracking."""

    def test_statistics_initialization(self, reward_based_scheduler):
        """Test that statistics start at zero."""
        stats = reward_based_scheduler.get_statistics()

        assert stats["teacher_usage_ratio"] == 0.0
        assert stats["student_usage_ratio"] == 0.0
        assert stats["switch_frequency"] == 0.0
        assert stats["total_steps"] == 0
        assert stats["total_switches"] == 0

    def test_statistics_update(self, reward_based_scheduler):
        """Test that statistics are updated correctly."""
        # Simulate some policy choices
        policies = ["teacher", "teacher", "student", "student"]
        reward_based_scheduler.update_statistics(policies)

        stats = reward_based_scheduler.get_statistics()
        assert stats["teacher_usage_ratio"] == 0.5
        assert stats["student_usage_ratio"] == 0.5
        assert stats["total_steps"] == 4

    def test_reset_functionality(self, reward_based_scheduler):
        """Test scheduler reset functionality."""
        # Make some changes
        reward_based_scheduler.steps_taken_on_last_policy = [10] * 4
        reward_based_scheduler.teacher_usage_count = 5

        # Reset
        reward_based_scheduler.reset()

        # Check everything is back to initial state
        assert all(
            steps == 0 for steps in reward_based_scheduler.steps_taken_on_last_policy
        )
        assert reward_based_scheduler.teacher_usage_count == 0
        assert all(reward == -1.0 for reward in reward_based_scheduler.prev_prev_reward)
