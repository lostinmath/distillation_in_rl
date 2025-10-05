"""Integration tests for PPO + Scheduler interaction."""

from unittest.mock import Mock, patch

import gymnasium as gym
import pytest
import torch

from adaptive_rl.core.ppo import PPOTrainer
from adaptive_rl.core.config import (
    DistillationConfig, EnvironmentConfig, ExperimentConfig,
    SchedulerConfig, TrainingConfig
)
from adaptive_rl.schedulers import EpsilonScheduler, RewardBasedScheduler


@pytest.mark.integration
class TestPPOSchedulerIntegration:
    """Test PPO agent with scheduling strategies."""

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_ppo_with_reward_based_scheduler(
        self, mock_logger_class, mock_make_vec_env
    ):
        """Test PPO trainer with reward-based scheduler."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_env.num_envs = 2
        mock_env.reset.return_value = (torch.randn(2, 4), {})
        mock_env.step.return_value = (
            torch.randn(2, 4),
            torch.randn(2, 1),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            [{}] * 2,
        )
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        # Create config
        config = DistillationConfig(
            environment=EnvironmentConfig(env_id="CartPole-v1", num_envs=2, num_steps=5),
            training=TrainingConfig(num_iterations=3),
            scheduler=SchedulerConfig(strategy="reward_based", trust_length=3),
            experiment=ExperimentConfig(device="cpu", name="test_integration")
        )

        # Create trainer with scheduler
        trainer = PPOTrainer(config)

        # Verify scheduler was created
        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, RewardBasedScheduler)
        assert trainer.scheduler.trust_length == 3

        # Test that scheduler is called during rollout collection
        # The actual rollout collection would call the scheduler
        policies = trainer.scheduler.choose_policy_type(
            iteration=0,
            global_step=0,
            steps_since_reset=torch.ones(2) * 5,
            prev_reward=torch.ones(2) * 0.5,
        )

        assert len(policies) == 2
        assert all(policy in ["teacher", "student"] for policy in policies)

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_ppo_with_epsilon_scheduler(self, mock_logger_class, mock_make_vec_env):
        """Test PPO trainer with epsilon scheduler."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_env.num_envs = 2
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        # Create config
        config = DistillationConfig(
            environment=EnvironmentConfig(env_id="CartPole-v1"),
            scheduler=SchedulerConfig(strategy="epsilon", epsilon=0.3, trust_length=5),
            experiment=ExperimentConfig(device="cpu", name="test_epsilon")
        )

        # Create trainer with scheduler
        trainer = PPOTrainer(config)

        # Verify scheduler was created
        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, EpsilonScheduler)
        assert trainer.scheduler.epsilon == 0.3

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_scheduler_statistics_collection(
        self, mock_logger_class, mock_make_vec_env
    ):
        """Test that scheduler statistics are properly collected."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_env.num_envs = 4
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        # Create config
        config = DistillationConfig(
            environment=EnvironmentConfig(env_id="CartPole-v1", num_envs=4),
            scheduler=SchedulerConfig(strategy="reward_based", trust_length=2),
            experiment=ExperimentConfig(device="cpu", name="test_stats")
        )

        # Create trainer with reward-based scheduler
        trainer = PPOTrainer(config)

        # Simulate policy choices
        policies = ["teacher", "teacher", "student", "student"]
        trainer.scheduler.update_statistics(policies)

        # Check statistics
        stats = trainer.scheduler.get_statistics()
        assert stats["teacher_usage_ratio"] == 0.5
        assert stats["student_usage_ratio"] == 0.5
        assert stats["total_steps"] == 4

    def test_scheduler_reward_based_switching_logic(self):
        """Test the core reward-based switching logic in isolation."""
        scheduler = RewardBasedScheduler(num_envs=2, trust_length=3, device="cpu")

        # Set up state for switching test
        scheduler.steps_taken_on_last_policy = [3, 3]  # Trust period expired
        scheduler.last_used_policy = ["teacher", "teacher"]
        scheduler.prev_prev_reward = [1.0, 1.0]  # Previous rewards

        # Current rewards are lower - should trigger switch
        current_rewards = torch.tensor([0.5, 0.5])
        policies = scheduler.choose_policy_type(
            iteration=1,
            global_step=100,
            steps_since_reset=torch.ones(2) * 10,
            prev_reward=current_rewards,
        )

        # Should switch to student
        assert all(policy == "student" for policy in policies)
        assert all(policy == "student" for policy in scheduler.last_used_policy)
        assert all(steps == 0 for steps in scheduler.steps_taken_on_last_policy)

    def test_scheduler_epsilon_probabilistic_behavior(self):
        """Test epsilon scheduler probabilistic behavior."""
        scheduler = EpsilonScheduler(
            num_envs=100,  # Large number for statistical test
            epsilon=0.5,  # 50% probability
            trust_length=1,
            device="cpu",
        )

        # Set up state where switching is allowed
        scheduler.steps_taken_on_last_policy = [2] * 100  # Above trust length
        scheduler.last_used_policy = ["teacher"] * 100

        policies = scheduler.choose_policy_type(
            iteration=1,
            global_step=100,
            steps_since_reset=torch.ones(100) * 10,
            prev_reward=torch.ones(100),
        )

        # With epsilon=0.5 and 100 environments, we should get roughly 50/50 split
        teacher_count = sum(1 for p in policies if p == "teacher")
        student_count = sum(1 for p in policies if p == "student")

        # Allow some variance (should be close to 50/50)
        assert 30 <= teacher_count <= 70
        assert 30 <= student_count <= 70
