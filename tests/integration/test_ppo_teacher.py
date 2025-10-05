"""Integration tests for PPO + Teacher interaction."""

from unittest.mock import Mock, patch

import gymnasium as gym
import pytest
import torch

from adaptive_rl.core.ppo import PPOTrainer
from adaptive_rl.core.config import (
    DistillationConfig, EnvironmentConfig, ExperimentConfig, TeacherConfig, SchedulerConfig
)
from adaptive_rl.teachers import CartPoleOptimalTeacher, RandomTeacher


@pytest.mark.integration
class TestPPOTeacherIntegration:
    """Test PPO agent with teacher policies."""

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_ppo_with_random_teacher(self, mock_logger_class, mock_make_vec_env):
        """Test PPO trainer with random teacher."""
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
            environment=EnvironmentConfig(env_id="CartPole-v1", num_envs=2),
            teacher=TeacherConfig(type="random"),
            experiment=ExperimentConfig(device="cpu", name="test_random_teacher")
        )

        # Create trainer with teacher
        trainer = PPOTrainer(config)

        # Verify teacher was created
        assert trainer.teacher is not None
        assert isinstance(trainer.teacher, RandomTeacher)

        # Test teacher action generation
        obs = torch.randn(2, 4)
        actions = trainer.teacher.act(obs)

        assert actions.shape[0] == 2
        assert torch.all(torch.logical_or(actions == 0, actions == 1))

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_ppo_with_optimal_teacher(self, mock_logger_class, mock_make_vec_env):
        """Test PPO trainer with optimal teacher."""
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
            environment=EnvironmentConfig(env_id="CartPole-v1", num_envs=2),
            teacher=TeacherConfig(type="optimal"),
            experiment=ExperimentConfig(device="cpu", name="test_optimal_teacher")
        )

        # Create trainer with teacher
        trainer = PPOTrainer(config)

        # Verify teacher was created
        assert trainer.teacher is not None
        assert isinstance(trainer.teacher, CartPoleOptimalTeacher)

        # Test teacher action generation with CartPole-like observations
        obs = torch.tensor(
            [
                [0.1, 0.0, 0.1, 0.0],  # Should move right
                [-0.1, 0.0, -0.1, 0.0],  # Should move left
            ]
        )
        actions = trainer.teacher.act(obs)

        assert actions.shape[0] == 2
        assert torch.all(torch.logical_or(actions == 0, actions == 1))

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_teacher_device_consistency(self, mock_logger_class, mock_make_vec_env):
        """Test that teacher respects device placement."""
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
            teacher=TeacherConfig(type="random"),
            experiment=ExperimentConfig(device="cpu", name="test_device")
        )

        # Create trainer with teacher
        trainer = PPOTrainer(config)

        # Test teacher on same device as trainer
        obs = torch.randn(2, 4, device="cpu")
        actions = trainer.teacher.act(obs)

        # Actions should be on same device
        assert actions.device.type == "cpu"

    def test_teacher_action_consistency(self):
        """Test that teacher actions are consistent for same observations."""
        teacher = CartPoleOptimalTeacher()

        # Test deterministic behavior
        obs = torch.tensor([[0.1, 0.0, 0.1, 0.0]])  # Fixed observation

        action1 = teacher.act(obs)
        action2 = teacher.act(obs)

        # Should be deterministic for optimal teacher
        assert torch.equal(action1, action2)

    def test_teacher_batch_processing(self):
        """Test that teacher handles different batch sizes correctly."""
        teacher = RandomTeacher(
            action_space=Mock(n=2), observation_space=Mock(shape=(4,))
        )

        # Single observation
        obs_single = torch.randn(4)
        action_single = teacher.act(obs_single)
        assert action_single.numel() == 1

        # Batch of observations
        obs_batch = torch.randn(5, 4)
        actions_batch = teacher.act(obs_batch)
        assert actions_batch.shape[0] == 5

        # Empty batch (edge case)
        obs_empty = torch.randn(0, 4)
        actions_empty = teacher.act(obs_empty)
        assert actions_empty.shape[0] == 0

    def test_teacher_observation_preprocessing(self):
        """Test that teacher handles different observation formats."""
        teacher = CartPoleOptimalTeacher()

        # Test with numpy array
        import numpy as np

        obs_numpy = np.array([[0.1, 0.0, 0.1, 0.0]])
        actions_numpy = teacher.act(obs_numpy)
        assert isinstance(actions_numpy, np.ndarray)

        # Test with torch tensor
        obs_torch = torch.tensor([[0.1, 0.0, 0.1, 0.0]])
        actions_torch = teacher.act(obs_torch)
        assert isinstance(actions_torch, torch.Tensor)

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_teacher_integration_with_scheduler(
        self, mock_logger_class, mock_make_vec_env
    ):
        """Test teacher working together with scheduler."""
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
            teacher=TeacherConfig(type="optimal"),
            scheduler=SchedulerConfig(strategy="reward_based", trust_length=2),
            experiment=ExperimentConfig(device="cpu", name="test_combined")
        )

        # Create trainer with both teacher and scheduler
        trainer = PPOTrainer(config)

        # Verify both components exist
        assert trainer.teacher is not None
        assert trainer.scheduler is not None

        # Test that scheduler can choose policies
        policies = trainer.scheduler.choose_policy_type(
            iteration=0,
            global_step=0,
            steps_since_reset=torch.ones(4) * 5,
            prev_reward=torch.ones(4) * 0.5,
        )

        # Test that teacher can generate actions for teacher policies
        obs = torch.randn(4, 4)
        teacher_mask = torch.tensor([p == "teacher" for p in policies])

        if teacher_mask.any():
            teacher_obs = obs[teacher_mask]
            teacher_actions = trainer.teacher.act(teacher_obs)
            assert teacher_actions.shape[0] == teacher_mask.sum()

    def test_teacher_reset_behavior(self):
        """Test teacher reset functionality."""
        teacher = RandomTeacher(
            action_space=Mock(n=2), observation_space=Mock(shape=(4,))
        )

        # Reset should not raise errors
        teacher.reset()

        # Teacher should still work after reset
        obs = torch.randn(2, 4)
        actions = teacher.act(obs)
        assert actions.shape[0] == 2
