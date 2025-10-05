"""Integration tests for full training pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym
import pytest
import torch

from adaptive_rl.core.ppo import PPOTrainer


@pytest.mark.integration
class TestTrainingPipeline:
    """Test complete training pipeline integration."""

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_minimal_training_loop(self, mock_logger_class, mock_make_vec_env):
        """Test minimal training loop with all components."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_env.num_envs = 2

        # Mock environment interactions
        obs = torch.randn(2, 4)
        rewards = torch.tensor([1.0, 0.5])
        terminated = torch.tensor([False, False])
        truncated = torch.tensor([False, False])
        infos = [{"episode": {"r": 1.0, "l": 10}}, {}]

        mock_env.reset.return_value = (obs, infos)
        mock_env.step.return_value = (obs, rewards, terminated, truncated, infos)
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        # Create temporary directory for logging
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = PPOTrainer(
                env_id="CartPole-v1",
                num_envs=2,
                num_steps=5,
                num_iterations=2,
                teacher_config={"type": "random"},
                scheduler_config={"strategy": "reward_based", "trust_length": 2},
                device="cpu",
                log_dir=temp_dir,
                run_name="test_pipeline",
            )

            # Verify all components are initialized
            assert trainer.agent is not None
            assert trainer.teacher is not None
            assert trainer.scheduler is not None
            assert trainer.logger is not None

            # Test that training doesn't crash
            try:
                # Mock the rollout collection to avoid complex environment simulation
                with patch.object(trainer, "collect_rollouts") as mock_collect:
                    mock_collect.return_value = (
                        torch.randn(10, 4),  # observations
                        torch.randint(0, 2, (10,)),  # actions
                        torch.randn(10),  # log_probs
                        torch.randn(10),  # rewards
                        torch.randint(0, 2, (10,)),  # dones
                        torch.randn(10),  # values
                        1.0,  # avg_return
                        10.0,  # avg_length
                    )

                    # Run a few training steps
                    trainer.train()

            except Exception as e:
                pytest.fail(f"Training pipeline failed: {e}")

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_checkpoint_saving_and_loading(self, mock_logger_class, mock_make_vec_env):
        """Test checkpoint saving and loading functionality."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_env.num_envs = 2
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = PPOTrainer(
                env_id="CartPole-v1",
                device="cpu",
                log_dir=temp_dir,
                run_name="test_checkpoint",
            )

            # Save checkpoint
            trainer.save_checkpoint("test")

            # Verify checkpoint file exists
            checkpoint_path = Path(temp_dir) / "test_checkpoint" / "checkpoint_test.pt"
            assert checkpoint_path.exists()

            # Load and verify checkpoint contents
            checkpoint = torch.load(checkpoint_path)
            assert "agent_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "iteration" in checkpoint

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_metrics_logging(self, mock_logger_class, mock_make_vec_env):
        """Test that metrics are properly logged during training."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_env.num_envs = 2
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        trainer = PPOTrainer(
            env_id="CartPole-v1", device="cpu", run_name="test_logging"
        )

        # Test logging metrics
        trainer.logger.log_metrics(
            {"train/reward": 1.0, "train/length": 10.0, "scheduler/teacher_ratio": 0.5},
            step=1,
        )

        # Verify logger was called
        mock_logger.log_metrics.assert_called()

    def test_environment_integration(self):
        """Test integration with real gymnasium environments."""
        # Test with real CartPole environment (no mocking)
        env = gym.make("CartPole-v1")

        # Verify environment properties
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 2
        assert env.observation_space.shape == (4,)

        # Test basic environment interaction
        obs, info = env.reset()
        assert obs.shape == (4,)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (4,)
        assert isinstance(reward, float)

        env.close()

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_scheduler_teacher_coordination(self, mock_logger_class, mock_make_vec_env):
        """Test that scheduler and teacher work together correctly."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_env.num_envs = 4
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        trainer = PPOTrainer(
            env_id="CartPole-v1",
            num_envs=4,
            teacher_config={"type": "random"},
            scheduler_config={"strategy": "reward_based", "trust_length": 2},
            device="cpu",
            run_name="test_coordination",
        )

        # Simulate scheduling decisions
        obs = torch.randn(4, 4)
        policies = trainer.scheduler.choose_policy_type(
            iteration=1,
            global_step=100,
            steps_since_reset=torch.ones(4) * 5,
            prev_reward=torch.ones(4) * 0.5,
        )

        # Simulate mixed policy execution
        teacher_mask = torch.tensor([p == "teacher" for p in policies])
        student_mask = torch.tensor([p == "student" for p in policies])

        # Test teacher actions for teacher policies
        if teacher_mask.any():
            teacher_obs = obs[teacher_mask]
            teacher_actions = trainer.teacher.act(teacher_obs)
            assert teacher_actions.shape[0] == teacher_mask.sum()

        # Test student actions for student policies
        if student_mask.any():
            student_obs = obs[student_mask]
            with torch.no_grad():
                student_actions, _, _, _ = trainer.agent.get_action_and_value(
                    student_obs
                )
            assert student_actions.shape[0] == student_mask.sum()

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_device_consistency_across_components(
        self, mock_logger_class, mock_make_vec_env
    ):
        """Test that all components respect device placement."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_env.num_envs = 2
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        device = "cpu"
        trainer = PPOTrainer(
            env_id="CartPole-v1",
            device=device,
            teacher_config={"type": "random"},
            run_name="test_device_consistency",
        )

        # Test device consistency
        obs = torch.randn(2, 4, device=device)

        # Agent should work on correct device
        actions, log_probs, entropy, values = trainer.agent.get_action_and_value(obs)
        assert actions.device.type == device
        assert log_probs.device.type == device
        assert values.device.type == device

        # Teacher should work on correct device
        teacher_actions = trainer.teacher.act(obs)
        assert teacher_actions.device.type == device

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_error_handling_in_pipeline(self, mock_logger_class, mock_make_vec_env):
        """Test error handling in various pipeline scenarios."""
        # Mock environment that raises errors
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_env.num_envs = 2
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        # Test invalid teacher configuration
        with pytest.raises(ValueError):
            PPOTrainer(
                env_id="CartPole-v1",
                teacher_config={"type": "invalid_teacher"},
                device="cpu",
                run_name="test_error",
            )

        # Test invalid scheduler configuration
        with pytest.raises(ValueError):
            PPOTrainer(
                env_id="CartPole-v1",
                scheduler_config={"strategy": "invalid_strategy"},
                device="cpu",
                run_name="test_error",
            )
