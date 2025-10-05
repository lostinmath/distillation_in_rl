"""Unit tests for PPO components."""

from unittest.mock import Mock, patch

import gymnasium as gym
import torch
from torch import nn

from adaptive_rl.core.ppo import PPOAgent, PPOTrainer, layer_init
from adaptive_rl.core.config import (
    DistillationConfig, EnvironmentConfig, ExperimentConfig,
    TrainingConfig, SchedulerConfig, TeacherConfig, PPOConfig
)


class TestLayerInit:
    """Test layer initialization function."""

    def test_layer_init_linear(self):
        """Test layer initialization for linear layers."""
        layer = nn.Linear(10, 5)
        initialized_layer = layer_init(layer, std=1.0, bias_const=0.5)

        assert initialized_layer is layer  # Should return same object
        assert torch.allclose(layer.bias, torch.tensor(0.5))

    def test_layer_init_default_params(self):
        """Test layer initialization with default parameters."""
        layer = nn.Linear(10, 5)
        layer_init(layer)

        # Should not raise errors and bias should be zero
        assert torch.allclose(layer.bias, torch.tensor(0.0))


class TestPPOAgent:
    """Test PPO agent implementation."""

    def test_initialization_discrete(self):
        """Test PPO agent initialization for discrete action space."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)

        agent = PPOAgent(obs_space, action_space)

        assert agent.discrete_actions is True
        assert hasattr(agent, "shared_net")
        assert hasattr(agent, "actor")
        assert hasattr(agent, "critic")

    def test_initialization_continuous(self):
        """Test PPO agent initialization for continuous action space."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

        agent = PPOAgent(obs_space, action_space)

        assert agent.discrete_actions is False
        assert hasattr(agent, "actor_mean")
        assert hasattr(agent, "actor_logstd")

    def test_get_value(self):
        """Test value function estimation."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        agent = PPOAgent(obs_space, action_space)

        obs = torch.randn(3, 4)
        values = agent.get_value(obs)

        assert values.shape == (3, 1)
        assert isinstance(values, torch.Tensor)

    def test_get_action_and_value_discrete(self):
        """Test action and value generation for discrete actions."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        agent = PPOAgent(obs_space, action_space)

        obs = torch.randn(3, 4)
        action, logprob, entropy, value = agent.get_action_and_value(obs)

        assert action.shape == (3,)
        assert logprob.shape == (3,)
        assert entropy.shape == (3,)
        assert value.shape == (3, 1)

        # Actions should be 0 or 1
        assert torch.all(torch.logical_or(action == 0, action == 1))

    def test_get_action_and_value_continuous(self):
        """Test action and value generation for continuous actions."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        agent = PPOAgent(obs_space, action_space)

        obs = torch.randn(3, 4)
        action, logprob, entropy, value = agent.get_action_and_value(obs)

        assert action.shape == (3, 2)
        assert logprob.shape == (3,)
        assert entropy.shape == (3,)
        assert value.shape == (3, 1)

    def test_get_action_and_value_with_given_action(self):
        """Test action evaluation mode (providing action)."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        agent = PPOAgent(obs_space, action_space)

        obs = torch.randn(3, 4)
        given_actions = torch.tensor([0, 1, 0])

        action, logprob, entropy, value = agent.get_action_and_value(obs, given_actions)

        assert torch.equal(action, given_actions)
        assert logprob.shape == (3,)
        assert entropy.shape == (3,)
        assert value.shape == (3, 1)

    def test_different_activations(self):
        """Test different activation functions."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)

        # Test ReLU activation
        agent_relu = PPOAgent(obs_space, action_space, activation="relu")
        assert isinstance(agent_relu.activation(), nn.ReLU)

        # Test Tanh activation (default)
        agent_tanh = PPOAgent(obs_space, action_space, activation="tanh")
        assert isinstance(agent_tanh.activation(), nn.Tanh)

    def test_different_network_sizes(self):
        """Test different network architectures."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)

        # Test larger network
        agent = PPOAgent(obs_space, action_space, hidden_size=128, n_hidden_layers=3)

        obs = torch.randn(2, 4)
        action, logprob, entropy, value = agent.get_action_and_value(obs)

        # Should still work with different architecture
        assert action.shape == (2,)
        assert value.shape == (2, 1)


class TestPPOTrainer:
    """Test PPO trainer (integration-level testing)."""

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_trainer_initialization(self, mock_logger_class, mock_make_vec_env):
        """Test PPO trainer initialization."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        # Create config
        config = DistillationConfig(
            environment=EnvironmentConfig(env_id="CartPole-v1", num_envs=2, num_steps=10),
            training=TrainingConfig(num_iterations=5),
            experiment=ExperimentConfig(device="cpu", name="test_run")
        )

        trainer = PPOTrainer(config)

        assert trainer.env_id == "CartPole-v1"
        assert trainer.num_envs == 2
        assert trainer.num_steps == 10
        assert trainer.num_iterations == 5
        assert isinstance(trainer.agent, PPOAgent)

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_trainer_with_scheduler_config(self, mock_logger_class, mock_make_vec_env):
        """Test trainer with scheduler configuration."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        # Create config with scheduler
        config = DistillationConfig(
            environment=EnvironmentConfig(env_id="CartPole-v1"),
            scheduler=SchedulerConfig(strategy="reward_based", trust_length=3),
            experiment=ExperimentConfig(device="cpu", name="test_run")
        )

        trainer = PPOTrainer(config)

        # Should have created a scheduler
        assert trainer.scheduler is not None
        assert hasattr(trainer.scheduler, "trust_length")

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_trainer_with_teacher_config(self, mock_logger_class, mock_make_vec_env):
        """Test trainer with teacher configuration."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        # Create config with teacher
        config = DistillationConfig(
            environment=EnvironmentConfig(env_id="CartPole-v1"),
            teacher=TeacherConfig(type="random"),
            experiment=ExperimentConfig(device="cpu", name="test_run")
        )

        trainer = PPOTrainer(config)

        # Should have created a teacher
        assert trainer.teacher is not None

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_trainer_hyperparameters(self, mock_logger_class, mock_make_vec_env):
        """Test that trainer stores hyperparameters correctly."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        # Create config with hyperparameters
        config = DistillationConfig(
            environment=EnvironmentConfig(env_id="CartPole-v1"),
            training=TrainingConfig(learning_rate=1e-3),
            ppo=PPOConfig(gamma=0.95, gae_lambda=0.9, clip_coef=0.15),
            experiment=ExperimentConfig(device="cpu")
        )

        trainer = PPOTrainer(config)

        assert trainer.learning_rate == 1e-3
        assert trainer.gamma == 0.95
        assert trainer.gae_lambda == 0.9
        assert trainer.clip_coef == 0.15

    @patch("adaptive_rl.envs.make_env.make_vec_env")
    @patch("adaptive_rl.utils.logging.Logger")
    def test_save_checkpoint(self, mock_logger_class, mock_make_vec_env, tmp_path):
        """Test checkpoint saving functionality."""
        # Mock environment
        mock_env = Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        mock_env.single_action_space = gym.spaces.Discrete(2)
        mock_make_vec_env.return_value = mock_env

        # Mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        # Create config for checkpoint test
        config = DistillationConfig(
            environment=EnvironmentConfig(env_id="CartPole-v1"),
            experiment=ExperimentConfig(
                device="cpu",
                log_dir=str(tmp_path),
                name="test_run"
            )
        )

        trainer = PPOTrainer(config)

        # Test saving checkpoint
        trainer.save_checkpoint("test")

        # Check that checkpoint file was created
        checkpoint_path = tmp_path / "test_run" / "checkpoint_test.pt"
        assert checkpoint_path.exists()

        # Load and verify checkpoint contents
        checkpoint = torch.load(checkpoint_path)
        assert "agent_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "iteration" in checkpoint
        assert "scheduler_stats" in checkpoint


class TestPPOComponents:
    """Test individual PPO components and utilities."""

    def test_discrete_vs_continuous_consistency(self):
        """Test that discrete and continuous agents behave consistently."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))

        # Create both types of agents
        discrete_agent = PPOAgent(obs_space, gym.spaces.Discrete(2))
        continuous_agent = PPOAgent(
            obs_space, gym.spaces.Box(low=-1, high=1, shape=(1,))
        )

        obs = torch.randn(3, 4)

        # Both should work
        discrete_out = discrete_agent.get_action_and_value(obs)
        continuous_out = continuous_agent.get_action_and_value(obs)

        # Check output structure
        assert len(discrete_out) == 4  # action, logprob, entropy, value
        assert len(continuous_out) == 4

        # Values should have same shape
        assert discrete_out[3].shape == continuous_out[3].shape  # values

    def test_agent_training_mode(self):
        """Test that agent can switch between train and eval modes."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        agent = PPOAgent(obs_space, action_space)

        # Test mode switching
        agent.train()
        assert agent.training is True

        agent.eval()
        assert agent.training is False

    def test_agent_device_handling(self):
        """Test that agent can be moved between devices."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        agent = PPOAgent(obs_space, action_space)

        # Test moving to CPU (should work on any system)
        agent = agent.to("cpu")

        # Test with CPU tensors
        obs = torch.randn(2, 4)
        action, logprob, entropy, value = agent.get_action_and_value(obs)

        # All outputs should be on CPU
        assert action.device.type == "cpu"
        assert logprob.device.type == "cpu"
        assert entropy.device.type == "cpu"
        assert value.device.type == "cpu"

    def test_agent_parameter_count(self):
        """Test that agent has reasonable number of parameters."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        agent = PPOAgent(obs_space, action_space, hidden_size=64, n_hidden_layers=2)

        total_params = sum(p.numel() for p in agent.parameters())

        # Should have reasonable number of parameters (not too few, not too many)
        assert 1000 < total_params < 50000

    def test_agent_gradient_flow(self):
        """Test that gradients flow through the network."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        agent = PPOAgent(obs_space, action_space)

        obs = torch.randn(2, 4, requires_grad=True)
        action, logprob, entropy, value = agent.get_action_and_value(obs)

        # Compute some loss and backpropagate
        loss = logprob.mean() + value.mean()
        loss.backward()

        # Check that gradients exist for agent parameters
        for param in agent.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
