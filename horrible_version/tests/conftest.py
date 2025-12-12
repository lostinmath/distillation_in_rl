"""Shared pytest fixtures for adaptive_rl tests."""

from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from adaptive_rl.schedulers import (
    EpsilonScheduler,
    RewardBasedScheduler,
    StudentOnlyScheduler,
)
from adaptive_rl.teachers import CartPoleOptimalTeacher, RandomTeacher


@pytest.fixture
def device():
    """Fixture for compute device."""
    return "cpu"


@pytest.fixture
def num_envs():
    """Fixture for number of environments."""
    return 4


@pytest.fixture
def trust_length():
    """Fixture for trust length parameter."""
    return 5


@pytest.fixture
def mock_action_space():
    """Mock discrete action space."""
    action_space = Mock()
    action_space.n = 2
    action_space.shape = ()
    return action_space


@pytest.fixture
def mock_observation_space():
    """Mock observation space."""
    obs_space = Mock()
    obs_space.shape = (4,)
    return obs_space


@pytest.fixture
def sample_observations(num_envs):
    """Sample observations for testing."""
    return torch.randn(num_envs, 4)


@pytest.fixture
def sample_rewards(num_envs):
    """Sample rewards for testing."""
    return torch.randn(num_envs)


@pytest.fixture
def reset_rewards(num_envs):
    """Reset rewards (-1) for testing."""
    return torch.full((num_envs,), -1.0)


@pytest.fixture
def steps_since_reset(num_envs):
    """Sample steps since reset."""
    return torch.randint(0, 10, (num_envs,))


@pytest.fixture
def reward_based_scheduler(num_envs, trust_length, device):
    """Fixture for reward-based scheduler."""
    return RewardBasedScheduler(
        num_envs=num_envs,
        trust_length=trust_length,
        device=device,
    )


@pytest.fixture
def epsilon_scheduler(num_envs, trust_length, device):
    """Fixture for epsilon scheduler."""
    return EpsilonScheduler(
        num_envs=num_envs,
        epsilon=0.3,
        trust_length=trust_length,
        device=device,
    )


@pytest.fixture
def student_only_scheduler(num_envs, device):
    """Fixture for student-only scheduler."""
    return StudentOnlyScheduler(
        num_envs=num_envs,
        device=device,
    )


@pytest.fixture
def random_teacher(mock_action_space, mock_observation_space):
    """Fixture for random teacher."""
    return RandomTeacher(
        action_space=mock_action_space,
        observation_space=mock_observation_space,
    )


@pytest.fixture
def cartpole_teacher():
    """Fixture for CartPole optimal teacher."""
    return CartPoleOptimalTeacher()


@pytest.fixture
def cartpole_env():
    """Real CartPole environment for integration tests."""
    return gym.make("CartPole-v1")


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "experiment": {
            "name": "test_experiment",
            "seed": 42,
            "device": "cpu",
        },
        "environment": {
            "env_id": "CartPole-v1",
            "num_envs": 2,
            "num_steps": 10,
        },
        "training": {
            "num_iterations": 5,
            "learning_rate": 1e-3,
            "batch_size": 32,
        },
        "scheduler": {
            "strategy": "reward_based",
            "trust_length": 3,
        },
        "teacher": {
            "type": "random",
        },
    }


@pytest.fixture
def temp_log_dir(tmp_path):
    """Temporary logging directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return str(log_dir)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    logger.log_metrics = Mock()
    logger.log_param = Mock()
    logger.close = Mock()
    return logger


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
