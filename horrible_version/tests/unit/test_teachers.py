"""Unit tests for teacher policies."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from adaptive_rl.teachers import (
    CartPoleOptimalTeacher,
    LunarLanderOptimalTeacher,
    PretrainedPPOTeacher,
    RandomTeacher,
    TeacherPolicy,
    create_optimal_teacher,
    create_teacher,
)


class TestRandomTeacher:
    """Test random teacher policy."""

    def test_initialization(self, mock_action_space, mock_observation_space):
        """Test random teacher initialization."""
        teacher = RandomTeacher(
            action_space=mock_action_space, observation_space=mock_observation_space
        )

        assert teacher.action_space == mock_action_space
        assert teacher.observation_space == mock_observation_space
        assert teacher.discrete is True
        assert teacher.n_actions == 2

    def test_discrete_action_generation(self, random_teacher, sample_observations):
        """Test discrete action generation."""
        actions = random_teacher.act(sample_observations)

        assert isinstance(actions, (torch.Tensor, np.ndarray))
        assert actions.shape[0] == sample_observations.shape[0]

        # Convert to numpy for testing
        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()

        # Check actions are in valid range
        assert np.all(actions >= 0)
        assert np.all(actions < 2)  # n_actions = 2

    def test_single_observation(self, random_teacher):
        """Test with single observation."""
        obs = torch.randn(4)  # Single observation
        action = random_teacher.act(obs)

        assert isinstance(action, (torch.Tensor, np.ndarray))
        # Should return scalar for single observation
        if isinstance(action, torch.Tensor):
            assert action.shape == () or action.shape == (1,)

    def test_continuous_action_space(self, mock_observation_space):
        """Test random teacher with continuous action space."""
        continuous_action_space = Mock()
        continuous_action_space.low = np.array([-1.0, -2.0])
        continuous_action_space.high = np.array([1.0, 2.0])
        continuous_action_space.shape = (2,)
        # Remove 'n' attribute to make it continuous
        if hasattr(continuous_action_space, "n"):
            delattr(continuous_action_space, "n")

        teacher = RandomTeacher(
            action_space=continuous_action_space,
            observation_space=mock_observation_space,
        )

        assert teacher.discrete is False

        obs = torch.randn(3, 4)  # Batch of 3 observations
        actions = teacher.act(obs)

        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()

        assert actions.shape == (3, 2)
        # Check actions are within bounds
        assert np.all(actions >= continuous_action_space.low)
        assert np.all(actions <= continuous_action_space.high)


class TestCartPoleOptimalTeacher:
    """Test CartPole optimal teacher policy."""

    def test_initialization(self):
        """Test CartPole teacher initialization."""
        teacher = CartPoleOptimalTeacher()
        assert isinstance(teacher, TeacherPolicy)

    def test_cartpole_action_generation(self, cartpole_teacher):
        """Test CartPole action generation."""
        # Create CartPole-like observations
        obs = torch.tensor(
            [
                [0.1, 0.0, 0.1, 0.0],  # Lean right, move right
                [-0.1, 0.0, -0.1, 0.0],  # Lean left, move left
                [0.0, 0.0, 0.0, 0.0],  # Balanced
            ]
        )

        actions = cartpole_teacher.act(obs)

        assert isinstance(actions, (torch.Tensor, np.ndarray))
        assert actions.shape[0] == 3

        # Convert to numpy for testing
        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()

        # Actions should be 0 or 1
        assert np.all(np.isin(actions, [0, 1]))

    def test_single_cartpole_observation(self, cartpole_teacher):
        """Test with single CartPole observation."""
        obs = torch.tensor([0.1, 0.0, 0.1, 0.0])  # Single observation
        action = cartpole_teacher.act(obs)

        assert isinstance(action, (torch.Tensor, np.ndarray))
        if isinstance(action, torch.Tensor):
            action_val = action.item()
        else:
            action_val = action

        assert action_val in [0, 1]

    def test_numpy_input(self, cartpole_teacher):
        """Test CartPole teacher with numpy input."""
        obs = np.array([[0.1, 0.0, 0.1, 0.0]])
        actions = cartpole_teacher.act(obs)

        assert isinstance(actions, np.ndarray)
        assert actions.shape[0] == 1
        assert actions[0] in [0, 1]

    def test_action_logic(self, cartpole_teacher):
        """Test the heuristic logic of CartPole teacher."""
        # Test cases where we expect specific actions
        test_cases = [
            # [cart_pos, cart_vel, pole_angle, pole_vel] -> expected tendency
            ([0.0, 0.0, 0.2, 0.1], 1),  # Pole leaning right with velocity -> move right
            ([0.0, 0.0, -0.2, -0.1], 0),  # Pole leaning left with velocity -> move left
        ]

        for obs_list, expected_action in test_cases:
            obs = torch.tensor([obs_list])
            action = cartpole_teacher.act(obs)

            if isinstance(action, torch.Tensor):
                action_val = action.item()
            else:
                action_val = action[0]

            assert action_val == expected_action


class TestLunarLanderOptimalTeacher:
    """Test LunarLander optimal teacher policy."""

    def test_initialization(self):
        """Test LunarLander teacher initialization."""
        teacher = LunarLanderOptimalTeacher()
        assert isinstance(teacher, TeacherPolicy)

    def test_lunarlander_action_generation(self):
        """Test LunarLander action generation."""
        teacher = LunarLanderOptimalTeacher()

        # Create LunarLander-like observations
        obs = torch.tensor(
            [
                [0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],  # High altitude, falling
                [0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 1.0, 1.0],  # Low altitude, landing
            ]
        )

        actions = teacher.act(obs)

        assert isinstance(actions, (torch.Tensor, np.ndarray))
        assert actions.shape[0] == 2

        # Convert to numpy for testing
        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()

        # Actions should be 0, 1, 2, or 3
        assert np.all(np.isin(actions, [0, 1, 2, 3]))

    def test_single_lunarlander_observation(self):
        """Test with single LunarLander observation."""
        teacher = LunarLanderOptimalTeacher()
        obs = torch.tensor([0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0])

        action = teacher.act(obs)

        assert isinstance(action, (torch.Tensor, np.ndarray))
        if isinstance(action, torch.Tensor):
            action_val = action.item()
        else:
            action_val = action

        assert action_val in [0, 1, 2, 3]


class TestPretrainedPPOTeacher:
    """Test pretrained PPO teacher policy."""

    @patch("torch.load")
    def test_initialization(self, mock_load, mock_action_space, mock_observation_space):
        """Test pretrained teacher initialization."""
        # Mock the loaded model (return the model directly, not a dict)
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_load.return_value = mock_model

        teacher = PretrainedPPOTeacher(
            checkpoint_path="fake_path.pt",
            action_space=mock_action_space,
            observation_space=mock_observation_space,
        )

        assert teacher.checkpoint_path == "fake_path.pt"
        assert teacher.deterministic is True
        mock_load.assert_called_once()
        mock_model.eval.assert_called_once()

    @patch("torch.load")
    def test_action_generation(
        self, mock_load, mock_action_space, mock_observation_space
    ):
        """Test pretrained teacher action generation."""
        # Mock the model and its outputs
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        # Mock both get_action method and direct forward call
        mock_model.get_action = Mock(return_value=torch.tensor([0, 1]))
        mock_model.return_value = torch.tensor([[0.8, 0.2], [0.3, 0.7]])  # Logits
        mock_load.return_value = mock_model

        teacher = PretrainedPPOTeacher(
            checkpoint_path="fake_path.pt",
            action_space=mock_action_space,
            observation_space=mock_observation_space,
            deterministic=True,
        )

        obs = torch.randn(2, 4)
        actions = teacher.act(obs)

        assert isinstance(actions, (torch.Tensor, np.ndarray))
        assert actions.shape[0] == 2

    @patch("torch.load")
    def test_device_handling(
        self, mock_load, mock_action_space, mock_observation_space
    ):
        """Test device handling for pretrained teacher."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        mock_load.return_value = mock_model

        teacher = PretrainedPPOTeacher(
            checkpoint_path="fake_path.pt",
            action_space=mock_action_space,
            observation_space=mock_observation_space,
            device="cuda",
        )

        # Test moving to device
        teacher.to("cpu")
        mock_model.to.assert_called_with("cpu")


class TestOptimalTeacherFactory:
    """Test optimal teacher factory function."""

    def test_create_cartpole_teacher(self):
        """Test creating CartPole optimal teacher."""
        teacher = create_optimal_teacher("CartPole-v1")
        assert isinstance(teacher, CartPoleOptimalTeacher)

    def test_create_lunarlander_teacher(self):
        """Test creating LunarLander optimal teacher."""
        teacher = create_optimal_teacher("LunarLander-v2")
        assert isinstance(teacher, LunarLanderOptimalTeacher)

    def test_invalid_environment(self):
        """Test error handling for unsupported environment."""
        with pytest.raises(ValueError, match="No optimal teacher available"):
            create_optimal_teacher("UnsupportedEnv-v1")


class TestTeacherFactory:
    """Test teacher factory function."""

    def test_create_random_teacher(self, mock_action_space, mock_observation_space):
        """Test creating random teacher via factory."""
        teacher = create_teacher(
            teacher_type="random",
            action_space=mock_action_space,
            observation_space=mock_observation_space,
        )

        assert isinstance(teacher, RandomTeacher)

    def test_create_optimal_teacher_via_factory(
        self, mock_action_space, mock_observation_space
    ):
        """Test creating optimal teacher via factory."""
        teacher = create_teacher(
            teacher_type="optimal",
            env_id="CartPole-v1",
            action_space=mock_action_space,
            observation_space=mock_observation_space,
        )

        assert isinstance(teacher, CartPoleOptimalTeacher)

    def test_optimal_teacher_missing_env_id(
        self, mock_action_space, mock_observation_space
    ):
        """Test error when env_id is missing for optimal teacher."""
        with pytest.raises(ValueError, match="env_id must be specified"):
            create_teacher(
                teacher_type="optimal",
                action_space=mock_action_space,
                observation_space=mock_observation_space,
            )

    @patch("torch.load")
    def test_create_pretrained_teacher(
        self, mock_load, mock_action_space, mock_observation_space
    ):
        """Test creating pretrained teacher via factory."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_load.return_value = mock_model

        teacher = create_teacher(
            teacher_type="pretrained",
            action_space=mock_action_space,
            observation_space=mock_observation_space,
            checkpoint_path="fake_path.pt",
        )

        assert isinstance(teacher, PretrainedPPOTeacher)

    def test_invalid_teacher_type(self, mock_action_space, mock_observation_space):
        """Test error handling for invalid teacher type."""
        with pytest.raises(ValueError, match="Unknown teacher type"):
            create_teacher(
                teacher_type="invalid_teacher",
                action_space=mock_action_space,
                observation_space=mock_observation_space,
            )


class TestTeacherInterface:
    """Test that all teachers implement the interface correctly."""

    def test_teacher_interface_compliance(self, random_teacher, cartpole_teacher):
        """Test that teachers implement required interface."""
        teachers = [random_teacher, cartpole_teacher]

        for teacher in teachers:
            # Check that act method exists and works
            assert hasattr(teacher, "act")
            assert callable(teacher.act)

            # Check that reset method exists (may be no-op)
            assert hasattr(teacher, "reset")
            assert callable(teacher.reset)

            # Check that to method exists for device handling
            assert hasattr(teacher, "to")
            assert callable(teacher.to)

    def test_teacher_reset_method(self, random_teacher, cartpole_teacher):
        """Test teacher reset method (should not raise errors)."""
        teachers = [random_teacher, cartpole_teacher]

        for teacher in teachers:
            # Should not raise any errors
            teacher.reset()

    def test_teacher_device_method(self, random_teacher, cartpole_teacher):
        """Test teacher device method."""
        teachers = [random_teacher, cartpole_teacher]

        for teacher in teachers:
            # Should return self and not raise errors
            result = teacher.to("cpu")
            assert result is teacher
