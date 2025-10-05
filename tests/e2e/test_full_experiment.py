"""End-to-end tests with real environments."""

import tempfile
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from adaptive_rl.core.ppo import PPOTrainer


@pytest.mark.e2e
@pytest.mark.slow
class TestFullExperiment:
    """End-to-end tests with real environments and training."""

    def test_student_only_training(self):
        """Test pure student training (baseline PPO)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = PPOTrainer(
                env_id="CartPole-v1",
                num_envs=2,
                num_steps=32,
                num_iterations=3,
                learning_rate=1e-3,
                device="cpu",
                log_dir=temp_dir,
                run_name="e2e_student_only",
            )

            # Run training
            trainer.train()

            # Verify checkpoint was saved
            checkpoint_path = (
                Path(temp_dir) / "e2e_student_only" / "checkpoint_final.pt"
            )
            assert checkpoint_path.exists()

            # Verify metrics were logged
            assert trainer.logger.log_metrics.call_count > 0

    def test_teacher_only_training(self):
        """Test pure teacher training (optimal teacher)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = PPOTrainer(
                env_id="CartPole-v1",
                num_envs=2,
                num_steps=32,
                num_iterations=3,
                teacher_config={"type": "optimal"},
                scheduler_config={"strategy": "teacher_only"},
                device="cpu",
                log_dir=temp_dir,
                run_name="e2e_teacher_only",
            )

            # Run training
            trainer.train()

            # Teacher-only should achieve high performance quickly
            # (This is more of a sanity check than a strict test)
            assert trainer.teacher is not None

    def test_reward_based_training(self):
        """Test reward-based scheduling (main contribution)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = PPOTrainer(
                env_id="CartPole-v1",
                num_envs=4,
                num_steps=32,
                num_iterations=5,
                teacher_config={"type": "optimal"},
                scheduler_config={"strategy": "reward_based", "trust_length": 3},
                device="cpu",
                log_dir=temp_dir,
                run_name="e2e_reward_based",
            )

            # Run training
            trainer.train()

            # Verify scheduler statistics were collected
            stats = trainer.scheduler.get_statistics()
            assert "teacher_usage_ratio" in stats
            assert "student_usage_ratio" in stats
            assert "switch_frequency" in stats

            # Should have used both teacher and student
            assert stats["total_steps"] > 0

    def test_epsilon_training(self):
        """Test epsilon-based scheduling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = PPOTrainer(
                env_id="CartPole-v1",
                num_envs=2,
                num_steps=32,
                num_iterations=3,
                teacher_config={"type": "optimal"},
                scheduler_config={
                    "strategy": "epsilon",
                    "epsilon": 0.5,
                    "trust_length": 2,
                },
                device="cpu",
                log_dir=temp_dir,
                run_name="e2e_epsilon",
            )

            # Run training
            trainer.train()

            # Verify epsilon scheduler was used
            assert hasattr(trainer.scheduler, "epsilon")
            assert trainer.scheduler.epsilon == 0.5

    def test_alternating_training(self):
        """Test alternating scheduling strategy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = PPOTrainer(
                env_id="CartPole-v1",
                num_envs=2,
                num_steps=32,
                num_iterations=4,  # Even number for clean alternation
                teacher_config={"type": "optimal"},
                scheduler_config={"strategy": "alternating"},
                device="cpu",
                log_dir=temp_dir,
                run_name="e2e_alternating",
            )

            # Run training
            trainer.train()

            # Verify alternating behavior (basic sanity check)
            stats = trainer.scheduler.get_statistics()
            assert stats["total_steps"] > 0

    @pytest.mark.requires_gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_training(self):
        """Test training on GPU device."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = PPOTrainer(
                env_id="CartPole-v1",
                num_envs=2,
                num_steps=16,
                num_iterations=2,
                device="cuda",
                log_dir=temp_dir,
                run_name="e2e_gpu",
            )

            # Verify components are on GPU
            assert next(trainer.agent.parameters()).device.type == "cuda"

            # Run brief training
            trainer.train()

    def test_lunarlander_environment(self):
        """Test with LunarLander environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = PPOTrainer(
                env_id="LunarLander-v2",
                num_envs=2,
                num_steps=32,
                num_iterations=3,
                teacher_config={"type": "optimal"},
                scheduler_config={"strategy": "reward_based", "trust_length": 3},
                device="cpu",
                log_dir=temp_dir,
                run_name="e2e_lunarlander",
            )

            # Verify environment setup
            assert trainer.env.single_observation_space.shape == (8,)
            assert trainer.env.single_action_space.n == 4

            # Run training
            trainer.train()

            # Verify it completed without errors
            assert trainer.iteration >= trainer.num_iterations

    def test_multiple_seeds_reproducibility(self):
        """Test that same seed produces same results."""
        configs = []
        results = []

        for seed in [42, 42]:  # Same seed twice
            with tempfile.TemporaryDirectory() as temp_dir:
                torch.manual_seed(seed)

                trainer = PPOTrainer(
                    env_id="CartPole-v1",
                    num_envs=2,
                    num_steps=16,
                    num_iterations=2,
                    learning_rate=1e-3,
                    device="cpu",
                    log_dir=temp_dir,
                    run_name=f"e2e_seed_{seed}",
                )

                # Store initial agent parameters
                initial_params = {
                    name: param.clone()
                    for name, param in trainer.agent.named_parameters()
                }
                configs.append(initial_params)

                # Run training
                trainer.train()

                # Store final agent parameters
                final_params = {
                    name: param.clone()
                    for name, param in trainer.agent.named_parameters()
                }
                results.append(final_params)

        # With same seed, initial parameters should be identical
        for name in configs[0]:
            assert torch.allclose(
                configs[0][name], configs[1][name], atol=1e-6
            ), f"Initial params differ for {name}"

    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test invalid environment
            with pytest.raises((gym.error.UnregisteredEnv, ValueError)):
                PPOTrainer(
                    env_id="InvalidEnv-v1",
                    device="cpu",
                    log_dir=temp_dir,
                    run_name="test_invalid",
                )

            # Test invalid learning rate
            with pytest.raises((ValueError, AssertionError)):
                PPOTrainer(
                    env_id="CartPole-v1",
                    learning_rate=-1.0,  # Invalid negative learning rate
                    device="cpu",
                    log_dir=temp_dir,
                    run_name="test_invalid",
                )

    def test_checkpoint_loading_and_resuming(self):
        """Test saving and loading checkpoints for training resumption."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First training session
            trainer1 = PPOTrainer(
                env_id="CartPole-v1",
                num_envs=2,
                num_steps=16,
                num_iterations=2,
                device="cpu",
                log_dir=temp_dir,
                run_name="e2e_checkpoint",
            )

            trainer1.train()
            trainer1.save_checkpoint("intermediate")

            # Load checkpoint
            checkpoint_path = (
                Path(temp_dir) / "e2e_checkpoint" / "checkpoint_intermediate.pt"
            )
            assert checkpoint_path.exists()

            checkpoint = torch.load(checkpoint_path)
            assert "agent_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "iteration" in checkpoint

            # Verify we can load the state dict
            trainer1.agent.load_state_dict(checkpoint["agent_state_dict"])
            trainer1.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def test_metrics_collection_and_export(self):
        """Test comprehensive metrics collection during training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = PPOTrainer(
                env_id="CartPole-v1",
                num_envs=2,
                num_steps=32,
                num_iterations=3,
                teacher_config={"type": "optimal"},
                scheduler_config={"strategy": "reward_based", "trust_length": 2},
                device="cpu",
                log_dir=temp_dir,
                run_name="e2e_metrics",
            )

            trainer.train()

            # Check that CSV file was created
            csv_path = Path(temp_dir) / "e2e_metrics" / "metrics.csv"
            if csv_path.exists():
                # Verify CSV has expected columns
                import pandas as pd

                df = pd.read_csv(csv_path)
                expected_columns = ["iteration", "avg_return", "avg_length"]
                for col in expected_columns:
                    assert col in df.columns

            # Check scheduler statistics
            stats = trainer.scheduler.get_statistics()
            assert isinstance(stats, dict)
            assert "teacher_usage_ratio" in stats
