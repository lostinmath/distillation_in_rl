#!/usr/bin/env python3
"""Test MVP improvements: error handling, performance, config validation."""

import torch
import numpy as np
from adaptive_rl.core.validated_config import validate_config_from_dict
from adaptive_rl.algorithms.behavioral_cloning import BehavioralCloning
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.core.error_handling import setup_error_handling
from adaptive_rl.envs import make_vec_env

def test_config_validation():
    """Test Pydantic config validation."""
    print("🧪 Testing Pydantic config validation...")

    # Valid config
    valid_config = {
        "experiment": {"name": "test", "seed": 42, "device": "cpu"},
        "algorithm": {"name": "ppo", "_target_": "adaptive_rl.algorithms.ppo.PPO"},
        "environment": {"name": "cartpole", "env_id": "CartPole-v1", "num_envs": 4},
        "scheduler": {"name": "student_only", "_target_": "adaptive_rl.schedulers.simple.StudentOnlyScheduler"},
        "training": {"total_timesteps": 1000, "eval_freq": 500},
        "paths": {"log_dir": "test_logs", "checkpoint_dir": "test_logs/checkpoints"}
    }

    try:
        config = validate_config_from_dict(valid_config)
        print("   ✅ Valid config passed validation")
    except Exception as e:
        print(f"   ❌ Valid config failed: {e}")
        return False

    # Invalid config - should fail
    invalid_config = valid_config.copy()
    invalid_config["algorithm"]["learning_rate"] = -1.0  # Invalid learning rate

    try:
        validate_config_from_dict(invalid_config)
        print("   ❌ Invalid config should have failed but passed")
        return False
    except Exception:
        print("   ✅ Invalid config correctly rejected")

    return True

def test_behavioral_cloning():
    """Test behavioral cloning implementation."""
    print("🧪 Testing Behavioral Cloning...")

    try:
        env = make_vec_env("CartPole-v1", num_envs=2, seed=42)

        bc = BehavioralCloning(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device="cpu"
        )

        # Test prediction
        obs = torch.randn(2, 4)  # Batch of 2 observations
        action, value, log_prob = bc.predict(obs)

        print(f"   ✅ BC prediction works: action={action.shape}, value={value.shape}")

        # Test training with dummy data
        demo_obs = np.random.randn(100, 4)
        demo_actions = np.random.randint(0, 2, size=100)

        metrics = bc.train_from_demonstrations(demo_obs, demo_actions)
        print(f"   ✅ BC training works: loss={metrics['bc/loss']:.3f}")

        return True

    except Exception as e:
        print(f"   ❌ BC test failed: {e}")
        return False

def test_evaluation_pipeline():
    """Test comprehensive evaluation."""
    print("🧪 Testing Evaluation Pipeline...")

    try:
        from adaptive_rl.algorithms.ppo import PPO

        env = make_vec_env("CartPole-v1", num_envs=1, seed=42)

        # Create a simple policy
        ppo = PPO(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device="cpu"
        )

        evaluator = Evaluator(env=env, n_eval_episodes=5)
        metrics = evaluator.evaluate_policy(ppo)

        print(f"   ✅ Evaluation works: return={metrics.mean_return:.2f} ± {metrics.std_return:.2f}")
        print(f"   📊 Success rate: {metrics.success_rate:.2f}")
        print(f"   📊 Episode length: {metrics.mean_episode_length:.1f}")

        return True

    except Exception as e:
        print(f"   ❌ Evaluation test failed: {e}")
        return False

def test_error_handling():
    """Test error handling capabilities."""
    print("🧪 Testing Error Handling...")

    try:
        setup_error_handling()

        from adaptive_rl.core.error_handling import (
            validate_tensors,
            SafeTensorOps,
            PerformanceMonitor
        )

        # Test tensor validation
        valid_tensor = torch.randn(10, 5)
        invalid_tensor = torch.tensor([float('nan'), 1.0, 2.0])

        validate_tensors(valid_tensor, names=["valid"])
        print("   ✅ Valid tensor passed validation")

        try:
            validate_tensors(invalid_tensor, names=["invalid"])
            print("   ❌ Invalid tensor should have failed")
            return False
        except ValueError:
            print("   ✅ Invalid tensor correctly rejected")

        # Test safe tensor operations
        tensors = [torch.randn(5, 3) for _ in range(4)]
        stacked = SafeTensorOps.safe_stack(tensors)
        print(f"   ✅ Safe tensor stacking: {stacked.shape}")

        # Test performance monitoring
        monitor = PerformanceMonitor()
        usage = monitor.check_memory_usage()
        print(f"   ✅ Performance monitoring: {usage['system_memory_gb']:.1f} GB")

        return True

    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False

def main():
    """Run all MVP improvement tests."""
    print("🚀 TESTING MVP IMPROVEMENTS")
    print("=" * 50)

    tests = [
        ("Config Validation", test_config_validation),
        ("Behavioral Cloning", test_behavioral_cloning),
        ("Evaluation Pipeline", test_evaluation_pipeline),
        ("Error Handling", test_error_handling),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        success = test_func()
        results.append((test_name, success))

    print("\n" + "=" * 50)
    print("🎯 TEST RESULTS:")

    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n🎉 ALL MVP IMPROVEMENTS WORKING!")
        print("✅ Ready for production experiments")
    else:
        print("\n💥 SOME TESTS FAILED")
        print("❌ Fix issues before running experiments")

    return all_passed

if __name__ == "__main__":
    main()