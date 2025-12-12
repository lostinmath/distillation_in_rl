#!/usr/bin/env python3
"""Simple MVP test - just verify core functionality works."""

import torch
import numpy as np
from adaptive_rl.algorithms.ppo import PPO
from adaptive_rl.envs import make_vec_env
from loguru import logger

def test_basic_functionality():
    """Test basic PPO + environment functionality."""
    print("🧪 Testing Basic MVP Functionality...")

    try:
        # Create environment
        env = make_vec_env("CartPole-v1", num_envs=2, seed=42)
        print("   ✅ Environment created")

        # Create PPO
        ppo = PPO(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device="cpu",
            n_steps=64,
            batch_size=32,
        )
        print("   ✅ PPO algorithm created")

        # Test prediction
        obs = torch.tensor(env.reset()[0], dtype=torch.float32)
        action, value, log_prob = ppo.predict(obs)
        print(f"   ✅ Prediction works: action={action.shape}")

        # Test mini training loop
        rollout_buffer = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "log_probs": [],
        }

        # Collect mini rollout
        for step in range(ppo.n_steps):
            action, value, log_prob = ppo.predict(obs)
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated | truncated

            rollout_buffer["observations"].append(obs)
            rollout_buffer["actions"].append(action)
            rollout_buffer["rewards"].append(torch.tensor(reward))
            rollout_buffer["dones"].append(torch.tensor(done))
            rollout_buffer["values"].append(value.squeeze(-1))
            rollout_buffer["log_probs"].append(log_prob)

            obs = torch.tensor(next_obs, dtype=torch.float32)

        # Convert to training format
        rollout_data = {}
        for key, values in rollout_buffer.items():
            rollout_data[key] = torch.stack(values)
        rollout_data["next_observations"] = obs

        # Test training step
        metrics = ppo.train_step(rollout_data)
        print(f"   ✅ Training step works: loss={metrics.get('loss/total', 0):.3f}")

        # Test save/load
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            ppo.save(f.name)
            ppo.load(f.name)
            print(f"   ✅ Save/load works")

        return True

    except Exception as e:
        print(f"   ❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_schedulers():
    """Test scheduling strategies."""
    print("🧪 Testing Schedulers...")

    try:
        from adaptive_rl.schedulers import StudentOnlyScheduler, RewardBasedScheduler
        from adaptive_rl.teachers import create_teacher

        env = make_vec_env("CartPole-v1", num_envs=2, seed=42)

        # Test student-only scheduler
        student_scheduler = StudentOnlyScheduler(num_envs=2)
        policies = student_scheduler.choose_policy_type(
            iteration=0,
            global_step=0,
            steps_since_reset=torch.zeros(2),
            prev_reward=torch.zeros(2),
        )
        print(f"   ✅ Student-only scheduler: {policies}")

        # Test reward-based scheduler with teacher
        teacher = create_teacher(
            teacher_type="optimal",
            env_id="CartPole-v1",
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

        reward_scheduler = RewardBasedScheduler(
            teacher_policy=teacher,
            num_envs=2,
            trust_period=3,
        )

        policies = reward_scheduler.choose_policy_type(
            iteration=0,
            global_step=0,
            steps_since_reset=torch.zeros(2),
            prev_reward=torch.zeros(2),
        )
        print(f"   ✅ Reward-based scheduler: {policies}")

        return True

    except Exception as e:
        print(f"   ❌ Scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_git_tracking():
    """Test git tracking in scripts."""
    print("🧪 Testing Git Tracking...")

    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()

        print(f"   ✅ Git tracking works: {branch}@{commit[:8]}")
        return True

    except Exception as e:
        print(f"   ❌ Git tracking failed: {e}")
        return False

def main():
    """Run simple MVP tests."""
    print("🚀 SIMPLE MVP FUNCTIONALITY TEST")
    print("=" * 40)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Schedulers", test_schedulers),
        ("Git Tracking", test_git_tracking),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        success = test_func()
        results.append((test_name, success))

    print("\n" + "=" * 40)
    print("🎯 RESULTS:")

    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n🎉 MVP CORE FUNCTIONALITY WORKS!")
        print("✅ Ready for experiments")
        print("\nTo run experiments:")
        print("   ./scripts/run_quick_test.sh")
        print("   ./scripts/run_baseline_comparison.sh")
    else:
        print("\n💥 CORE ISSUES DETECTED")

    return all_passed

if __name__ == "__main__":
    main()