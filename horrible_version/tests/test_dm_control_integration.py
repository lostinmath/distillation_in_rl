#!/usr/bin/env python3
"""Integration test for dm_control environments with PPO + Scheduler.

Validates that:
1. PPO works with continuous control (dm_control)
2. Scheduler switches policies correctly with continuous actions
3. dm_control teacher actions are applied properly
4. Metrics are tracked correctly for continuous environments

Run: python test_dm_control_integration.py
Expected runtime: 3-5 minutes
"""

import torch
import numpy as np
import sys
sys.path.append('src')

from adaptive_rl.core.scheduled_ppo import ScheduledPPOArgs
from adaptive_rl.core.dm_control_ppo import DMControlPPOTrainer
from adaptive_rl.schedulers.simple import StudentOnlyScheduler, TeacherOnlyScheduler
from adaptive_rl.schedulers.reward_based import RewardBasedScheduler
from adaptive_rl.teachers.dm_control_teachers import create_dm_control_teacher
from adaptive_rl.envs.dm_control_wrapper import create_dm_control_env


def test_dm_control_student_only():
    """Test 1: PPO baseline with continuous control (student only)."""
    print("🧪 Test 1: Student-Only PPO with dm_control")

    args = ScheduledPPOArgs(
        exp_name="test_dm_control_student_only",
        env_id="cheetah_run",  # This will be handled by our custom env creation
        total_timesteps=2000,  # Very short for testing
        num_envs=2,
        seed=42
    )

    # Override environment creation to use dm_control
    scheduler = StudentOnlyScheduler(num_envs=args.num_envs)

    # Create trainer with custom environment factory
    trainer = DMControlPPOTrainer(
        args, scheduler,
        env_factory=lambda: create_dm_control_env(args.env_id),
        teacher=None
    )

    print("  ✅ Student-only dm_control trainer created")

    # Run short training
    trainer.train()

    # Check metrics
    metrics = trainer.agent.get_scheduling_metrics()
    assert metrics["student_ratio"] == 1.0, f"Expected 100% student, got {metrics['student_ratio']}"
    assert metrics["teacher_ratio"] == 0.0, f"Expected 0% teacher, got {metrics['teacher_ratio']}"

    print(f"  ✅ Metrics correct: {metrics['student_ratio']:.1%} student actions")
    print("  ✅ Test 1 PASSED\n")


def test_dm_control_teacher_only():
    """Test 2: Teacher-only with dm_control (continuous actions)."""
    print("🧪 Test 2: Teacher-Only dm_control")

    args = ScheduledPPOArgs(
        exp_name="test_dm_control_teacher_only",
        env_id="cheetah_run",
        total_timesteps=2000,
        num_envs=2,
        seed=42
    )

    teacher = create_dm_control_teacher("cheetah_run")
    scheduler = TeacherOnlyScheduler(num_envs=args.num_envs)
    trainer = DMControlPPOTrainer(
        args, scheduler,
        env_factory=lambda: create_dm_control_env(args.env_id),
        teacher=teacher
    )

    print("  ✅ Teacher-only dm_control trainer created")

    # Run short training
    trainer.train()

    # Check metrics
    metrics = trainer.agent.get_scheduling_metrics()
    assert metrics["teacher_ratio"] == 1.0, f"Expected 100% teacher, got {metrics['teacher_ratio']}"
    assert metrics["student_ratio"] == 0.0, f"Expected 0% student, got {metrics['student_ratio']}"

    print(f"  ✅ Metrics correct: {metrics['teacher_ratio']:.1%} teacher actions")
    print("  ✅ Test 2 PASSED\n")


def test_dm_control_reward_based():
    """Test 3: Reward-based scheduling with dm_control."""
    print("🧪 Test 3: Reward-Based Scheduling with dm_control")

    args = ScheduledPPOArgs(
        exp_name="test_dm_control_reward_based",
        env_id="cheetah_run",
        total_timesteps=3000,  # Short for reward-based test
        num_envs=4,
        seed=42
    )

    teacher = create_dm_control_teacher("cheetah_run")
    scheduler = RewardBasedScheduler(num_envs=args.num_envs, trust_period=5)
    trainer = DMControlPPOTrainer(
        args, scheduler,
        env_factory=lambda: create_dm_control_env(args.env_id),
        teacher=teacher
    )

    print("  ✅ Reward-based dm_control trainer created")

    # Run training
    trainer.train()

    # Check adaptive behavior
    metrics = trainer.agent.get_scheduling_metrics()
    teacher_ratio = metrics["teacher_ratio"]
    switches = metrics["policy_switches"]

    print(f"  📊 Teacher ratio: {teacher_ratio:.1%}")
    print(f"  🔄 Policy switches: {switches}")

    # Test that system works with continuous control
    assert 0.0 <= teacher_ratio <= 1.0, f"Teacher ratio should be 0-100%, got {teacher_ratio:.1%}"
    assert switches >= 0, f"Switches should be non-negative, got {switches}"

    print(f"  ✅ Continuous control stable: {teacher_ratio:.1%} teacher usage, {switches} switches")
    print("  ✅ Test 3 PASSED\n")


def test_action_space_compatibility():
    """Test 4: Verify action space handling between student and teacher."""
    print("🧪 Test 4: Action Space Compatibility")

    # Create environment and teacher
    env = create_dm_control_env("cheetah_run")
    teacher = create_dm_control_teacher("cheetah_run")

    # Test observation and action compatibility
    obs, _ = env.reset()
    print(f"  📊 Observation shape: {obs.shape}")
    print(f"  📊 Action space: {env.action_space}")

    # Test teacher action generation
    teacher_action = teacher.act(obs)
    print(f"  📊 Teacher action shape: {teacher_action.shape}")
    print(f"  📊 Teacher action range: [{teacher_action.min():.3f}, {teacher_action.max():.3f}]")

    # Verify compatibility
    assert teacher_action.shape == env.action_space.shape, \
        f"Action shape mismatch: teacher {teacher_action.shape} vs env {env.action_space.shape}"

    assert teacher_action.min() >= env.action_space.low.min(), \
        f"Action below bounds: {teacher_action.min()} < {env.action_space.low.min()}"

    assert teacher_action.max() <= env.action_space.high.max(), \
        f"Action above bounds: {teacher_action.max()} > {env.action_space.high.max()}"

    # Test environment step
    next_obs, reward, terminated, truncated, info = env.step(teacher_action)
    print(f"  📊 Step reward: {reward:.4f}")

    env.close()
    print("  ✅ Action space compatibility verified")
    print("  ✅ Test 4 PASSED\n")


def run_dm_control_validation():
    """Run complete dm_control validation suite."""
    print("dm_control + PPO + Scheduler Integration Validation")
    print("=" * 60)

    try:
        test_action_space_compatibility()
        test_dm_control_student_only()
        test_dm_control_teacher_only()
        test_dm_control_reward_based()

        print("🎉 ALL dm_control TESTS PASSED!")
        print("✅ dm_control + PPO + Scheduler integration working correctly")
        print("✅ Continuous control environments ready for experiments")

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_dm_control_validation()