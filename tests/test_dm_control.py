#!/usr/bin/env python3

import sys
sys.path.append('src')

import numpy as np
import torch
from adaptive_rl.envs.dm_control_wrapper import create_dm_control_env
from adaptive_rl.teachers.dm_control_teachers import create_dm_control_teacher

def test_dm_control_basic():
    """Test basic dm_control environment functionality."""
    print("Testing dm_control basic functionality...")

    # Create environment and teacher
    env = create_dm_control_env('cheetah_run')
    teacher = create_dm_control_teacher('cheetah_run')

    # Test environment reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs[:5]}...")  # First 5 values

    # Test teacher action
    teacher_action = teacher.act(obs)
    print(f"Teacher action shape: {teacher_action.shape}")
    print(f"Teacher action: {teacher_action}")

    # Test environment step
    next_obs, reward, terminated, truncated, info = env.step(teacher_action)
    print(f"Step reward: {reward}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    # Test batch processing
    batch_obs = np.stack([obs, next_obs])
    batch_actions = teacher.act(batch_obs)
    print(f"Batch actions shape: {batch_actions.shape}")

    env.close()
    print("✓ Basic dm_control test passed\n")

def test_dm_control_episode():
    """Test a short episode with dm_control."""
    print("Testing dm_control episode...")

    env = create_dm_control_env('cheetah_run')
    teacher = create_dm_control_teacher('cheetah_run')

    obs, info = env.reset()
    total_reward = 0
    steps = 0
    max_steps = 100
    terminated = False
    truncated = False

    while steps < max_steps and not (terminated or truncated):
        action = teacher.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Episode finished after {steps} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward/steps:.3f}")

    env.close()
    print("✓ Episode test passed\n")

def test_continuous_action_space():
    """Test that we properly handle continuous actions."""
    print("Testing continuous action handling...")

    env = create_dm_control_env('cheetah_run')
    teacher = create_dm_control_teacher('cheetah_run')

    # Verify action space is continuous
    assert env.action_space.shape == (6,), f"Expected 6D action space, got {env.action_space.shape}"
    assert env.action_space.low.min() == -1.0, "Expected action bounds [-1, 1]"
    assert env.action_space.high.max() == 1.0, "Expected action bounds [-1, 1]"

    # Test action generation
    obs, _ = env.reset()
    action = teacher.act(obs)

    # Check action is properly bounded
    assert action.min() >= -1.0, f"Action below bounds: {action.min()}"
    assert action.max() <= 1.0, f"Action above bounds: {action.max()}"
    assert action.shape == (6,), f"Wrong action shape: {action.shape}"

    env.close()
    print("✓ Continuous action test passed\n")

if __name__ == "__main__":
    test_dm_control_basic()
    test_dm_control_episode()
    test_continuous_action_space()
    print("🎉 All dm_control tests passed!")