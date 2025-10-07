#!/usr/bin/env python3
"""Test dm_control integration and environment setup."""

import numpy as np
import gymnasium as gym
from dm_control import suite
import shimmy


def test_dm_control_basic():
    """Test basic dm_control environment creation."""
    print("Testing dm_control basic integration...")

    # Test direct dm_control
    try:
        env = suite.load(domain_name="cheetah", task_name="run")
        time_step = env.reset()
        print(f"  dm_control cheetah_run: obs_shape={time_step.observation['position'].shape}")

        # Test action
        action_spec = env.action_spec()
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
        time_step = env.step(action)
        print(f"  Action executed: reward={time_step.reward}")

        print("  ✓ dm_control basic test passed")
        return True

    except Exception as e:
        print(f"  ✗ dm_control basic test failed: {e}")
        return False


def test_shimmy_wrapper():
    """Test shimmy gymnasium wrapper for dm_control."""
    print("\nTesting shimmy wrapper integration...")

    try:
        # Use shimmy to wrap dm_control as gymnasium env
        from shimmy.dm_control_compatibility import DmControlCompatibilityV0

        dm_env = suite.load(domain_name="cheetah", task_name="run")
        gym_env = DmControlCompatibilityV0(dm_env)

        obs, info = gym_env.reset()
        print(f"  Shimmy wrapper: obs_shape={obs.shape}, obs_type={type(obs)}")

        # Test action
        action = gym_env.action_space.sample()
        obs, reward, terminated, truncated, info = gym_env.step(action)
        print(f"  Action executed: reward={reward}, obs_shape={obs.shape}")

        print("  ✓ Shimmy wrapper test passed")
        return True

    except Exception as e:
        print(f"  ✗ Shimmy wrapper test failed: {e}")
        return False


def test_gymnasium_registration():
    """Test if dm_control environments are registered with gymnasium."""
    print("\nTesting gymnasium dm_control registration...")

    try:
        # Check available dm_control environments
        envs = [env_id for env_id in gym.envs.registration.registry.keys() if 'dm_control' in env_id.lower()]
        print(f"  Found dm_control envs: {len(envs)}")
        if envs:
            print(f"  Examples: {envs[:3]}")

        # Try manual registration if needed
        if not envs:
            print("  No pre-registered dm_control envs, will need manual wrapper")

        return True

    except Exception as e:
        print(f"  ✗ Gymnasium registration test failed: {e}")
        return False


def create_cheetah_env():
    """Create cheetah_run environment using best available method."""
    print("\nCreating cheetah_run environment...")

    try:
        # Method 1: Direct shimmy wrapper (most reliable)
        from shimmy.dm_control_compatibility import DmControlCompatibilityV0

        dm_env = suite.load(domain_name="cheetah", task_name="run")
        gym_env = DmControlCompatibilityV0(dm_env)

        print(f"  Created via shimmy wrapper")
        print(f"  Observation space: {gym_env.observation_space}")
        print(f"  Action space: {gym_env.action_space}")

        # Test episode
        obs, info = gym_env.reset()
        for i in range(5):
            action = gym_env.action_space.sample()
            obs, reward, terminated, truncated, info = gym_env.step(action)
            print(f"  Step {i}: reward={reward:.3f}")

            if terminated or truncated:
                obs, info = gym_env.reset()

        gym_env.close()
        print("  ✓ Cheetah environment test completed successfully")
        return True

    except Exception as e:
        print(f"  ✗ Cheetah environment creation failed: {e}")
        return False


def main():
    """Run all dm_control integration tests."""
    print("dm_control Integration Test Suite")
    print("=" * 40)

    tests = [
        test_dm_control_basic,
        test_shimmy_wrapper,
        test_gymnasium_registration,
        create_cheetah_env
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\nTest Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("✓ dm_control integration ready for scheduler testing")
    else:
        print("✗ Some dm_control tests failed - check dependencies")

    return all(results)


if __name__ == "__main__":
    main()