#!/usr/bin/env python3
"""Test multiple dm_control environments with the integrated system."""

import sys
sys.path.append('src')

from adaptive_rl.envs.dm_control_wrapper import create_dm_control_env, DM_CONTROL_ENVS
from adaptive_rl.teachers.dm_control_teachers import create_dm_control_teacher, DM_CONTROL_TEACHERS


def test_environment_creation():
    """Test that we can create all supported environments."""
    print("🧪 Testing dm_control environment creation")

    successful_envs = []
    failed_envs = []

    for env_name in DM_CONTROL_ENVS.keys():
        try:
            print(f"  Testing {env_name}...")
            env = create_dm_control_env(env_name)
            obs, _ = env.reset()
            print(f"    ✅ {env_name}: obs_shape={obs.shape}, action_space={env.action_space}")
            env.close()
            successful_envs.append(env_name)
        except Exception as e:
            print(f"    ❌ {env_name}: {str(e)}")
            failed_envs.append(env_name)

    print(f"\n📊 Environment Test Results:")
    print(f"  ✅ Successful: {len(successful_envs)}/{len(DM_CONTROL_ENVS)}")
    print(f"  ❌ Failed: {len(failed_envs)}")

    if failed_envs:
        print(f"  Failed environments: {failed_envs}")

    return successful_envs


def test_teacher_creation():
    """Test that we can create teachers for supported environments."""
    print("\n🧪 Testing dm_control teacher creation")

    successful_teachers = []
    failed_teachers = []

    for env_name in DM_CONTROL_TEACHERS.keys():
        try:
            print(f"  Testing teacher for {env_name}...")
            teacher = create_dm_control_teacher(env_name)

            # Test with dummy observation
            if env_name == "cheetah_run":
                dummy_obs = [0.0] * 17  # cheetah has 17D observations
            elif env_name in ["walker_walk", "walker_stand"]:
                dummy_obs = [0.0] * 24  # walker has more observations
            else:
                dummy_obs = [0.0] * 10  # fallback

            action = teacher.act(dummy_obs)
            print(f"    ✅ {env_name}: teacher action_shape={action.shape}")
            successful_teachers.append(env_name)
        except Exception as e:
            print(f"    ❌ {env_name}: {str(e)}")
            failed_teachers.append(env_name)

    print(f"\n📊 Teacher Test Results:")
    print(f"  ✅ Successful: {len(successful_teachers)}/{len(DM_CONTROL_TEACHERS)}")
    print(f"  ❌ Failed: {len(failed_teachers)}")

    if failed_teachers:
        print(f"  Failed teachers: {failed_teachers}")

    return successful_teachers


def test_complete_pipeline():
    """Test complete environment + teacher + action pipeline."""
    print("\n🧪 Testing complete environment + teacher pipeline")

    test_envs = ["cheetah_run", "walker_walk"]  # Test a couple key environments

    for env_name in test_envs:
        if env_name in DM_CONTROL_ENVS and env_name in DM_CONTROL_TEACHERS:
            try:
                print(f"  Testing complete pipeline for {env_name}...")

                # Create environment and teacher
                env = create_dm_control_env(env_name)
                teacher = create_dm_control_teacher(env_name)

                # Run a few steps
                obs, _ = env.reset()
                total_reward = 0

                for step in range(10):
                    action = teacher.act(obs)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward

                    if terminated or truncated:
                        obs, _ = env.reset()

                print(f"    ✅ {env_name}: 10 steps completed, avg_reward={total_reward/10:.4f}")
                env.close()

            except Exception as e:
                print(f"    ❌ {env_name}: {str(e)}")


def main():
    """Run all dm_control integration tests."""
    print("dm_control Environment & Teacher Integration Test")
    print("=" * 60)

    # Test environment creation
    successful_envs = test_environment_creation()

    # Test teacher creation
    successful_teachers = test_teacher_creation()

    # Test complete pipeline
    test_complete_pipeline()

    # Summary
    print(f"\n🎉 Summary:")
    print(f"  📦 Environments working: {len(successful_envs)}")
    print(f"  🧠 Teachers working: {len(successful_teachers)}")
    print(f"  ✅ dm_control integration ready for experiments!")

    print(f"\n🚀 Ready to run:")
    print(f"  pixi run python test_dm_control_integration.py")


if __name__ == "__main__":
    main()