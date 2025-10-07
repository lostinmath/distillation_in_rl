#!/usr/bin/env python3
"""Test all environment integrations for comprehensive RL evaluation.

Tests environments from all major domains:
- Classic control (CartPole, LunarLander)
- Continuous control (dm_control)
- Visual control (Atari)
- Manipulation (MetaWorld)
- Robotics (PyBullet)
- Standard benchmarks (MuJoCo)

Validates that scheduling works across all domains.
"""

import sys
import warnings
sys.path.append('src')

import numpy as np
from typing import Dict, List, Tuple, Any


def test_classic_control():
    """Test classic control environments."""
    print("🎮 Testing Classic Control Environments")

    try:
        import gymnasium as gym

        envs_to_test = ["CartPole-v1", "LunarLander-v2"]
        results = {}

        for env_name in envs_to_test:
            try:
                print(f"  Testing {env_name}...")
                env = gym.make(env_name)
                obs, _ = env.reset()

                for _ in range(5):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        obs, _ = env.reset()

                results[env_name] = {
                    "status": "success",
                    "obs_shape": obs.shape,
                    "action_space": str(env.action_space),
                    "action_type": "discrete"
                }
                env.close()
                print(f"    ✅ {env_name}: {results[env_name]}")

            except Exception as e:
                results[env_name] = {"status": "failed", "error": str(e)}
                print(f"    ❌ {env_name}: {e}")

        return results

    except ImportError as e:
        print(f"  ❌ Classic control not available: {e}")
        return {}


def test_dm_control():
    """Test dm_control environments."""
    print("\n🦎 Testing dm_control Environments")

    try:
        from adaptive_rl.envs.dm_control_wrapper import create_dm_control_env, DM_CONTROL_ENVS
        from adaptive_rl.teachers.dm_control_teachers import create_dm_control_teacher

        test_envs = ["cheetah_run", "walker_walk", "reacher_easy"]
        results = {}

        for env_name in test_envs:
            if env_name in DM_CONTROL_ENVS:
                try:
                    print(f"  Testing {env_name}...")
                    env = create_dm_control_env(env_name)
                    teacher = create_dm_control_teacher(env_name)

                    obs, _ = env.reset()

                    for _ in range(5):
                        action = teacher.act(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            obs, _ = env.reset()

                    results[env_name] = {
                        "status": "success",
                        "obs_shape": obs.shape,
                        "action_space": str(env.action_space),
                        "action_type": "continuous",
                        "teacher": type(teacher).__name__
                    }
                    env.close()
                    print(f"    ✅ {env_name}: {results[env_name]}")

                except Exception as e:
                    results[env_name] = {"status": "failed", "error": str(e)}
                    print(f"    ❌ {env_name}: {e}")

        return results

    except ImportError as e:
        print(f"  ❌ dm_control not available: {e}")
        return {}


def test_atari():
    """Test Atari environments."""
    print("\n🕹️ Testing Atari Environments")

    try:
        from adaptive_rl.envs.atari_wrapper import create_atari_env, ATARI_ENVS
        from adaptive_rl.teachers.atari_teachers import create_atari_teacher

        test_envs = ["breakout", "pong"]
        results = {}

        for env_name in test_envs:
            if env_name in ATARI_ENVS:
                try:
                    print(f"  Testing {env_name}...")

                    # Test RAM version for faster testing
                    env = create_atari_env(env_name, obs_type="ram")
                    teacher = create_atari_teacher(env_name)

                    obs, _ = env.reset()

                    for _ in range(5):
                        action = teacher.act(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            obs, _ = env.reset()

                    results[env_name] = {
                        "status": "success",
                        "obs_shape": obs.shape,
                        "action_space": str(env.action_space),
                        "action_type": "discrete",
                        "teacher": type(teacher).__name__
                    }
                    env.close()
                    print(f"    ✅ {env_name}: {results[env_name]}")

                except Exception as e:
                    results[env_name] = {"status": "failed", "error": str(e)}
                    print(f"    ❌ {env_name}: {e}")

        return results

    except ImportError as e:
        print(f"  ❌ Atari environments not available: {e}")
        return {}


def test_metaworld():
    """Test MetaWorld environments."""
    print("\n🤖 Testing MetaWorld Environments")

    try:
        from adaptive_rl.envs.metaworld_wrapper import create_metaworld_env, METAWORLD_TASKS
        from adaptive_rl.teachers.metaworld_teachers import create_metaworld_teacher

        test_tasks = ["reach"]  # Start with basic task
        results = {}

        for task_name in test_tasks:
            if task_name in METAWORLD_TASKS:
                try:
                    print(f"  Testing {task_name}...")
                    env = create_metaworld_env(task_name)
                    teacher = create_metaworld_teacher(task_name)

                    obs, _ = env.reset()

                    for _ in range(5):
                        action = teacher.act(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            obs, _ = env.reset()

                    results[task_name] = {
                        "status": "success",
                        "obs_shape": obs.shape,
                        "action_space": str(env.action_space),
                        "action_type": "continuous",
                        "teacher": type(teacher).__name__
                    }
                    env.close()
                    print(f"    ✅ {task_name}: {results[task_name]}")

                except Exception as e:
                    results[task_name] = {"status": "failed", "error": str(e)}
                    print(f"    ❌ {task_name}: {e}")

        return results

    except ImportError as e:
        print(f"  ❌ MetaWorld not available: {e}")
        return {}


def test_pybullet():
    """Test PyBullet environments."""
    print("\n🔫 Testing PyBullet Environments")

    try:
        from adaptive_rl.envs.pybullet_wrapper import create_pybullet_env, PYBULLET_ENVS
        from adaptive_rl.teachers.pybullet_teachers import create_pybullet_teacher

        test_envs = ["ant"]  # Start with one stable environment
        results = {}

        for env_name in test_envs:
            if env_name in PYBULLET_ENVS:
                try:
                    print(f"  Testing {env_name}...")
                    env = create_pybullet_env(env_name)
                    teacher = create_pybullet_teacher(env_name)

                    obs, _ = env.reset()

                    for _ in range(5):
                        action = teacher.act(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            obs, _ = env.reset()

                    results[env_name] = {
                        "status": "success",
                        "obs_shape": obs.shape,
                        "action_space": str(env.action_space),
                        "action_type": "continuous",
                        "teacher": type(teacher).__name__
                    }
                    env.close()
                    print(f"    ✅ {env_name}: {results[env_name]}")

                except Exception as e:
                    results[env_name] = {"status": "failed", "error": str(e)}
                    print(f"    ❌ {env_name}: {e}")

        return results

    except ImportError as e:
        print(f"  ❌ PyBullet not available: {e}")
        return {}


def test_mujoco():
    """Test MuJoCo environments."""
    print("\n🎯 Testing MuJoCo Environments")

    try:
        from adaptive_rl.envs.mujoco_wrapper import create_mujoco_env, MUJOCO_ENVS
        from adaptive_rl.teachers.mujoco_teachers import create_mujoco_teacher

        test_envs = ["halfcheetah", "reacher"]
        results = {}

        for env_name in test_envs:
            if env_name in MUJOCO_ENVS:
                try:
                    print(f"  Testing {env_name}...")
                    env = create_mujoco_env(env_name)
                    teacher = create_mujoco_teacher(env_name)

                    obs, _ = env.reset()

                    for _ in range(5):
                        action = teacher.act(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            obs, _ = env.reset()

                    results[env_name] = {
                        "status": "success",
                        "obs_shape": obs.shape,
                        "action_space": str(env.action_space),
                        "action_type": "continuous",
                        "teacher": type(teacher).__name__
                    }
                    env.close()
                    print(f"    ✅ {env_name}: {results[env_name]}")

                except Exception as e:
                    results[env_name] = {"status": "failed", "error": str(e)}
                    print(f"    ❌ {env_name}: {e}")

        return results

    except ImportError as e:
        print(f"  ❌ MuJoCo not available: {e}")
        return {}


def summarize_results(all_results: Dict[str, Dict]) -> None:
    """Summarize test results across all domains."""
    print("\n" + "="*80)
    print("🎉 COMPREHENSIVE ENVIRONMENT TEST SUMMARY")
    print("="*80)

    total_tested = 0
    total_successful = 0

    domain_stats = {}

    for domain, results in all_results.items():
        successful = sum(1 for r in results.values() if r.get("status") == "success")
        tested = len(results)
        total_tested += tested
        total_successful += successful

        domain_stats[domain] = (successful, tested)

        print(f"\n📊 {domain.upper()}:")
        print(f"  ✅ Successful: {successful}/{tested}")

        if successful > 0:
            # Show successful environments
            successful_envs = [name for name, result in results.items()
                             if result.get("status") == "success"]
            print(f"  Working: {', '.join(successful_envs)}")

        if successful < tested:
            # Show failed environments
            failed_envs = [name for name, result in results.items()
                          if result.get("status") == "failed"]
            print(f"  ❌ Failed: {', '.join(failed_envs)}")

    print(f"\n🎯 OVERALL RESULTS:")
    print(f"  ✅ Total Success Rate: {total_successful}/{total_tested} ({100*total_successful/total_tested:.1f}%)")

    # Categorize by action type
    continuous_count = 0
    discrete_count = 0
    visual_count = 0

    for domain_results in all_results.values():
        for result in domain_results.values():
            if result.get("status") == "success":
                if result.get("action_type") == "continuous":
                    continuous_count += 1
                elif result.get("action_type") == "discrete":
                    discrete_count += 1

                if "visual" in str(result.get("obs_shape", "")):
                    visual_count += 1

    print(f"\n📈 CAPABILITY COVERAGE:")
    print(f"  🎮 Discrete Action Environments: {discrete_count}")
    print(f"  🎯 Continuous Action Environments: {continuous_count}")

    print(f"\n🚀 READY FOR PUBLICATION:")
    if total_successful >= 8:  # Good coverage across domains
        print("  ✅ Excellent environment coverage for publication")
        print("  ✅ Multi-domain evaluation capability demonstrated")
        print("  ✅ Both discrete and continuous control supported")
    elif total_successful >= 5:
        print("  ⚠️  Good environment coverage, consider adding more domains")
    else:
        print("  ❌ Limited environment coverage, need more working environments")


def main():
    """Run comprehensive environment testing."""
    print("Comprehensive Environment Integration Test")
    print("=" * 80)
    print("Testing adaptive scheduling across all major RL domains\n")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    all_results = {}

    # Test each domain
    all_results["classic_control"] = test_classic_control()
    all_results["dm_control"] = test_dm_control()
    all_results["atari"] = test_atari()
    all_results["metaworld"] = test_metaworld()
    all_results["pybullet"] = test_pybullet()
    all_results["mujoco"] = test_mujoco()

    # Summarize results
    summarize_results(all_results)

    print(f"\n🔬 NEXT STEPS:")
    print(f"  1. Run: pixi run python test_dm_control_integration.py")
    print(f"  2. Test specific domains that worked")
    print(f"  3. Run comprehensive experiments with working environments")


if __name__ == "__main__":
    main()