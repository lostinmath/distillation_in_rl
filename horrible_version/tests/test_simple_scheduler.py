#!/usr/bin/env python3
"""Simple test to validate reward-based scheduler logic."""

import torch
from src.adaptive_rl.schedulers.reward_based import RewardBasedScheduler


def test_reward_based_logic():
    """Test that reward-based scheduler switches when rewards decrease."""
    print("🧪 Testing Reward-Based Scheduler Logic")

    # Create scheduler for 1 environment
    scheduler = RewardBasedScheduler(num_envs=1, trust_period=3)

    # Simulate episode sequence
    test_cases = [
        # (iteration, global_step, steps_since_reset, prev_reward, expected_policy, description)
        (1, 100, torch.tensor([0]), torch.tensor([-1.0]), "teacher", "Reset: start with teacher"),
        (2, 200, torch.tensor([1]), torch.tensor([100.0]), "teacher", "Trust period: stay teacher"),
        (3, 300, torch.tensor([2]), torch.tensor([120.0]), "teacher", "Trust period: stay teacher"),
        (4, 400, torch.tensor([3]), torch.tensor([150.0]), "teacher", "Trust period: stay teacher"),
        (5, 500, torch.tensor([4]), torch.tensor([140.0]), "student", "Reward decreased: switch to student"),
        (6, 600, torch.tensor([5]), torch.tensor([80.0]), "student", "Trust period: stay student"),
        (7, 700, torch.tensor([6]), torch.tensor([90.0]), "student", "Trust period: stay student"),
        (8, 800, torch.tensor([7]), torch.tensor([100.0]), "student", "Trust period: stay student"),
        (9, 900, torch.tensor([8]), torch.tensor([70.0]), "teacher", "Reward decreased: switch to teacher"),
    ]

    for iteration, global_step, steps_since_reset, prev_reward, expected, description in test_cases:
        policies = scheduler.choose_policy_type(
            iteration=iteration,
            global_step=global_step,
            steps_since_reset=steps_since_reset,
            prev_reward=prev_reward
        )

        actual = policies[0]
        status = "✅" if actual == expected else "❌"

        print(f"  {status} Step {iteration}: {description}")
        print(f"      Reward: {prev_reward[0]:.1f}, Expected: {expected}, Got: {actual}")

        if actual != expected:
            print(f"      ❌ FAIL: Expected {expected}, got {actual}")
            return False

    print("  🎉 All scheduler logic tests passed!")
    return True


def test_scheduler_state_tracking():
    """Test that scheduler properly tracks state across calls."""
    print("\n🧪 Testing Scheduler State Tracking")

    scheduler = RewardBasedScheduler(num_envs=2, trust_period=2)

    # Test parallel environments with different reward patterns
    test_sequence = [
        # (prev_rewards, expected_policies, description)
        (torch.tensor([-1.0, -1.0]), ["teacher", "teacher"], "Both reset"),
        (torch.tensor([100.0, 50.0]), ["teacher", "teacher"], "Both in trust period"),
        (torch.tensor([120.0, 60.0]), ["teacher", "teacher"], "Both still in trust period"),
        (torch.tensor([90.0, 80.0]), ["student", "student"], "Both rewards decreased -> switch"),
        (torch.tensor([95.0, 70.0]), ["student", "student"], "Both in new trust period"),
        (torch.tensor([100.0, 60.0]), ["student", "student"], "Still in trust period"),
        (torch.tensor([80.0, 90.0]), ["teacher", "teacher"], "Env0: decreased, Env1: increased -> both switch"),
    ]

    for i, (prev_rewards, expected_policies, description) in enumerate(test_sequence):
        policies = scheduler.choose_policy_type(
            iteration=i+1,
            global_step=(i+1)*100,
            steps_since_reset=torch.tensor([i, i]),
            prev_reward=prev_rewards
        )

        success = policies == expected_policies
        status = "✅" if success else "❌"

        print(f"  {status} Step {i+1}: {description}")
        print(f"      Rewards: {prev_rewards.tolist()}, Expected: {expected_policies}, Got: {policies}")

        if not success:
            print(f"      ❌ FAIL: Expected {expected_policies}, got {policies}")
            return False

    print("  🎉 All state tracking tests passed!")
    return True


if __name__ == "__main__":
    print("🚀 Reward-Based Scheduler Unit Tests")
    print("=" * 50)

    success1 = test_reward_based_logic()
    success2 = test_scheduler_state_tracking()

    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Reward-based scheduler logic is working correctly")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("🔧 Need to fix scheduler implementation")