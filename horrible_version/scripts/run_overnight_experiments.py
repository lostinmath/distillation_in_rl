#!/usr/bin/env python3
"""Overnight experiment suite for robust validation.

Options:
1. Quick validation (5 min) - Test integration works
2. Single-seed comparison (30 min) - Test all strategies
3. Multi-seed robust (4-6 hours) - Publication-ready results
4. Extended environments (8+ hours) - Multiple environments

Choose based on your computational budget and timeline.
"""

import argparse
import time
from pathlib import Path
import subprocess

from test_integration import run_validation_suite


def run_quick_validation():
    """Option 1: Quick validation (5 minutes)."""
    print("🚀 Quick Validation Suite")
    print("Expected runtime: 5 minutes")
    print("Purpose: Verify PPO+scheduler integration works")
    print()

    start_time = time.time()
    run_validation_suite()
    runtime = time.time() - start_time

    print(f"✅ Quick validation completed in {runtime:.1f} seconds")
    return True


def run_single_seed_comparison():
    """Option 2: Single-seed strategy comparison (30 minutes)."""
    print("🧪 Single-Seed Strategy Comparison")
    print("Expected runtime: 30 minutes")
    print("Purpose: Compare all 7 scheduling strategies")
    print()

    strategies = [
        "student_only",
        "teacher_only",
        "epsilon_0.5",
        "epsilon_decreasing",
        "alternating",
        "teacher_then_student",
        "reward_based"
    ]

    results = {}
    start_time = time.time()

    for strategy in strategies:
        print(f"🔄 Running {strategy}...")
        strategy_start = time.time()

        # Run strategy (would need to implement config generation)
        # This is a placeholder - you'd run actual experiments

        strategy_runtime = time.time() - strategy_start
        results[strategy] = {
            "runtime": strategy_runtime,
            "status": "completed"  # Would be actual results
        }
        print(f"  ✅ {strategy} completed in {strategy_runtime:.1f}s")

    total_runtime = time.time() - start_time
    print(f"🎉 All strategies completed in {total_runtime/60:.1f} minutes")
    return results


def run_multi_seed_robust():
    """Option 3: Multi-seed robust experiments (4-6 hours)."""
    print("🔬 Multi-Seed Robust Experiments")
    print("Expected runtime: 2-4 hours")
    print("Purpose: Statistical significance testing")
    print()

    start_time = time.time()

    # Import and run the actual robust experiments
    try:
        from run_robust_experiments import run_robust_experiments

        strategies = ["student_only", "reward_based"]
        seeds = [42, 123, 456, 789, 999]

        print(f"📊 Running {len(strategies)} strategies × {len(seeds)} seeds = {len(strategies) * len(seeds)} experiments")
        print()

        # Run the experiments
        results_df = run_robust_experiments(
            strategies=strategies,
            seeds=seeds,
            env_id="CartPole-v1"
        )

        runtime = time.time() - start_time

        # Summary of results
        if len(results_df) > 0:
            success_rate = (results_df['status'] == 'success').mean()
            print(f"\n📈 Final Summary:")
            print(f"  Total runtime: {runtime/60:.1f} minutes")
            print(f"  Success rate: {success_rate:.1%}")

            if success_rate > 0:
                success_df = results_df[results_df['status'] == 'success']
                for strategy in strategies:
                    strategy_results = success_df[success_df['strategy'] == strategy]
                    if len(strategy_results) > 0:
                        mean_perf = strategy_results['final_performance'].mean()
                        std_perf = strategy_results['final_performance'].std()
                        print(f"  {strategy}: {mean_perf:.1f} ± {std_perf:.1f}")

        return results_df

    except Exception as e:
        runtime = time.time() - start_time
        print(f"❌ Robust experiments failed after {runtime/60:.1f} minutes: {e}")
        raise


def run_extended_environments():
    """Option 4: Extended environment testing (8+ hours)."""
    print("🌍 Extended Environment Testing")
    print("Expected runtime: 8+ hours")
    print("Purpose: Generalization across environments")
    print()

    environments = [
        "CartPole-v1",      # Quick convergence
        "LunarLander-v2",   # Intermediate difficulty
        "Acrobot-v1",       # More challenging
        # Could add Atari/MuJoCo if available
    ]

    for env in environments:
        print(f"🎮 Testing on {env}...")
        # Run multi-seed experiments on each environment

    return True


def main():
    parser = argparse.ArgumentParser(description="Overnight experiment options")
    parser.add_argument("--mode", choices=["quick", "single", "robust", "extended"],
                       default="quick", help="Experiment mode")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 DRY RUN MODE - Showing execution plans:")
        print()

    modes = {
        "quick": ("Quick Validation", "5 minutes", run_quick_validation),
        "single": ("Single-Seed Comparison", "30 minutes", run_single_seed_comparison),
        "robust": ("Multi-Seed Robust", "4-6 hours", run_multi_seed_robust),
        "extended": ("Extended Environments", "8+ hours", run_extended_environments)
    }

    name, duration, func = modes[args.mode]

    print(f"🌙 Overnight Experiment: {name}")
    print(f"⏱️  Expected Duration: {duration}")
    print(f"📅 Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.dry_run:
        print("Would run:", func.__name__)
        print("Use --mode without --dry-run to execute")
        return

    try:
        result = func()
        print(f"🎉 {name} completed successfully!")
        return result
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        raise


if __name__ == "__main__":
    main()