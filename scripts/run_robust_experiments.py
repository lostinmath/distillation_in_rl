#!/usr/bin/env python3
"""Robust multi-seed experiments for statistical validation."""

import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from src.adaptive_rl.core.scheduled_ppo import ScheduledPPOTrainer, ScheduledPPOArgs
from src.adaptive_rl.schedulers.simple import StudentOnlyScheduler, TeacherOnlyScheduler
from src.adaptive_rl.schedulers.reward_based import RewardBasedScheduler
from src.adaptive_rl.teachers.optimal import create_optimal_teacher


def create_experiment_config(strategy: str, seed: int, env_id: str = "CartPole-v1") -> tuple:
    """Create experiment configuration for strategy and seed."""

    base_args = ScheduledPPOArgs(
        exp_name=f"{strategy}_seed{seed}",
        env_id=env_id,
        total_timesteps=100000,  # Reasonable for statistical testing
        num_envs=4,
        seed=seed,
        track=False  # Disable wandb for cleaner runs
    )

    # Create teacher if needed
    teacher = None
    if strategy != "student_only":
        teacher = create_optimal_teacher(env_id)

    # Create scheduler
    if strategy == "student_only":
        scheduler = StudentOnlyScheduler(num_envs=base_args.num_envs)
    elif strategy == "teacher_only":
        scheduler = TeacherOnlyScheduler(num_envs=base_args.num_envs)
    elif strategy == "reward_based":
        scheduler = RewardBasedScheduler(num_envs=base_args.num_envs, trust_period=5)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return base_args, scheduler, teacher


def run_single_experiment(strategy: str, seed: int, exp_num: int, total_exps: int) -> Dict[str, Any]:
    """Run a single experiment and return results."""

    print(f"[{exp_num:2d}/{total_exps:2d}] {strategy}_seed{seed} ", end="", flush=True)

    start_time = time.time()

    try:
        # Create experiment components
        args, scheduler, teacher = create_experiment_config(strategy, seed)

        # Create trainer
        trainer = ScheduledPPOTrainer(args, scheduler, teacher=teacher, writer=None)

        # Run training
        trainer.train()

        # Get results
        metrics = trainer.agent.get_scheduling_metrics()

        runtime = time.time() - start_time

        result = {
            "strategy": strategy,
            "seed": seed,
            "runtime_seconds": runtime,
            "teacher_ratio": metrics.get("teacher_ratio", 0.0),
            "student_ratio": metrics.get("student_ratio", 1.0),
            "policy_switches": metrics.get("policy_switches", 0),
            "total_actions": metrics.get("total_actions", 0),
            "final_episode_rewards": trainer.episode_rewards[-10:] if trainer.episode_rewards else [],
            "status": "success"
        }

        # Calculate final performance
        if trainer.episode_rewards:
            result["final_performance"] = sum(trainer.episode_rewards[-10:]) / len(trainer.episode_rewards[-10:])
        else:
            result["final_performance"] = 0.0

        print(f"DONE {runtime:.0f}s | Perf: {result['final_performance']:.1f} | Teacher: {result['teacher_ratio']:.1%}")

        return result

    except Exception as e:
        runtime = time.time() - start_time
        print(f"FAIL {runtime:.0f}s | Error: {str(e)[:50]}")

        return {
            "strategy": strategy,
            "seed": seed,
            "runtime_seconds": runtime,
            "status": "failed",
            "error": str(e)
        }


def run_robust_experiments(strategies: List[str], seeds: List[int], env_id: str = "CartPole-v1") -> pd.DataFrame:
    """Run multi-seed robust experiments."""

    print("Starting Robust Multi-Seed Experiments")
    print("=" * 50)
    print(f"Environment: {env_id}")
    print(f"Strategies: {strategies}")
    print(f"Seeds: {seeds}")

    total_experiments = len(strategies) * len(seeds)
    print(f"Total experiments: {total_experiments}")
    print()

    results = []
    current_exp = 0
    overall_start = time.time()

    for strategy in strategies:
        for seed in seeds:
            current_exp += 1

            result = run_single_experiment(strategy, seed, current_exp, total_experiments)
            results.append(result)

    overall_runtime = time.time() - overall_start

    print()
    print("EXPERIMENT SUMMARY")
    print("=" * 30)
    print(f"Total runtime: {overall_runtime/60:.1f} minutes")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Success rate
    success_rate = (df['status'] == 'success').mean()
    print(f"Success rate: {success_rate:.1%}")

    if success_rate > 0:
        success_df = df[df['status'] == 'success']

        # Performance by strategy
        print("\nPerformance by Strategy:")
        for strategy in strategies:
            strategy_results = success_df[success_df['strategy'] == strategy]
            if len(strategy_results) > 0:
                mean_perf = strategy_results['final_performance'].mean()
                std_perf = strategy_results['final_performance'].std()
                mean_teacher = strategy_results['teacher_ratio'].mean() if 'teacher_ratio' in strategy_results else 0

                print(f"  {strategy:15s}: {mean_perf:6.1f} ± {std_perf:5.1f} | Teacher: {mean_teacher:.1%}")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"robust_results_{timestamp}.csv"
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Run robust multi-seed experiments")
    parser.add_argument("--strategies", nargs="+",
                       default=["student_only", "reward_based"],
                       help="Strategies to test")
    parser.add_argument("--seeds", nargs="+", type=int,
                       default=[42, 123, 456, 789, 999],
                       help="Seeds for experiments")
    parser.add_argument("--env", default="CartPole-v1",
                       help="Environment to test")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with fewer seeds")

    args = parser.parse_args()

    if args.quick:
        seeds = [42, 123]  # Just 2 seeds for quick testing
        print("Quick mode: Using 2 seeds for faster testing")
    else:
        seeds = args.seeds

    # Run experiments
    results_df = run_robust_experiments(
        strategies=args.strategies,
        seeds=seeds,
        env_id=args.env
    )

    print("\nRobust experiments completed!")


if __name__ == "__main__":
    main()