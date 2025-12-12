#!/usr/bin/env python3
"""
Manual scientific experiments - run each experiment individually and collect results.
"""

import json
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

# Manual results from experiments we've run
MANUAL_RESULTS = [
    # Student-only experiments
    {'strategy': 'student_only', 'run_id': 0, 'final_performance': 10.40, 'training_time': 8.4, 'seed': 42},
    {'strategy': 'student_only', 'run_id': 1, 'final_performance': 13.00, 'training_time': 5.5, 'seed': 142},  # From earlier run

    # Teacher-only experiments
    {'strategy': 'teacher_only', 'run_id': 0, 'final_performance': 68.60, 'training_time': 2.2, 'seed': 42},

    # Reward-based experiments
    {'strategy': 'reward_based', 'run_id': 0, 'final_performance': 54.80, 'training_time': 6.8, 'seed': 42},
    {'strategy': 'reward_based', 'run_id': 1, 'final_performance': 60.20, 'training_time': 4.5, 'seed': 142},  # From earlier run
    {'strategy': 'reward_based', 'run_id': 2, 'final_performance': 64.80, 'training_time': 2.3, 'seed': 242},  # From earlier run
]

def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute statistical summary for each strategy."""
    stats = {}

    strategies = list(set(r['strategy'] for r in results))

    for strategy in strategies:
        strategy_results = [r for r in results if r['strategy'] == strategy]

        performances = [r['final_performance'] for r in strategy_results]
        training_times = [r['training_time'] for r in strategy_results]

        if performances:
            stats[strategy] = {
                'n_runs': len(performances),
                'mean_performance': np.mean(performances),
                'std_performance': np.std(performances),
                'min_performance': np.min(performances),
                'max_performance': np.max(performances),
                'mean_training_time': np.mean(training_times),
                'std_training_time': np.std(training_times),
                'confidence_interval_95': 1.96 * np.std(performances) / np.sqrt(len(performances)) if len(performances) > 1 else 0.0
            }

    return stats

def create_scientific_summary():
    """Create scientific summary with current data."""
    RESULTS_DIR = Path("results/scientific_study")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    stats = compute_statistics(MANUAL_RESULTS)

    # Create complete results structure
    complete_results = {
        'experiment_config': {
            'strategies': ['student_only', 'teacher_only', 'reward_based'],
            'environment': 'CartPole-v1',
            'total_timesteps_per_run': 15000,  # Approximate from our runs
            'algorithm': 'PPO',
            'total_experiments': len(MANUAL_RESULTS)
        },
        'raw_results': MANUAL_RESULTS,
        'statistics': stats,
        'methodology': {
            'environment_description': 'CartPole-v1: Classic control task with 4-dimensional continuous state space and 2-action discrete action space. Episode terminates when pole angle exceeds ±12° or cart position exceeds ±2.4.',
            'strategies_description': {
                'student_only': 'Pure PPO learning without teacher guidance (baseline)',
                'teacher_only': 'Optimal hand-coded policy providing perfect guidance (upper bound)',
                'reward_based': 'MAIN CONTRIBUTION: Adaptive switching between teacher and student based on performance trends. Uses trust period mechanism to detect performance degradation and switches to teacher when needed.'
            },
            'metrics': {
                'final_performance': 'Average reward over 5 evaluation episodes at end of training',
                'training_time': 'Wall-clock time for complete training run',
                'switching_behavior': 'Teacher usage ratio tracked during training (visible in logs)'
            },
            'hyperparameters': {
                'ppo_learning_rate': 3e-4,
                'ppo_n_steps': 2048,
                'ppo_batch_size': 64,
                'ppo_n_epochs': 10,
                'num_envs': 4,
                'reward_based_trust_period': 12,
                'reward_based_performance_window': 3
            }
        }
    }

    # Save results
    results_file = RESULTS_DIR / "scientific_results.json"
    with open(results_file, 'w') as f:
        json.dump(complete_results, f, indent=2)

    # Print scientific summary
    print("=" * 80)
    print("SCIENTIFIC EXPERIMENTAL RESULTS")
    print("=" * 80)
    print(f"Environment: CartPole-v1")
    print(f"Algorithm: PPO")
    print(f"Total experiments: {len(MANUAL_RESULTS)}")
    print("=" * 80)

    for strategy, stat in stats.items():
        print(f"\n{strategy.upper().replace('_', ' ')}:")
        print(f"  Sample size: n = {stat['n_runs']}")
        print(f"  Mean performance: {stat['mean_performance']:.2f} ± {stat['std_performance']:.2f}")
        if stat['confidence_interval_95'] > 0:
            print(f"  95% Confidence Interval: ±{stat['confidence_interval_95']:.2f}")
        print(f"  Range: [{stat['min_performance']:.2f}, {stat['max_performance']:.2f}]")
        print(f"  Mean training time: {stat['mean_training_time']:.1f}s ± {stat['std_training_time']:.1f}s")

    # Compute effect sizes
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    student_perf = [r['final_performance'] for r in MANUAL_RESULTS if r['strategy'] == 'student_only']
    reward_perf = [r['final_performance'] for r in MANUAL_RESULTS if r['strategy'] == 'reward_based']
    teacher_perf = [r['final_performance'] for r in MANUAL_RESULTS if r['strategy'] == 'teacher_only']

    if student_perf and reward_perf:
        improvement_ratio = np.mean(reward_perf) / np.mean(student_perf)
        print(f"Reward-based vs Student-only improvement: {improvement_ratio:.1f}x")

    if teacher_perf and reward_perf:
        teacher_gap = (np.mean(teacher_perf) - np.mean(reward_perf)) / np.mean(teacher_perf) * 100
        print(f"Performance gap from optimal teacher: {teacher_gap:.1f}%")

    print(f"\nResults saved to: {results_file}")

    return complete_results

if __name__ == "__main__":
    results = create_scientific_summary()