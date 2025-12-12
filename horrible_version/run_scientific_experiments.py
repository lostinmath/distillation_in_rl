#!/usr/bin/env python3
"""
Scientific experimental pipeline for evaluating adaptive RL strategies.

This script runs controlled experiments comparing different teacher-student
scheduling strategies and collects comprehensive metrics for analysis.
"""

import json
import time
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Any
import numpy as np

# Experimental configuration
STRATEGIES = [
    'student_only_cartpole',
    'teacher_only_cartpole',
    'reward_based_cartpole'
]

RUNS_PER_STRATEGY = 5
TOTAL_TIMESTEPS = 20000
RESULTS_DIR = Path("results/scientific_study")

def run_single_experiment(strategy: str, run_id: int, seed: int) -> Dict[str, Any]:
    """Run a single experiment and extract metrics."""
    print(f"Running {strategy} - Run {run_id+1}/{RUNS_PER_STRATEGY} (seed={seed})")

    # Run the experiment
    cmd = [
        "pixi", "run", "python", "train_real.py",
        "--config-name", strategy,
        f"total_timesteps={TOTAL_TIMESTEPS}",
        f"seed={seed}"
    ]

    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={"PATH": "/home/piscenco/.pixi/bin:" + sys.path[0], "PYTHONPATH": "src"}
    )
    end_time = time.time()

    if result.returncode != 0:
        print(f"Error running experiment: {result.stderr}")
        return None

    # Parse output for metrics
    output_lines = result.stdout.split('\n')

    # Extract final performance
    final_performance = None
    eval_reward = None
    training_time = end_time - start_time

    for line in output_lines:
        if "Final performance:" in line:
            final_performance = float(line.split(": ")[1])
        elif "Eval reward =" in line:
            eval_reward = float(line.split("= ")[1])

    # Count episodes (approximate from log entries)
    episode_count = len([line for line in output_lines if "episode_reward_env_" in line])

    return {
        'strategy': strategy,
        'run_id': run_id,
        'seed': seed,
        'final_performance': final_performance,
        'eval_reward': eval_reward,
        'training_time': training_time,
        'total_timesteps': TOTAL_TIMESTEPS,
        'episode_count': episode_count,
        'stdout': result.stdout,
        'stderr': result.stderr
    }

def collect_all_experiments() -> List[Dict[str, Any]]:
    """Run all experiments systematically."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for strategy in STRATEGIES:
        strategy_results = []

        for run_id in range(RUNS_PER_STRATEGY):
            # Use different seeds for reproducibility
            seed = 42 + run_id * 100

            result = run_single_experiment(strategy, run_id, seed)
            if result:
                strategy_results.append(result)
                all_results.append(result)

                # Save individual result
                result_file = RESULTS_DIR / f"{strategy}_run_{run_id}_seed_{seed}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)

        # Save strategy summary
        strategy_file = RESULTS_DIR / f"{strategy}_summary.json"
        with open(strategy_file, 'w') as f:
            json.dump(strategy_results, f, indent=2)

        print(f"Completed {strategy}: {len(strategy_results)} successful runs")

    return all_results

def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute statistical summary for each strategy."""
    stats = {}

    for strategy in STRATEGIES:
        strategy_results = [r for r in results if r['strategy'] == strategy]

        if not strategy_results:
            continue

        performances = [r['final_performance'] for r in strategy_results if r['final_performance'] is not None]
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
                'confidence_interval_95': 1.96 * np.std(performances) / np.sqrt(len(performances))
            }

    return stats

def main():
    """Run the complete scientific experimental pipeline."""
    print("=" * 60)
    print("SCIENTIFIC EXPERIMENTAL PIPELINE")
    print("=" * 60)
    print(f"Strategies: {STRATEGIES}")
    print(f"Runs per strategy: {RUNS_PER_STRATEGY}")
    print(f"Total timesteps per run: {TOTAL_TIMESTEPS}")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 60)

    # Run all experiments
    start_time = time.time()
    all_results = collect_all_experiments()
    total_time = time.time() - start_time

    # Compute statistics
    stats = compute_statistics(all_results)

    # Save complete results
    complete_results = {
        'experiment_config': {
            'strategies': STRATEGIES,
            'runs_per_strategy': RUNS_PER_STRATEGY,
            'total_timesteps': TOTAL_TIMESTEPS,
            'total_experiments': len(all_results),
            'total_time_seconds': total_time
        },
        'raw_results': all_results,
        'statistics': stats
    }

    results_file = RESULTS_DIR / "complete_results.json"
    with open(results_file, 'w') as f:
        json.dump(complete_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 60)

    for strategy, stat in stats.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Runs: {stat['n_runs']}")
        print(f"  Mean performance: {stat['mean_performance']:.2f} ± {stat['std_performance']:.2f}")
        print(f"  95% CI: ±{stat['confidence_interval_95']:.2f}")
        print(f"  Range: [{stat['min_performance']:.2f}, {stat['max_performance']:.2f}]")
        print(f"  Training time: {stat['mean_training_time']:.1f}s ± {stat['std_training_time']:.1f}s")

    print(f"\nTotal experiment time: {total_time:.1f}s")
    print(f"Results saved to: {results_file}")

    return complete_results

if __name__ == "__main__":
    results = main()