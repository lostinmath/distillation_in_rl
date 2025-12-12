#!/usr/bin/env python3
"""
Run proper scientific experiments with learning curves to convergence.
"""

import json
import pickle
import time
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt

def run_experiment_with_curves(strategy: str, total_timesteps: int = 80000, seed: int = 42):
    """Run single experiment and return learning curve data."""
    print(f"Running {strategy} for {total_timesteps} steps (seed={seed})")

    # Import and run directly to capture return data
    import os
    os.environ['PYTHONPATH'] = 'src'

    sys.path.insert(0, 'src')
    from train_real import main as train_main, parse_args, get_config_from_args, setup_experiment, run_training

    # Mock command line args
    class MockArgs:
        config_path = None
        config_name = strategy

    args = MockArgs()
    overrides = {'total_timesteps': total_timesteps, 'seed': seed}

    config = get_config_from_args(args, overrides)
    env, ppo, teacher, scheduler, logger = setup_experiment(config)

    results = run_training(env, ppo, teacher, scheduler, logger, config)

    # Cleanup
    logger.close()
    env.close()

    return {
        'strategy': strategy,
        'seed': seed,
        'total_timesteps': total_timesteps,
        'learning_curve': results['learning_curve'],
        'eval_rewards': results['eval_rewards'],
        'final_performance': results['final_performance'],
        'training_time': results['training_time']
    }

def main():
    """Run proper experiments for learning curves."""
    results_dir = Path("results/learning_curves")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Experiment configuration
    strategies = ['student_only_cartpole', 'reward_based_cartpole', 'teacher_only_cartpole']
    timesteps_per_strategy = {
        'student_only_cartpole': 100000,  # Needs more time to converge
        'reward_based_cartpole': 80000,   # Should converge faster
        'teacher_only_cartpole': 40000    # Already optimal, just for comparison
    }

    all_results = []

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"RUNNING {strategy.upper()}")
        print(f"{'='*60}")

        total_timesteps = timesteps_per_strategy[strategy]

        try:
            # Run 2 seeds for basic statistics
            for seed in [42, 142]:
                result = run_experiment_with_curves(strategy, total_timesteps, seed)
                all_results.append(result)

                # Save individual result
                result_file = results_dir / f"{strategy}_seed_{seed}_curves.pkl"
                with open(result_file, 'wb') as f:
                    pickle.dump(result, f)

                print(f"  Seed {seed}: Final performance = {result['final_performance']:.2f}")

        except Exception as e:
            print(f"Error running {strategy}: {e}")
            continue

    # Save all results
    all_results_file = results_dir / "all_learning_curves.pkl"
    with open(all_results_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    for strategy in strategies:
        strategy_results = [r for r in all_results if r['strategy'] == strategy]
        if strategy_results:
            performances = [r['final_performance'] for r in strategy_results]
            mean_perf = np.mean(performances)
            std_perf = np.std(performances)
            print(f"{strategy}: {mean_perf:.2f} ± {std_perf:.2f} (n={len(strategy_results)})")

    return all_results

if __name__ == "__main__":
    results = main()