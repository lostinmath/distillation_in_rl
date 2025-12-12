#!/usr/bin/env python3
"""Hyperparameter search for reward-based scheduling strategy."""

import itertools
import subprocess
import time
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from typing import Dict, List, Tuple
import random


def create_search_space() -> List[Dict]:
    """Create comprehensive hyperparameter search space."""

    # Define parameter ranges
    param_grid = {
        'trust_period': [3, 5, 8, 12, 15, 20, 30],  # Steps before allowing switch
        'performance_window': [1, 2, 3, 4, 5, 7, 10],  # Window for performance comparison
        'switch_threshold': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],  # Minimum performance drop to switch
        'learning_rate': [1e-4, 3e-4, 5e-4, 1e-3],  # PPO learning rate
        'n_steps': [1024, 2048, 4096],  # Rollout length
        'batch_size': [32, 64, 128],  # Batch size
        'total_timesteps': [100000, 150000, 200000],  # Training length
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    search_space = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        search_space.append(params)

    return search_space


def create_focused_search_space() -> List[Dict]:
    """Create smaller, more focused search space for quicker results."""

    param_combinations = [
        # Conservative switching (higher trust periods)
        {'trust_period': 15, 'performance_window': 3, 'switch_threshold': 0.1, 'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'total_timesteps': 150000},
        {'trust_period': 20, 'performance_window': 5, 'switch_threshold': 0.05, 'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'total_timesteps': 150000},
        {'trust_period': 30, 'performance_window': 3, 'switch_threshold': 0.0, 'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'total_timesteps': 150000},

        # Aggressive switching (lower trust periods)
        {'trust_period': 5, 'performance_window': 1, 'switch_threshold': 0.2, 'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'total_timesteps': 150000},
        {'trust_period': 8, 'performance_window': 2, 'switch_threshold': 0.15, 'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'total_timesteps': 150000},
        {'trust_period': 3, 'performance_window': 1, 'switch_threshold': 0.3, 'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'total_timesteps': 150000},

        # Different learning rates
        {'trust_period': 12, 'performance_window': 3, 'switch_threshold': 0.1, 'learning_rate': 1e-4, 'n_steps': 2048, 'batch_size': 64, 'total_timesteps': 150000},
        {'trust_period': 12, 'performance_window': 3, 'switch_threshold': 0.1, 'learning_rate': 5e-4, 'n_steps': 2048, 'batch_size': 64, 'total_timesteps': 150000},
        {'trust_period': 12, 'performance_window': 3, 'switch_threshold': 0.1, 'learning_rate': 1e-3, 'n_steps': 2048, 'batch_size': 64, 'total_timesteps': 150000},

        # Different rollout lengths
        {'trust_period': 12, 'performance_window': 3, 'switch_threshold': 0.1, 'learning_rate': 3e-4, 'n_steps': 1024, 'batch_size': 64, 'total_timesteps': 150000},
        {'trust_period': 12, 'performance_window': 3, 'switch_threshold': 0.1, 'learning_rate': 3e-4, 'n_steps': 4096, 'batch_size': 64, 'total_timesteps': 150000},

        # Different batch sizes
        {'trust_period': 12, 'performance_window': 3, 'switch_threshold': 0.1, 'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 32, 'total_timesteps': 150000},
        {'trust_period': 12, 'performance_window': 3, 'switch_threshold': 0.1, 'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 128, 'total_timesteps': 150000},
    ]

    return param_combinations


def run_single_experiment(params: Dict, experiment_id: int, results_dir: Path) -> Dict:
    """Run a single hyperparameter configuration."""

    start_time = time.time()

    # Create unique experiment name
    exp_name = f"hp_search_{experiment_id:03d}"
    log_file = results_dir / f"{exp_name}.log"

    try:
        # Build command using run_experiment.sh wrapper
        cmd = [
            "../../run_experiment.sh",
            "--config-path", "configs/experiments/comprehensive",
            "--config-name", "cartpole_reward_based",
            f"scheduler=reward_based",
            f"total_timesteps={params['total_timesteps']}",
            f"learning_rate={params['learning_rate']}",
            f"n_steps={params['n_steps']}",
            f"batch_size={params['batch_size']}",
            f"seed={random.randint(1, 9999)}",
        ]

        # Run experiment
        print(f"🚀 Starting experiment {experiment_id} with trust_period={params['trust_period']}, window={params['performance_window']}")

        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            f.write(result.stdout)
            f.write(result.stderr)

        # Extract final performance
        final_performance = 0.0
        if result.returncode == 0:
            # Parse output for final performance
            for line in result.stdout.split('\n'):
                if "Final performance:" in line:
                    try:
                        final_performance = float(line.split(':')[1].strip())
                        break
                    except:
                        pass

        duration = time.time() - start_time

        result_data = {
            'experiment_id': experiment_id,
            'final_performance': final_performance,
            'duration': duration,
            'success': result.returncode == 0,
            **params
        }

        print(f"✅ Completed experiment {experiment_id}: {final_performance:.1f} ({duration:.1f}s)")

        return result_data

    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {experiment_id} timed out")
        return {
            'experiment_id': experiment_id,
            'final_performance': 0.0,
            'duration': 3600,
            'success': False,
            'error': 'timeout',
            **params
        }
    except Exception as e:
        print(f"❌ Experiment {experiment_id} failed: {e}")
        return {
            'experiment_id': experiment_id,
            'final_performance': 0.0,
            'duration': time.time() - start_time,
            'success': False,
            'error': str(e),
            **params
        }


def run_hyperparameter_search(search_space: List[Dict], max_workers: int = 4,
                             max_experiments: int = None) -> pd.DataFrame:
    """Run hyperparameter search with parallel execution."""

    # Create results directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/hp_search_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔬 Starting hyperparameter search")
    print(f"📂 Results directory: {results_dir}")
    print(f"🎯 Search space size: {len(search_space)}")
    print(f"⚡ Max workers: {max_workers}")

    if max_experiments:
        search_space = search_space[:max_experiments]
        print(f"🎲 Running {max_experiments} experiments")

    # Save search space
    with open(results_dir / "search_space.json", 'w') as f:
        json.dump(search_space, f, indent=2)

    results = []

    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_params = {
            executor.submit(run_single_experiment, params, i, results_dir): (params, i)
            for i, params in enumerate(search_space)
        }

        # Collect results as they complete
        for future in as_completed(future_to_params):
            params, exp_id = future_to_params[future]
            try:
                result = future.result()
                results.append(result)

                # Save intermediate results
                df = pd.DataFrame(results)
                df.to_csv(results_dir / "intermediate_results.csv", index=False)

            except Exception as e:
                print(f"❌ Experiment {exp_id} generated an exception: {e}")
                results.append({
                    'experiment_id': exp_id,
                    'final_performance': 0.0,
                    'duration': 0,
                    'success': False,
                    'error': str(e),
                    **params
                })

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "final_results.csv", index=False)

    return results_df, results_dir


def analyze_results(results_df: pd.DataFrame, results_dir: Path):
    """Analyze hyperparameter search results."""

    print("\n🎯 HYPERPARAMETER SEARCH RESULTS")
    print("=" * 40)

    # Filter successful experiments
    successful = results_df[results_df['success'] == True]

    if len(successful) == 0:
        print("❌ No successful experiments!")
        return

    # Top 10 configurations
    top_configs = successful.nlargest(10, 'final_performance')

    print(f"\n🏆 TOP 10 CONFIGURATIONS:")
    print("-" * 30)
    for i, (_, row) in enumerate(top_configs.iterrows(), 1):
        print(f"{i}. Performance: {row['final_performance']:.1f}")
        print(f"   trust_period: {row['trust_period']}, window: {row['performance_window']}")
        print(f"   threshold: {row.get('switch_threshold', 0.0)}, lr: {row['learning_rate']}")
        print(f"   Duration: {row['duration']:.1f}s")
        print()

    # Best configuration
    best_config = top_configs.iloc[0]
    print(f"🥇 BEST CONFIGURATION:")
    print(f"   Performance: {best_config['final_performance']:.1f}")
    print(f"   Parameters: trust_period={best_config['trust_period']}, window={best_config['performance_window']}")
    print(f"   Threshold: {best_config.get('switch_threshold', 0.0)}, LR: {best_config['learning_rate']}")

    # Save best config for easy reuse - convert numpy types to Python types
    best_params = {}
    for k, v in best_config.items():
        if k not in ['experiment_id', 'final_performance', 'duration', 'success']:
            # Convert numpy types to Python types for JSON serialization
            if hasattr(v, 'item'):
                best_params[k] = v.item()
            else:
                best_params[k] = v

    with open(results_dir / "best_config.json", 'w') as f:
        json.dump(best_params, f, indent=2)

    # Parameter importance analysis
    print(f"\n📊 PARAMETER IMPORTANCE:")
    print("-" * 25)

    for param in ['trust_period', 'performance_window', 'learning_rate']:
        if param in successful.columns:
            correlation = successful[param].corr(successful['final_performance'])
            print(f"   {param}: {correlation:.3f}")

    print(f"\n💾 Results saved to: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for reward-based scheduling")
    parser.add_argument("--search-type", choices=["full", "focused"], default="focused",
                       help="Type of search space")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--max-experiments", type=int, default=None,
                       help="Maximum number of experiments to run")
    parser.add_argument("--time-limit", type=int, default=None,
                       help="Time limit in hours")

    args = parser.parse_args()

    # Create search space
    if args.search_type == "full":
        search_space = create_search_space()
    else:
        search_space = create_focused_search_space()

    # Shuffle for better parallel distribution
    random.shuffle(search_space)

    start_time = time.time()

    print(f"🔬 Starting {args.search_type} hyperparameter search")
    print(f"⏰ Time limit: {args.time_limit}h" if args.time_limit else "⏰ No time limit")

    # Run search
    results_df, results_dir = run_hyperparameter_search(
        search_space,
        max_workers=args.max_workers,
        max_experiments=args.max_experiments
    )

    # Analyze results
    analyze_results(results_df, results_dir)

    total_time = (time.time() - start_time) / 3600
    print(f"\n⏱️  Total search time: {total_time:.2f} hours")

    return results_dir


if __name__ == "__main__":
    main()