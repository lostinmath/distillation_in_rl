#!/usr/bin/env python3
"""Simple hyperparameter search for reward-based scheduling strategy."""

import subprocess
import time
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import random


def create_search_space():
    """Create focused search space for key parameters."""

    # Focus on the most critical parameters that showed up in our analysis
    configurations = [
        # Conservative switching - higher trust periods
        {'name': 'conservative_1', 'total_timesteps': 150000, 'learning_rate': 3e-4, 'description': 'Conservative switching with longer trust periods'},
        {'name': 'conservative_2', 'total_timesteps': 200000, 'learning_rate': 3e-4, 'description': 'Conservative with more training'},
        {'name': 'conservative_3', 'total_timesteps': 150000, 'learning_rate': 1e-4, 'description': 'Conservative with lower learning rate'},

        # Balanced approaches
        {'name': 'balanced_1', 'total_timesteps': 150000, 'learning_rate': 3e-4, 'description': 'Standard balanced approach'},
        {'name': 'balanced_2', 'total_timesteps': 200000, 'learning_rate': 5e-4, 'description': 'Balanced with higher LR'},
        {'name': 'balanced_3', 'total_timesteps': 100000, 'learning_rate': 3e-4, 'description': 'Balanced with faster training'},

        # More training time
        {'name': 'extended_1', 'total_timesteps': 300000, 'learning_rate': 3e-4, 'description': 'Extended training time'},
        {'name': 'extended_2', 'total_timesteps': 250000, 'learning_rate': 1e-4, 'description': 'Extended with lower LR'},
        {'name': 'extended_3', 'total_timesteps': 300000, 'learning_rate': 1e-4, 'description': 'Very extended training'},

        # Different learning rates
        {'name': 'high_lr_1', 'total_timesteps': 150000, 'learning_rate': 1e-3, 'description': 'High learning rate'},
        {'name': 'low_lr_1', 'total_timesteps': 150000, 'learning_rate': 5e-5, 'description': 'Very low learning rate'},
        {'name': 'low_lr_2', 'total_timesteps': 200000, 'learning_rate': 5e-5, 'description': 'Very low LR with more training'},
    ]

    return configurations


def run_experiment(config, experiment_id, results_dir):
    """Run a single experiment configuration."""

    start_time = time.time()
    exp_name = f"hp_{config['name']}_{experiment_id:02d}"
    log_file = results_dir / f"{exp_name}.log"

    try:
        # Use the working run_experiment.sh approach
        cmd = [
            "../../run_experiment.sh",
            "--config-path", "configs/experiments/comprehensive",
            "--config-name", "cartpole_reward_based",
            f"total_timesteps={config['total_timesteps']}",
            f"learning_rate={config['learning_rate']}",
            f"seed={random.randint(1, 9999)}"
        ]

        print(f"🚀 Starting {exp_name}: {config['description']}")
        print(f"   Parameters: timesteps={config['total_timesteps']}, lr={config['learning_rate']}")

        # Run with timeout
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            f.write(result.stdout)
            f.write(result.stderr)

        # Extract final performance
        final_performance = 0.0
        training_time = 0.0

        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "Final performance:" in line:
                    try:
                        final_performance = float(line.split(':')[1].strip())
                    except:
                        pass
                if "Training completed in" in line:
                    try:
                        # Extract time from "Training completed in 15.8s"
                        time_str = line.split('in')[1].split('s')[0].strip()
                        training_time = float(time_str)
                    except:
                        pass

        duration = time.time() - start_time

        result_data = {
            'experiment_id': experiment_id,
            'name': config['name'],
            'description': config['description'],
            'final_performance': final_performance,
            'training_time': training_time,
            'total_duration': duration,
            'success': result.returncode == 0,
            'total_timesteps': config['total_timesteps'],
            'learning_rate': config['learning_rate'],
        }

        status = "✅" if result.returncode == 0 else "❌"
        print(f"{status} Completed {exp_name}: {final_performance:.1f} ({duration:.1f}s total)")

        return result_data

    except subprocess.TimeoutExpired:
        print(f"⏰ {exp_name} timed out")
        return {
            'experiment_id': experiment_id,
            'name': config['name'],
            'final_performance': 0.0,
            'success': False,
            'error': 'timeout',
            **config
        }
    except Exception as e:
        print(f"❌ {exp_name} failed: {e}")
        return {
            'experiment_id': experiment_id,
            'name': config['name'],
            'final_performance': 0.0,
            'success': False,
            'error': str(e),
            **config
        }


def main():
    print("🔬 Simple Hyperparameter Search for Reward-Based Scheduling")
    print("=" * 60)

    # Create results directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/simple_hp_search_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get search space
    search_space = create_search_space()
    print(f"📊 Testing {len(search_space)} configurations")
    print(f"📂 Results: {results_dir}")

    # Run experiments sequentially (to avoid overloading)
    results = []

    start_time = time.time()

    for i, config in enumerate(search_space):
        result = run_experiment(config, i, results_dir)
        results.append(result)

        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(results_dir / "results.csv", index=False)

    # Final analysis
    print(f"\n🎯 RESULTS SUMMARY")
    print("=" * 30)

    df = pd.DataFrame(results)
    successful = df[df['success'] == True]

    if len(successful) > 0:
        # Top configurations
        top_configs = successful.nlargest(5, 'final_performance')

        print(f"🏆 TOP CONFIGURATIONS:")
        for i, (_, row) in enumerate(top_configs.iterrows(), 1):
            print(f"{i}. {row['name']}: {row['final_performance']:.1f}")
            print(f"   {row['description']}")
            print(f"   LR: {row['learning_rate']}, Steps: {row['total_timesteps']}")
            print()

        # Best vs baseline comparison
        best_perf = top_configs.iloc[0]['final_performance']
        baseline_perf = 195.8  # From our previous results

        print(f"📊 PERFORMANCE COMPARISON:")
        print(f"   Best config: {best_perf:.1f}")
        print(f"   Student baseline: {baseline_perf:.1f}")
        print(f"   Improvement: {((best_perf/baseline_perf-1)*100):+.1f}%")

        if best_perf > baseline_perf:
            print("🎉 SUCCESS! Found configuration that beats baseline!")
        else:
            print("⚠️  No configuration beat the baseline yet. Need more tuning.")

    else:
        print("❌ No successful experiments")

    total_time = (time.time() - start_time) / 3600
    print(f"\n⏱️  Total time: {total_time:.2f} hours")
    print(f"💾 Results saved: {results_dir}")


if __name__ == "__main__":
    main()