#!/usr/bin/env python3
"""Example script showing comprehensive statistical validation.

This demonstrates how to run multi-seed experiments across multiple methods
and environments, then perform rigorous statistical analysis.
"""

import sys
sys.path.append('src')

import numpy as np
import time
from typing import Dict, Any

from adaptive_rl.validation.experiment_runner import ExperimentRunner, ExperimentConfig
from adaptive_rl.validation.statistical_validator import StatisticalValidator


def mock_training_function(
    environment: str,
    method: str,
    seed: int,
    total_timesteps: int,
    eval_episodes: int,
    config_overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Mock training function for demonstration.

    In practice, this would call your actual training pipeline.
    """
    np.random.seed(seed)

    # Simulate different method performance characteristics
    if method == "student_only":
        # Baseline performance
        base_performance = 0.7
        variance = 0.1
    elif method == "teacher_only":
        # High performance but no learning
        base_performance = 0.9
        variance = 0.05
    elif method == "reward_based":
        # Your main contribution - should outperform baseline
        base_performance = 0.85
        variance = 0.08
    elif method == "epsilon_05":
        # Fixed epsilon scheduling
        base_performance = 0.75
        variance = 0.12
    else:
        base_performance = 0.6
        variance = 0.15

    # Environment-specific adjustments
    if environment == "cheetah_run":
        base_performance *= 1.1
    elif environment == "cartpole":
        base_performance *= 0.9
    elif environment == "walker_walk":
        base_performance *= 1.05

    # Add realistic noise and learning curves
    final_performance = np.random.normal(base_performance, variance)
    final_performance = np.clip(final_performance, 0, 1)

    # Simulate other metrics
    sample_efficiency = np.random.exponential(total_timesteps * (1 - base_performance))
    sample_efficiency = min(sample_efficiency, total_timesteps)

    area_under_curve = final_performance * total_timesteps * np.random.uniform(0.8, 1.2)

    teacher_usage = {
        "student_only": 0.0,
        "teacher_only": 1.0,
        "reward_based": np.random.uniform(0.2, 0.7),
        "epsilon_05": 0.5
    }.get(method, np.random.uniform(0, 1))

    policy_switches = {
        "student_only": 0,
        "teacher_only": 0,
        "reward_based": np.random.poisson(5),
        "epsilon_05": np.random.poisson(10)
    }.get(method, 0)

    # Simulate training time
    time.sleep(0.1)  # Quick simulation

    return {
        'final_performance': final_performance,
        'sample_efficiency': sample_efficiency,
        'area_under_curve': area_under_curve,
        'total_reward': final_performance * 1000,
        'episode_length_mean': np.random.uniform(100, 500),
        'teacher_usage_ratio': teacher_usage,
        'policy_switches': policy_switches,
        'convergence_step': int(sample_efficiency * 0.8),
        'stability_metric': 1 - variance,
        'training_time': 0.1
    }


def run_comprehensive_validation():
    """Run comprehensive statistical validation example."""

    print("🧪 Running Comprehensive Statistical Validation Example")
    print("=" * 60)

    # Define experiment configurations
    methods = [
        "student_only",      # Baseline
        "teacher_only",      # Upper bound
        "reward_based",      # Your main contribution
        "epsilon_05",        # Fixed epsilon baseline
    ]

    environments = [
        "cartpole",
        "cheetah_run",
        "walker_walk"
    ]

    seeds = [42, 123, 456, 789, 1011]  # 5 seeds for statistical power

    # Create experiment configurations
    experiment_configs = []
    for method in methods:
        for env in environments:
            config = ExperimentConfig(
                name=f"{method}_{env}",
                environment=env,
                method=method,
                seeds=seeds,
                total_timesteps=50000,  # Reasonable for testing
                eval_episodes=5
            )
            experiment_configs.append(config)

    print(f"📊 Experiment Plan:")
    print(f"   Methods: {len(methods)} ({', '.join(methods)})")
    print(f"   Environments: {len(environments)} ({', '.join(environments)})")
    print(f"   Seeds per experiment: {len(seeds)}")
    print(f"   Total experiments: {len(experiment_configs) * len(seeds)}")

    # Run experiments
    runner = ExperimentRunner(
        output_dir="statistical_validation_results",
        parallel_jobs=1,  # Sequential for demo
        save_individual_runs=True
    )

    results = runner.run_experiments(
        experiment_configs=experiment_configs,
        training_function=mock_training_function,
        baseline_method="student_only"
    )

    # Print summary
    print("\n" + "="*60)
    print("📈 STATISTICAL VALIDATION RESULTS")
    print("="*60)

    summary = results['summary']

    print(f"\n🎯 Overall Best Method: {summary['overall_best']['method']}")
    print(f"   Mean Performance: {summary['overall_best']['mean_performance']:.3f}")

    print(f"\n📊 Statistical Summary:")
    print(f"   Significant Improvements: {summary['statistical_summary']['significant_improvements']}")
    print(f"   Total Comparisons: {summary['statistical_summary']['total_comparisons']}")
    print(f"   Significance Rate: {summary['statistical_summary']['significance_rate']:.2%}")

    print(f"\n🏆 Best Methods by Environment:")
    for env, info in summary['best_methods_by_env'].items():
        print(f"   {env}: {info['method']} ({info['performance']:.3f})")

    print(f"\n📁 Results saved to: statistical_validation_results/")
    print(f"   - experiment_results.csv: Raw experimental data")
    print(f"   - statistical_analysis/: Complete statistical analysis")
    print(f"   - statistical_analysis/summary_statistics.csv: Summary stats")
    print(f"   - statistical_analysis/pairwise_tests.csv: All statistical tests")
    print(f"   - statistical_analysis/*.png: Publication-quality plots")
    print(f"   - statistical_analysis/statistical_summary.md: Human-readable report")

    return results


def analyze_existing_data():
    """Example of analyzing existing experimental data."""
    print("\n🔍 Example: Analyzing Existing Data")
    print("-" * 40)

    # This would analyze your actual experimental data
    validator = StatisticalValidator(
        output_dir="existing_data_analysis",
        significance_level=0.05,  # Standard in RL literature
        bonferroni_correction=True,  # Conservative for multiple comparisons
        min_effect_size=0.2  # Small-to-medium effect
    )

    # Load from CSV (if you have existing data)
    # validator.add_results_from_csv("your_experiment_data.csv")

    # Or add individual results
    from adaptive_rl.validation.statistical_validator import ExperimentResult

    # Example data
    for method in ["student_only", "reward_based"]:
        for env in ["cartpole", "cheetah_run"]:
            for seed in [1, 2, 3, 4, 5]:
                if method == "reward_based":
                    perf = np.random.normal(0.85, 0.08)
                    teacher_usage = np.random.uniform(0.3, 0.6)
                else:
                    perf = np.random.normal(0.70, 0.10)
                    teacher_usage = 0.0

                result = ExperimentResult(
                    method=method,
                    environment=env,
                    seed=seed,
                    final_performance=max(0, min(1, perf)),
                    sample_efficiency=np.random.uniform(5000, 15000),
                    area_under_curve=perf * 50000,
                    total_reward=perf * 1000,
                    episode_length_mean=200,
                    teacher_usage_ratio=teacher_usage,
                    policy_switches=5 if method == "reward_based" else 0
                )
                validator.add_result(result)

    # Run analysis
    analysis = validator.run_comprehensive_analysis(baseline_method="student_only")

    print("✅ Analysis complete! Check existing_data_analysis/ for results")

    return analysis


def main():
    """Main function demonstrating statistical validation capabilities."""
    print("Statistical Validation Pipeline for Adaptive RL")
    print("=" * 80)

    print("\n🎯 This script demonstrates:")
    print("   ✅ Multi-seed experiment execution")
    print("   ✅ Comprehensive statistical testing")
    print("   ✅ Effect size calculations")
    print("   ✅ Multiple comparison corrections")
    print("   ✅ Publication-quality plots and tables")
    print("   ✅ Human-readable statistical reports")

    # Run comprehensive validation
    results = run_comprehensive_validation()

    # Analyze existing data example
    existing_analysis = analyze_existing_data()

    print("\n" + "="*80)
    print("🎉 STATISTICAL VALIDATION PIPELINE COMPLETE")
    print("="*80)

    print("\n📚 For your research:")
    print("   1. Replace mock_training_function with your actual training code")
    print("   2. Use your real experimental data")
    print("   3. Adjust significance levels and effect sizes as needed")
    print("   4. Use generated plots and tables in your paper")

    print("\n🔬 Statistical Tests Included:")
    print("   - Mann-Whitney U (non-parametric)")
    print("   - Welch's t-test (parametric)")
    print("   - Bootstrap confidence intervals")
    print("   - Cohen's d effect sizes")
    print("   - Kruskal-Wallis (omnibus)")
    print("   - ANOVA (parametric omnibus)")
    print("   - Bonferroni correction for multiple comparisons")

    return results


if __name__ == "__main__":
    main()