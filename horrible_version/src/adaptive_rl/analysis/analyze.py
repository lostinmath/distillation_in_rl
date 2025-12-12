"""Main post-training analysis script."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from adaptive_rl.analysis.policy_comparison import (
    PolicyAnalyzer,
    plot_policy_comparison,
    plot_trajectory_comparison,
)
from adaptive_rl.teachers import create_teacher


def run_deep_analysis(
    checkpoint_path: Path,
    env_id: str,
    teacher_type: str = "optimal",
    output_dir: Path = None,
) -> dict[str, Any]:
    """Run comprehensive post-training analysis.

    Args:
        checkpoint_path: Path to trained model checkpoint
        env_id: Environment ID
        teacher_type: Type of teacher to compare against
        output_dir: Directory to save results

    Returns:
        Dictionary with all analysis results
    """
    print(f"Analyzing checkpoint: {checkpoint_path}")
    print("=" * 60)

    # Create teacher
    import gymnasium as gym

    env = gym.make(env_id)
    teacher = create_teacher(
        teacher_type=teacher_type,
        env_id=env_id,
        action_space=env.action_space,
        observation_space=env.observation_space,
    )
    env.close()

    # Initialize analyzer
    analyzer = PolicyAnalyzer(
        student_checkpoint_path=checkpoint_path, teacher_policy=teacher, env_id=env_id
    )

    results = {}

    # 1. KL Divergence Analysis
    print("\n1. Computing KL Divergence...")
    kl_stats = analyzer.compute_kl_divergence(n_episodes=50)
    results["kl_divergence"] = kl_stats
    print(f"   Mean KL: {kl_stats['mean_kl']:.4f} ± {kl_stats['std_kl']:.4f}")
    print(f"   Median KL: {kl_stats['median_kl']:.4f}")

    # 2. Trajectory Comparison
    print("\n2. Collecting Trajectories...")
    trajectories = analyzer.compare_trajectories(n_episodes=20)
    results["trajectories"] = {
        "teacher_returns": [sum(t["rewards"]) for t in trajectories["teacher"]],
        "student_returns": [sum(t["rewards"]) for t in trajectories["student"]],
    }

    # 3. Behavioral Pattern Analysis
    print("\n3. Analyzing Behavioral Patterns...")
    behavioral_analysis = analyzer.analyze_behavioral_patterns(trajectories)
    results["behavioral_patterns"] = behavioral_analysis

    print(
        f"   Teacher - Mean Return: {behavioral_analysis['teacher']['mean_return']:.2f}"
    )
    print(
        f"   Student - Mean Return: {behavioral_analysis['student']['mean_return']:.2f}"
    )
    print(
        f"   Return Improvement: {behavioral_analysis['comparison']['return_improvement']:.2f}"
    )
    print(
        f"   Action Entropy Difference: {behavioral_analysis['comparison']['entropy_difference']:.4f}"
    )

    # 4. Novel Solution Detection
    print("\n4. Detecting Novel Solutions...")
    novelty_analysis = analyzer.detect_novel_solutions(trajectories)
    results["novelty"] = novelty_analysis

    print(
        f"   Novel Solutions Found: {novelty_analysis['n_novel_solutions']}/{len(trajectories['student'])}"
    )
    print(f"   Novelty Rate: {novelty_analysis['novelty_rate']*100:.1f}%")
    print(f"   Mean Similarity to Teacher: {novelty_analysis['mean_similarity']:.3f}")

    # 5. Failure Pattern Analysis
    print("\n5. Analyzing Failure Patterns...")
    failure_analysis = analyzer.analyze_failure_patterns(n_episodes=50)
    results["failures"] = failure_analysis

    print(
        f"   Teacher Failure Rate: {failure_analysis['teacher_failure_rate']*100:.1f}%"
    )
    print(
        f"   Student Failure Rate: {failure_analysis['student_failure_rate']*100:.1f}%"
    )
    print(f"   Improvement: {failure_analysis['improvement']*100:.1f}%")

    # 6. Create Visualizations
    if output_dir:
        print("\n6. Creating Visualizations...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Policy comparison plot
        plot_policy_comparison(
            kl_stats,
            behavioral_analysis,
            novelty_analysis,
            failure_analysis,
            save_path=output_dir / "policy_comparison.png",
        )

        # Trajectory comparison plot
        plot_trajectory_comparison(
            trajectories, save_path=output_dir / "trajectory_comparison.png"
        )

        # Save results as JSON
        json_results = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in results.items()
        }
        with open(output_dir / "analysis_results.json", "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"   Saved visualizations to {output_dir}")

    return results


def analyze_scheduling_strategy(
    strategy: str, log_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Analyze a specific scheduling strategy.

    Args:
        strategy: Strategy name
        log_dir: Base log directory
        output_dir: Output directory for results

    Returns:
        Analysis results for the strategy
    """
    print(f"\nAnalyzing {strategy} strategy...")
    print("-" * 50)

    # Find latest checkpoint for this strategy
    strategy_runs = list(log_dir.glob(f"{strategy}_*"))

    if not strategy_runs:
        print(f"No runs found for {strategy}")
        return {}

    # Use most recent run
    latest_run = sorted(strategy_runs, key=lambda p: p.stat().st_mtime)[-1]

    # Find final checkpoint
    checkpoints = list(latest_run.glob("checkpoint_*.pt"))
    if not checkpoints:
        print(f"No checkpoints found in {latest_run}")
        return {}

    final_checkpoint = sorted(checkpoints, key=lambda p: p.name)[-1]

    # Determine environment from logs or config
    env_id = "CartPole-v1"  # Default, could be extracted from config

    # Run analysis
    strategy_output = output_dir / strategy
    results = run_deep_analysis(
        checkpoint_path=final_checkpoint, env_id=env_id, output_dir=strategy_output
    )

    return {strategy: results}


def compare_all_strategies(log_dir: Path, output_dir: Path, strategies: list = None):
    """Compare all scheduling strategies with deep analysis.

    Args:
        log_dir: Base log directory
        output_dir: Output directory for results
        strategies: List of strategies to analyze
    """
    if strategies is None:
        strategies = [
            "student_only",
            "teacher_only",
            "epsilon",
            "epsilon_decreasing",
            "alternating",
            "teacher_then_student",
            "reward_based",
        ]

    all_results = {}

    # Analyze each strategy
    for strategy in strategies:
        results = analyze_scheduling_strategy(strategy, log_dir, output_dir)
        all_results.update(results)

    # Create comparison visualizations
    if all_results:
        create_strategy_comparison_plots(all_results, output_dir)

    return all_results


def create_strategy_comparison_plots(
    all_results: dict[str, dict[str, Any]], output_dir: Path
):
    """Create comparison plots across all strategies."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    strategies = list(all_results.keys())

    # KL Divergence Comparison
    ax = axes[0, 0]
    kl_means = [
        all_results[s].get("kl_divergence", {}).get("mean_kl", 0) for s in strategies
    ]
    kl_stds = [
        all_results[s].get("kl_divergence", {}).get("std_kl", 0) for s in strategies
    ]
    ax.bar(strategies, kl_means, yerr=kl_stds, capsize=5)
    ax.set_title("KL Divergence Across Strategies")
    ax.set_ylabel("Mean KL")
    ax.set_xticklabels(strategies, rotation=45, ha="right")

    # Return Improvement
    ax = axes[0, 1]
    improvements = [
        all_results[s]
        .get("behavioral_patterns", {})
        .get("comparison", {})
        .get("return_improvement", 0)
        for s in strategies
    ]
    colors = ["green" if imp > 0 else "red" for imp in improvements]
    ax.bar(strategies, improvements, color=colors)
    ax.set_title("Return Improvement vs Teacher")
    ax.set_ylabel("Improvement")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticklabels(strategies, rotation=45, ha="right")

    # Novelty Rate
    ax = axes[0, 2]
    novelty_rates = [
        all_results[s].get("novelty", {}).get("novelty_rate", 0) * 100
        for s in strategies
    ]
    ax.bar(strategies, novelty_rates, color="purple")
    ax.set_title("Solution Novelty Rate")
    ax.set_ylabel("Novelty (%)")
    ax.set_xticklabels(strategies, rotation=45, ha="right")

    # Action Entropy
    ax = axes[1, 0]
    entropies_teacher = [
        all_results[s]
        .get("behavioral_patterns", {})
        .get("teacher", {})
        .get("action_entropy", 0)
        for s in strategies
    ]
    entropies_student = [
        all_results[s]
        .get("behavioral_patterns", {})
        .get("student", {})
        .get("action_entropy", 0)
        for s in strategies
    ]

    x = np.arange(len(strategies))
    width = 0.35
    ax.bar(x - width / 2, entropies_teacher, width, label="Teacher", color="blue")
    ax.bar(x + width / 2, entropies_student, width, label="Student", color="orange")
    ax.set_title("Action Entropy Comparison")
    ax.set_ylabel("Entropy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.legend()

    # Failure Rate Improvement
    ax = axes[1, 1]
    failure_improvements = [
        all_results[s].get("failures", {}).get("improvement", 0) * 100
        for s in strategies
    ]
    colors = ["green" if imp > 0 else "red" for imp in failure_improvements]
    ax.bar(strategies, failure_improvements, color=colors)
    ax.set_title("Failure Rate Improvement")
    ax.set_ylabel("Improvement (%)")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticklabels(strategies, rotation=45, ha="right")

    # State Coverage Difference
    ax = axes[1, 2]
    coverage_diffs = [
        all_results[s]
        .get("behavioral_patterns", {})
        .get("comparison", {})
        .get("coverage_difference", 0)
        for s in strategies
    ]
    colors = ["green" if diff > 0 else "red" for diff in coverage_diffs]
    ax.bar(strategies, coverage_diffs, color=colors)
    ax.set_title("State Coverage Difference")
    ax.set_ylabel("Coverage Difference")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticklabels(strategies, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(
        output_dir / "strategy_deep_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.show()


def main():
    """Main analysis entry point."""
    parser = argparse.ArgumentParser(description="Deep post-training analysis")

    parser.add_argument(
        "--checkpoint", type=str, help="Path to specific checkpoint to analyze"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Base log directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/deep_analysis",
        help="Output directory",
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment ID")
    parser.add_argument("--teacher", type=str, default="optimal", help="Teacher type")
    parser.add_argument(
        "--compare-all", action="store_true", help="Compare all strategies"
    )
    parser.add_argument("--strategies", nargs="+", help="Strategies to compare")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_all:
        # Compare all strategies
        print("Comparing all scheduling strategies...")
        print("=" * 60)
        compare_all_strategies(Path(args.log_dir), output_dir, args.strategies)
    elif args.checkpoint:
        # Analyze single checkpoint
        results = run_deep_analysis(
            checkpoint_path=Path(args.checkpoint),
            env_id=args.env,
            teacher_type=args.teacher,
            output_dir=output_dir,
        )

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Results saved to {output_dir}")
    else:
        print("Please specify --checkpoint or --compare-all")


if __name__ == "__main__":
    main()
