"""Compare performance across all scheduling strategies."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_data(
    log_dir: Path, metric: str = "episode/return"
) -> pd.DataFrame:
    """Load metric data from TensorBoard logs."""
    event_acc = EventAccumulator(str(log_dir))
    event_acc.Reload()

    if metric not in event_acc.Tags()["scalars"]:
        return pd.DataFrame()

    events = event_acc.Scalars(metric)
    data = pd.DataFrame(
        [{"step": e.step, "value": e.value, "wall_time": e.wall_time} for e in events]
    )
    return data


def load_csv_metrics(csv_path: Path) -> pd.DataFrame:
    """Load metrics from CSV file."""
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def aggregate_runs(
    base_dir: Path,
    strategies: list[str],
    metric: str = "episode/return",
    source: str = "tensorboard",
) -> dict[str, pd.DataFrame]:
    """Aggregate data across multiple runs for each strategy."""
    results = {}

    for strategy in strategies:
        strategy_data = []

        # Find all runs for this strategy
        pattern = f"{strategy}_*"
        for run_dir in base_dir.glob(pattern):
            if source == "tensorboard":
                data = load_tensorboard_data(run_dir, metric)
            else:  # CSV
                csv_path = run_dir / "csv" / "metrics.csv"
                data = load_csv_metrics(csv_path)
                if not data.empty and metric in data.columns:
                    data = data[["step", metric]].rename(columns={metric: "value"})

            if not data.empty:
                strategy_data.append(data)

        if strategy_data:
            # Combine and aggregate
            combined = pd.concat(strategy_data, ignore_index=True)
            results[strategy] = combined

    return results


def plot_learning_curves(
    data: dict[str, pd.DataFrame],
    title: str = "Learning Curves Comparison",
    ylabel: str = "Episode Return",
    window: int = 100,
    save_path: Path | None = None,
):
    """Plot learning curves for all strategies."""
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for (strategy, df), color in zip(data.items(), colors, strict=False):
        if df.empty:
            continue

        # Group by step and aggregate
        grouped = df.groupby("step")["value"].agg(["mean", "std", "count"])

        # Apply smoothing
        grouped["smoothed_mean"] = (
            grouped["mean"].rolling(window=window, min_periods=1).mean()
        )
        grouped["smoothed_std"] = (
            grouped["std"].rolling(window=window, min_periods=1).mean()
        )

        # Plot
        steps = grouped.index
        mean = grouped["smoothed_mean"]
        std = grouped["smoothed_std"]

        plt.plot(steps, mean, label=strategy, color=color, linewidth=2)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    plt.xlabel("Training Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_final_performance_comparison(
    data: dict[str, pd.DataFrame],
    last_n_episodes: int = 100,
    save_path: Path | None = None,
):
    """Compare final performance across strategies."""
    final_scores = []

    for strategy, df in data.items():
        if df.empty:
            continue

        # Get last n episodes
        last_values = df.nlargest(last_n_episodes, "step")["value"]

        for value in last_values:
            final_scores.append({"Strategy": strategy, "Episode Return": value})

    if not final_scores:
        return

    scores_df = pd.DataFrame(final_scores)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=scores_df, x="Strategy", y="Episode Return")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Final Performance Comparison (Last {last_n_episodes} Episodes)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_teacher_usage_comparison(
    base_dir: Path, strategies: list[str], save_path: Path | None = None
):
    """Compare teacher usage ratios across strategies."""
    usage_data = []

    for strategy in strategies:
        # Skip pure strategies
        if strategy in ["student_only", "teacher_only"]:
            continue

        # Load teacher usage data
        pattern = f"{strategy}_*"
        for run_dir in base_dir.glob(pattern):
            data = load_tensorboard_data(run_dir, "scheduler/teacher_usage_ratio")
            if not data.empty:
                avg_usage = data["value"].mean()
                usage_data.append(
                    {"Strategy": strategy, "Teacher Usage Ratio": avg_usage}
                )

    if not usage_data:
        return

    usage_df = pd.DataFrame(usage_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=usage_df, x="Strategy", y="Teacher Usage Ratio")
    plt.xticks(rotation=45, ha="right")
    plt.title("Average Teacher Usage Across Strategies")
    plt.ylabel("Teacher Usage Ratio")
    plt.ylim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def generate_comparison_table(
    data: dict[str, pd.DataFrame],
    metrics_to_compute: list[str] = [
        "final_mean",
        "final_std",
        "best_mean",
        "convergence_step",
    ],
) -> pd.DataFrame:
    """Generate a comparison table of key metrics."""
    results = []

    for strategy, df in data.items():
        if df.empty:
            continue

        result = {"Strategy": strategy}

        # Final performance
        last_100 = df.nlargest(100, "step")["value"]
        result["Final Mean Return"] = last_100.mean()
        result["Final Std"] = last_100.std()

        # Best performance
        window_mean = df["value"].rolling(window=100, min_periods=1).mean()
        result["Best Mean Return"] = window_mean.max()

        # Convergence speed (first time reaching 90% of best)
        threshold = 0.9 * result["Best Mean Return"]
        converged = df[df["value"] >= threshold]
        if not converged.empty:
            result["Convergence Step"] = converged.iloc[0]["step"]
        else:
            result["Convergence Step"] = np.inf

        results.append(result)

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values("Final Mean Return", ascending=False)

    return comparison_df


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare scheduling strategies")
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Base log directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "student_only",
            "teacher_only",
            "epsilon",
            "epsilon_decreasing",
            "alternating",
            "teacher_then_student",
            "reward_based",
        ],
        help="Strategies to compare",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data = aggregate_runs(log_dir, args.strategies)

    # Generate plots
    print("Generating learning curves...")
    plot_learning_curves(data, save_path=output_dir / "learning_curves.png")

    print("Generating final performance comparison...")
    plot_final_performance_comparison(
        data, save_path=output_dir / "final_performance.png"
    )

    print("Generating teacher usage comparison...")
    plot_teacher_usage_comparison(
        log_dir, args.strategies, save_path=output_dir / "teacher_usage.png"
    )

    # Generate comparison table
    print("Generating comparison table...")
    comparison_table = generate_comparison_table(data)
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    print(comparison_table.to_string(index=False))

    # Save table
    comparison_table.to_csv(output_dir / "comparison_table.csv", index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
