#!/usr/bin/env python3
"""Automated analysis pipeline for comparing scheduling strategies."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse


def load_experiment_data(results_dir: Path) -> Dict:
    """Load experiment data from logs directory."""
    data = {}

    # Look for CSV logs
    csv_files = list(results_dir.glob("**/*.csv"))
    tensorboard_dirs = list(results_dir.glob("**/events.out.tfevents.*"))

    for csv_file in csv_files:
        experiment_name = csv_file.parent.name
        try:
            df = pd.read_csv(csv_file)
            data[experiment_name] = df
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")

    return data


def extract_metrics(data: Dict) -> pd.DataFrame:
    """Extract key metrics from all experiments."""
    results = []

    for exp_name, df in data.items():
        if df.empty:
            continue

        # Extract strategy name from experiment name
        if 'student_only' in exp_name:
            strategy = 'Student Only'
        elif 'teacher_only' in exp_name:
            strategy = 'Teacher Only'
        elif 'reward_based' in exp_name:
            strategy = 'Reward-Based'
        elif 'epsilon' in exp_name and 'decreasing' in exp_name:
            strategy = 'Epsilon Decreasing'
        elif 'epsilon' in exp_name:
            strategy = 'Fixed Epsilon'
        elif 'interchangeably' in exp_name:
            strategy = 'Interchangeably'
        else:
            strategy = exp_name

        # Extract metrics
        if 'eval_reward' in df.columns:
            final_performance = df['eval_reward'].iloc[-1] if len(df) > 0 else 0
            max_performance = df['eval_reward'].max()
            mean_performance = df['eval_reward'].mean()
        else:
            final_performance = max_performance = mean_performance = 0

        if 'step' in df.columns:
            total_steps = df['step'].max() if len(df) > 0 else 0
            steps_to_threshold = None  # Will calculate below
        else:
            total_steps = steps_to_threshold = 0

        # Calculate sample efficiency (steps to reach 100 reward)
        if 'eval_reward' in df.columns and len(df) > 0:
            threshold_mask = df['eval_reward'] >= 100
            if threshold_mask.any():
                steps_to_threshold = df[threshold_mask]['step'].iloc[0]

        results.append({
            'Strategy': strategy,
            'Experiment': exp_name,
            'Final Performance': final_performance,
            'Max Performance': max_performance,
            'Mean Performance': mean_performance,
            'Total Steps': total_steps,
            'Steps to 100 Reward': steps_to_threshold,
            'Sample Efficiency': 1.0 / (steps_to_threshold or total_steps or 1)
        })

    return pd.DataFrame(results)


def create_comparison_plots(results_df: pd.DataFrame, output_dir: Path):
    """Create comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Final Performance Comparison
    sns.barplot(data=results_df, x='Strategy', y='Final Performance', ax=axes[0,0])
    axes[0,0].set_title('Final Performance by Strategy')
    axes[0,0].tick_params(axis='x', rotation=45)

    # 2. Sample Efficiency (Steps to Threshold)
    efficiency_data = results_df[results_df['Steps to 100 Reward'].notna()]
    if not efficiency_data.empty:
        sns.barplot(data=efficiency_data, x='Strategy', y='Steps to 100 Reward', ax=axes[0,1])
        axes[0,1].set_title('Sample Efficiency (Steps to 100 Reward)')
        axes[0,1].tick_params(axis='x', rotation=45)

    # 3. Performance Distribution
    sns.boxplot(data=results_df, x='Strategy', y='Max Performance', ax=axes[1,0])
    axes[1,0].set_title('Max Performance Distribution')
    axes[1,0].tick_params(axis='x', rotation=45)

    # 4. Strategy Ranking
    strategy_ranking = results_df.groupby('Strategy')['Final Performance'].mean().sort_values(ascending=False)
    strategy_ranking.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Strategy Ranking (Mean Final Performance)')
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results_df: pd.DataFrame, output_dir: Path):
    """Generate a comprehensive comparison report."""
    report_path = output_dir / 'comparison_report.md'

    with open(report_path, 'w') as f:
        f.write("# Scheduling Strategy Comparison Report\n\n")

        # Summary Table
        f.write("## Performance Summary\n\n")
        summary_table = results_df.groupby('Strategy').agg({
            'Final Performance': ['mean', 'std'],
            'Steps to 100 Reward': ['mean', 'std'],
            'Total Steps': 'mean'
        }).round(2)
        f.write(summary_table.to_string())
        f.write("\n\n")

        # Rankings
        f.write("## Strategy Rankings\n\n")

        # By final performance
        final_ranking = results_df.groupby('Strategy')['Final Performance'].mean().sort_values(ascending=False)
        f.write("### By Final Performance:\n")
        for i, (strategy, performance) in enumerate(final_ranking.items(), 1):
            f.write(f"{i}. **{strategy}**: {performance:.2f}\n")
        f.write("\n")

        # By sample efficiency
        efficiency_ranking = results_df[results_df['Steps to 100 Reward'].notna()].groupby('Strategy')['Steps to 100 Reward'].mean().sort_values()
        if not efficiency_ranking.empty:
            f.write("### By Sample Efficiency (Steps to 100 Reward):\n")
            for i, (strategy, steps) in enumerate(efficiency_ranking.items(), 1):
                f.write(f"{i}. **{strategy}**: {steps:.0f} steps\n")
        f.write("\n")

        # Key Insights
        f.write("## Key Insights\n\n")

        best_final = final_ranking.index[0]
        f.write(f"- **Best Final Performance**: {best_final} ({final_ranking.iloc[0]:.2f})\n")

        if not efficiency_ranking.empty:
            best_efficiency = efficiency_ranking.index[0]
            f.write(f"- **Most Sample Efficient**: {best_efficiency} ({efficiency_ranking.iloc[0]:.0f} steps)\n")

        # Statistical significance tests could be added here
        f.write("- **Recommendation**: ")
        if 'Reward-Based' in final_ranking.index[:2]:
            f.write("Reward-based scheduling shows competitive performance, validating the adaptive approach.\n")
        else:
            f.write("Consider hyperparameter tuning for reward-based scheduling to improve performance.\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze scheduling strategy experiments")
    parser.add_argument("--results-dir", type=str, default="logs", help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="analysis", help="Output directory for analysis")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print("🔬 Loading experiment data...")
    data = load_experiment_data(results_dir)
    print(f"Found {len(data)} experiments")

    if not data:
        print("❌ No experiment data found. Check results directory.")
        return

    print("📊 Extracting metrics...")
    results_df = extract_metrics(data)

    print("📈 Creating visualizations...")
    create_comparison_plots(results_df, output_dir)

    print("📝 Generating report...")
    generate_report(results_df, output_dir)

    print(f"✅ Analysis complete! Results saved in {output_dir}")
    print(f"📄 View report: {output_dir}/comparison_report.md")
    print(f"📊 View plots: {output_dir}/strategy_comparison.png")


if __name__ == "__main__":
    main()