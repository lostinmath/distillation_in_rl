"""Comprehensive statistical validation pipeline for adaptive RL experiments.

Provides publication-quality statistical analysis comparing multiple scheduling strategies
across different environments with proper significance testing and effect size reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, friedmanchisquare
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import itertools


@dataclass
class ExperimentResult:
    """Single experiment result."""
    method: str
    environment: str
    seed: int
    final_performance: float
    sample_efficiency: float  # Steps to reach threshold
    area_under_curve: float
    total_reward: float
    episode_length_mean: float
    teacher_usage_ratio: float
    policy_switches: int
    convergence_step: Optional[int] = None
    stability_metric: Optional[float] = None


@dataclass
class StatisticalTest:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""


class StatisticalValidator:
    """Comprehensive statistical validation for RL experiments."""

    def __init__(
        self,
        significance_level: float = 0.05,
        bonferroni_correction: bool = True,
        min_effect_size: float = 0.2,
        output_dir: str = "statistical_results"
    ):
        """Initialize statistical validator.

        Args:
            significance_level: Alpha level for significance testing (0.05 standard in RL)
            bonferroni_correction: Apply Bonferroni correction for multiple comparisons
            min_effect_size: Minimum effect size to consider practically significant
            output_dir: Directory for output files
        """
        self.significance_level = significance_level
        self.bonferroni_correction = bonferroni_correction
        self.min_effect_size = min_effect_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Store results
        self.results: List[ExperimentResult] = []
        self.statistical_tests: List[StatisticalTest] = []

    def add_result(self, result: ExperimentResult) -> None:
        """Add experiment result."""
        self.results.append(result)

    def add_results_from_csv(self, csv_path: str) -> None:
        """Load results from CSV file."""
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            result = ExperimentResult(
                method=row['method'],
                environment=row['environment'],
                seed=int(row['seed']),
                final_performance=float(row['final_performance']),
                sample_efficiency=float(row.get('sample_efficiency', 0)),
                area_under_curve=float(row.get('area_under_curve', 0)),
                total_reward=float(row.get('total_reward', 0)),
                episode_length_mean=float(row.get('episode_length_mean', 0)),
                teacher_usage_ratio=float(row.get('teacher_usage_ratio', 0)),
                policy_switches=int(row.get('policy_switches', 0)),
                convergence_step=row.get('convergence_step'),
                stability_metric=row.get('stability_metric')
            )
            self.add_result(result)

    def run_comprehensive_analysis(
        self,
        baseline_method: str = "student_only",
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive statistical analysis.

        Args:
            baseline_method: Method to use as baseline for comparisons
            metrics: List of metrics to analyze

        Returns:
            Dictionary with all statistical results
        """
        if metrics is None:
            metrics = [
                'final_performance',
                'sample_efficiency',
                'area_under_curve',
                'total_reward',
                'teacher_usage_ratio',
                'policy_switches'
            ]

        print(f"🔬 Running comprehensive statistical analysis")
        print(f"   Baseline method: {baseline_method}")
        print(f"   Significance level: {self.significance_level}")
        print(f"   Metrics: {metrics}")

        # Convert results to DataFrame
        df = self._results_to_dataframe()

        analysis_results = {
            'summary_statistics': self._compute_summary_statistics(df),
            'normality_tests': self._test_normality(df, metrics),
            'pairwise_comparisons': self._run_pairwise_tests(df, baseline_method, metrics),
            'omnibus_tests': self._run_omnibus_tests(df, metrics),
            'effect_sizes': self._compute_effect_sizes(df, baseline_method, metrics),
            'ranking_analysis': self._run_ranking_analysis(df, metrics),
            'environment_analysis': self._analyze_by_environment(df, metrics),
            'meta_analysis': self._run_meta_analysis(df, metrics)
        }

        # Save results
        self._save_results(analysis_results)

        # Generate plots
        self._generate_plots(df, metrics)

        # Generate summary report
        self._generate_summary_report(analysis_results)

        print(f"✅ Analysis complete! Results saved to {self.output_dir}")

        return analysis_results

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = [asdict(result) for result in self.results]
        return pd.DataFrame(data)

    def _compute_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute descriptive statistics for each method."""
        print("  📊 Computing summary statistics...")

        summary = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            summary[method] = {
                'n_runs': len(method_data),
                'n_environments': method_data['environment'].nunique(),
                'n_seeds': method_data['seed'].nunique(),
            }

            for col in numeric_cols:
                if col not in ['seed']:
                    values = method_data[col].dropna()
                    summary[method][col] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75),
                        'min': values.min(),
                        'max': values.max(),
                        'ci_lower': values.mean() - 1.96 * values.std() / np.sqrt(len(values)),
                        'ci_upper': values.mean() + 1.96 * values.std() / np.sqrt(len(values))
                    }

        return summary

    def _test_normality(self, df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict]:
        """Test normality assumptions for each metric and method."""
        print("  📈 Testing normality assumptions...")

        normality_results = {}

        for metric in metrics:
            if metric not in df.columns:
                continue

            normality_results[metric] = {}

            for method in df['method'].unique():
                method_data = df[df['method'] == method][metric].dropna()

                if len(method_data) < 3:
                    continue

                # Shapiro-Wilk test (best for n < 50)
                if len(method_data) <= 50:
                    stat, p_value = stats.shapiro(method_data)
                    test_name = "Shapiro-Wilk"
                else:
                    # Anderson-Darling test for larger samples
                    result = stats.anderson(method_data)
                    stat = result.statistic
                    p_value = result.significance_level[2] / 100  # 5% level
                    test_name = "Anderson-Darling"

                normality_results[metric][method] = {
                    'test': test_name,
                    'statistic': stat,
                    'p_value': p_value,
                    'normal': p_value > self.significance_level,
                    'n_samples': len(method_data)
                }

        return normality_results

    def _run_pairwise_tests(
        self,
        df: pd.DataFrame,
        baseline_method: str,
        metrics: List[str]
    ) -> Dict[str, Dict]:
        """Run pairwise statistical tests against baseline."""
        print(f"  🎯 Running pairwise tests against {baseline_method}...")

        pairwise_results = {}
        methods = [m for m in df['method'].unique() if m != baseline_method]

        # Calculate adjusted alpha for multiple comparisons
        n_comparisons = len(methods) * len(metrics)
        alpha_adj = self.significance_level / n_comparisons if self.bonferroni_correction else self.significance_level

        for metric in metrics:
            if metric not in df.columns:
                continue

            pairwise_results[metric] = {}
            baseline_data = df[df['method'] == baseline_method][metric].dropna()

            for method in methods:
                method_data = df[df['method'] == method][metric].dropna()

                if len(baseline_data) < 3 or len(method_data) < 3:
                    continue

                # Multiple statistical tests for robustness
                tests = self._run_multiple_tests(baseline_data, method_data, alpha_adj)

                pairwise_results[metric][method] = {
                    'baseline_mean': baseline_data.mean(),
                    'method_mean': method_data.mean(),
                    'improvement': (method_data.mean() - baseline_data.mean()) / baseline_data.mean() * 100 if baseline_data.mean() != 0 else 0,
                    'tests': tests,
                    'recommendation': self._interpret_pairwise_result(tests, method_data.mean() > baseline_data.mean())
                }

        return pairwise_results

    def _run_multiple_tests(
        self,
        baseline_data: np.ndarray,
        method_data: np.ndarray,
        alpha: float
    ) -> Dict[str, StatisticalTest]:
        """Run multiple statistical tests for robustness."""
        tests = {}

        # 1. Mann-Whitney U test (non-parametric, robust)
        statistic, p_value = mannwhitneyu(method_data, baseline_data, alternative='two-sided')
        tests['mann_whitney'] = StatisticalTest(
            test_name="Mann-Whitney U",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < alpha,
            interpretation="Non-parametric comparison of medians"
        )

        # 2. Welch's t-test (parametric, unequal variances)
        statistic, p_value = stats.ttest_ind(method_data, baseline_data, equal_var=False)
        tests['welch_ttest'] = StatisticalTest(
            test_name="Welch's t-test",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < alpha,
            interpretation="Parametric comparison assuming unequal variances"
        )

        # 3. Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            boot_method = np.random.choice(method_data, len(method_data), replace=True)
            boot_baseline = np.random.choice(baseline_data, len(baseline_data), replace=True)
            bootstrap_diffs.append(boot_method.mean() - boot_baseline.mean())

        ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
        significant_bootstrap = not (ci_lower <= 0 <= ci_upper)

        tests['bootstrap'] = StatisticalTest(
            test_name="Bootstrap CI",
            statistic=np.mean(bootstrap_diffs),
            p_value=np.nan,  # CI-based, not p-value based
            significant=significant_bootstrap,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=f"{(1-alpha)*100:.0f}% CI for difference in means"
        )

        # 4. Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(method_data) - 1) * method_data.var() +
                             (len(baseline_data) - 1) * baseline_data.var()) /
                            (len(method_data) + len(baseline_data) - 2))
        if pooled_std == 0:
            cohens_d = 0.0
        else:
            cohens_d = (method_data.mean() - baseline_data.mean()) / pooled_std

        tests['effect_size'] = StatisticalTest(
            test_name="Cohen's d",
            statistic=cohens_d,
            p_value=np.nan,
            significant=abs(cohens_d) >= self.min_effect_size,
            effect_size=cohens_d,
            interpretation=self._interpret_cohens_d(cohens_d)
        )

        return tests

    def _run_omnibus_tests(self, df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict]:
        """Run omnibus tests across all methods."""
        print("  🌐 Running omnibus tests...")

        omnibus_results = {}

        for metric in metrics:
            if metric not in df.columns:
                continue

            omnibus_results[metric] = {}

            # Prepare data for omnibus tests
            groups = []
            for method in df['method'].unique():
                method_data = df[df['method'] == method][metric].dropna()
                if len(method_data) >= 3:
                    groups.append(method_data.values)

            if len(groups) < 2:
                continue

            # Kruskal-Wallis test (non-parametric ANOVA)
            statistic, p_value = kruskal(*groups)
            omnibus_results[metric]['kruskal_wallis'] = StatisticalTest(
                test_name="Kruskal-Wallis",
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.significance_level,
                interpretation="Non-parametric test for differences across all methods"
            )

            # One-way ANOVA (parametric)
            statistic, p_value = stats.f_oneway(*groups)
            omnibus_results[metric]['anova'] = StatisticalTest(
                test_name="One-way ANOVA",
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.significance_level,
                interpretation="Parametric test for differences across all methods"
            )

        return omnibus_results

    def _compute_effect_sizes(
        self,
        df: pd.DataFrame,
        baseline_method: str,
        metrics: List[str]
    ) -> Dict[str, Dict]:
        """Compute effect sizes for practical significance."""
        print("  📏 Computing effect sizes...")

        effect_sizes = {}
        baseline_data = df[df['method'] == baseline_method]

        for metric in metrics:
            if metric not in df.columns:
                continue

            effect_sizes[metric] = {}
            baseline_values = baseline_data[metric].dropna()

            for method in df['method'].unique():
                if method == baseline_method:
                    continue

                method_values = df[df['method'] == method][metric].dropna()

                if len(baseline_values) < 3 or len(method_values) < 3:
                    continue

                # Cohen's d
                pooled_std = np.sqrt(((len(method_values) - 1) * method_values.var() +
                                     (len(baseline_values) - 1) * baseline_values.var()) /
                                    (len(method_values) + len(baseline_values) - 2))
                cohens_d = (method_values.mean() - baseline_values.mean()) / pooled_std

                # Eta squared (proportion of variance explained)
                all_values = np.concatenate([baseline_values, method_values])
                between_ss = len(baseline_values) * (baseline_values.mean() - all_values.mean())**2 + \
                            len(method_values) * (method_values.mean() - all_values.mean())**2
                total_ss = np.sum((all_values - all_values.mean())**2)
                eta_squared = between_ss / total_ss if total_ss > 0 else 0

                effect_sizes[metric][method] = {
                    'cohens_d': cohens_d,
                    'eta_squared': eta_squared,
                    'cohens_d_interpretation': self._interpret_cohens_d(cohens_d),
                    'eta_squared_interpretation': self._interpret_eta_squared(eta_squared),
                    'practically_significant': abs(cohens_d) >= self.min_effect_size
                }

        return effect_sizes

    def _run_ranking_analysis(self, df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """Rank methods across all metrics and environments."""
        print("  🏆 Running ranking analysis...")

        ranking_results = {}

        for metric in metrics:
            if metric not in df.columns:
                continue

            # Compute average rank across environments
            method_ranks = {}

            for env in df['environment'].unique():
                env_data = df[df['environment'] == env]
                method_means = env_data.groupby('method')[metric].mean()

                # Rank methods (1 = best)
                if metric in ['sample_efficiency', 'policy_switches']:
                    # Lower is better
                    ranks = method_means.rank(ascending=True)
                else:
                    # Higher is better
                    ranks = method_means.rank(ascending=False)

                for method, rank in ranks.items():
                    if method not in method_ranks:
                        method_ranks[method] = []
                    method_ranks[method].append(rank)

            # Average ranks across environments
            avg_ranks = {method: np.mean(ranks) for method, ranks in method_ranks.items()}

            ranking_results[metric] = {
                'average_ranks': avg_ranks,
                'best_method': min(avg_ranks.items(), key=lambda x: x[1])[0],
                'worst_method': max(avg_ranks.items(), key=lambda x: x[1])[0]
            }

        # Overall ranking across all metrics
        method_total_ranks = {}
        for method in df['method'].unique():
            total_rank = sum(ranking_results[metric]['average_ranks'].get(method, np.inf)
                           for metric in metrics if metric in ranking_results)
            method_total_ranks[method] = total_rank

        ranking_results['overall'] = {
            'total_ranks': method_total_ranks,
            'ranking_order': sorted(method_total_ranks.items(), key=lambda x: x[1])
        }

        return ranking_results

    def _analyze_by_environment(self, df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict]:
        """Analyze performance by environment."""
        print("  🌍 Analyzing by environment...")

        env_analysis = {}

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]
            env_analysis[env] = {}

            for metric in metrics:
                if metric not in df.columns:
                    continue

                method_performance = env_data.groupby('method')[metric].agg(['mean', 'std', 'count'])
                env_analysis[env][metric] = method_performance.to_dict('index')

        return env_analysis

    def _run_meta_analysis(self, df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """Run meta-analysis across environments."""
        print("  🔬 Running meta-analysis...")

        meta_results = {}

        for metric in metrics:
            if metric not in df.columns:
                continue

            meta_results[metric] = {}

            # Effect sizes across environments
            effect_sizes_by_env = {}

            baseline_method = 'student_only'  # Assume this exists
            if baseline_method not in df['method'].unique():
                baseline_method = df['method'].unique()[0]

            for env in df['environment'].unique():
                env_data = df[df['environment'] == env]
                baseline_values = env_data[env_data['method'] == baseline_method][metric].dropna()

                effect_sizes_by_env[env] = {}

                for method in env_data['method'].unique():
                    if method == baseline_method:
                        continue

                    method_values = env_data[env_data['method'] == method][metric].dropna()

                    if len(baseline_values) >= 3 and len(method_values) >= 3:
                        pooled_std = np.sqrt(((len(method_values) - 1) * method_values.var() +
                                             (len(baseline_values) - 1) * baseline_values.var()) /
                                            (len(method_values) + len(baseline_values) - 2))
                        cohens_d = (method_values.mean() - baseline_values.mean()) / pooled_std
                        effect_sizes_by_env[env][method] = cohens_d

            # Average effect sizes across environments
            method_avg_effects = {}
            for method in df['method'].unique():
                if method == baseline_method:
                    continue

                effects = [effect_sizes_by_env[env].get(method, 0)
                          for env in effect_sizes_by_env.keys()]
                effects = [e for e in effects if not np.isnan(e)]

                if effects:
                    method_avg_effects[method] = {
                        'mean_effect': np.mean(effects),
                        'std_effect': np.std(effects),
                        'consistent_positive': all(e > 0 for e in effects),
                        'n_environments': len(effects)
                    }

            meta_results[metric] = {
                'effect_sizes_by_env': effect_sizes_by_env,
                'average_effects': method_avg_effects
            }

        return meta_results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save statistical results to files."""
        print("  💾 Saving results...")

        # Save summary statistics as CSV
        summary_data = []
        for method, stats in results['summary_statistics'].items():
            for metric, values in stats.items():
                if isinstance(values, dict) and 'mean' in values:
                    summary_data.append({
                        'method': method,
                        'metric': metric,
                        'mean': values['mean'],
                        'std': values['std'],
                        'ci_lower': values['ci_lower'],
                        'ci_upper': values['ci_upper'],
                        'median': values['median']
                    })

        pd.DataFrame(summary_data).to_csv(self.output_dir / 'summary_statistics.csv', index=False)

        # Save pairwise test results
        pairwise_data = []
        for metric, comparisons in results['pairwise_comparisons'].items():
            for method, result in comparisons.items():
                for test_name, test in result['tests'].items():
                    pairwise_data.append({
                        'metric': metric,
                        'method': method,
                        'test': test_name,
                        'statistic': test.statistic,
                        'p_value': test.p_value,
                        'significant': test.significant,
                        'effect_size': getattr(test, 'effect_size', None),
                        'improvement_percent': result['improvement']
                    })

        pd.DataFrame(pairwise_data).to_csv(self.output_dir / 'pairwise_tests.csv', index=False)

        # Save full results as JSON
        with open(self.output_dir / 'full_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def _generate_plots(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """Generate publication-quality plots."""
        print("  📊 Generating plots...")

        # Set style for publication
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Performance comparison plots
        for metric in metrics:
            if metric not in df.columns:
                continue

            plt.figure(figsize=(10, 6))

            # Box plot with individual points
            ax = sns.boxplot(data=df, x='method', y=metric)
            sns.stripplot(data=df, x='method', y=metric, color='black', alpha=0.6, size=3)

            plt.title(f'{metric.replace("_", " ").title()} by Method', fontsize=14, fontweight='bold')
            plt.xlabel('Method', fontsize=12)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(self.output_dir / f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 2. Environment-specific performance
        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()

            plot_metrics = ['final_performance', 'sample_efficiency', 'area_under_curve', 'teacher_usage_ratio']

            for i, metric in enumerate(plot_metrics):
                if metric in env_data.columns and i < len(axes):
                    sns.barplot(data=env_data, x='method', y=metric, ax=axes[i])
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].tick_params(axis='x', rotation=45)

            plt.suptitle(f'Performance in {env.replace("_", " ").title()}', fontsize=16, fontweight='bold')
            plt.tight_layout()

            plt.savefig(self.output_dir / f'{env}_performance.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _generate_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate human-readable summary report."""
        print("  📝 Generating summary report...")

        with open(self.output_dir / 'statistical_summary.md', 'w') as f:
            f.write("# Statistical Analysis Summary\n\n")

            # Overall statistics
            f.write("## Summary Statistics\n\n")
            f.write("| Method | N Runs | Final Performance (Mean ± SD) | Sample Efficiency |\n")
            f.write("|--------|--------|------------------------------|-------------------|\n")

            for method, stats in results['summary_statistics'].items():
                if 'final_performance' in stats:
                    perf = stats['final_performance']
                    eff = stats.get('sample_efficiency', {})
                    f.write(f"| {method} | {stats['n_runs']} | "
                           f"{perf['mean']:.3f} ± {perf['std']:.3f} | "
                           f"{eff.get('mean', 0):.0f} |\n")

            # Significance test results
            f.write("\n## Statistical Significance\n\n")
            f.write("| Metric | Method | Mann-Whitney p-value | Effect Size (Cohen's d) | Significant |\n")
            f.write("|--------|--------|----------------------|-------------------------|-------------|\n")

            for metric, comparisons in results['pairwise_comparisons'].items():
                for method, result in comparisons.items():
                    mw_test = result['tests'].get('mann_whitney', {})
                    es_test = result['tests'].get('effect_size', {})

                    f.write(f"| {metric} | {method} | "
                           f"{mw_test.p_value:.4f} | "
                           f"{es_test.statistic:.3f} | "
                           f"{'Yes' if mw_test.significant else 'No'} |\n")

            # Rankings
            f.write("\n## Method Rankings\n\n")
            if 'overall' in results['ranking_analysis']:
                rankings = results['ranking_analysis']['overall']['ranking_order']
                f.write("Overall ranking across all metrics:\n\n")
                for i, (method, score) in enumerate(rankings, 1):
                    f.write(f"{i}. {method} (score: {score:.2f})\n")

            # Recommendations
            f.write("\n## Recommendations\n\n")
            f.write("Based on statistical analysis:\n\n")

            # Find best performing method
            if results['ranking_analysis'] and 'overall' in results['ranking_analysis']:
                best_method = results['ranking_analysis']['overall']['ranking_order'][0][0]
                f.write(f"- **Best overall method**: {best_method}\n")

            # Count significant improvements
            sig_count = 0
            total_tests = 0
            for metric_results in results['pairwise_comparisons'].values():
                for method_results in metric_results.values():
                    for test in method_results['tests'].values():
                        if hasattr(test, 'significant'):
                            total_tests += 1
                            if test.significant:
                                sig_count += 1

            f.write(f"- **Statistical significance rate**: {sig_count}/{total_tests} tests significant\n")
            f.write(f"- **Significance level used**: α = {self.significance_level}\n")
            f.write(f"- **Multiple comparison correction**: {'Bonferroni' if self.bonferroni_correction else 'None'}\n")

    def _interpret_pairwise_result(self, tests: Dict[str, StatisticalTest], method_better: bool) -> str:
        """Interpret pairwise comparison results."""
        mw_sig = tests.get('mann_whitney', StatisticalTest('', 0, 1, False)).significant
        effect_size = tests.get('effect_size', StatisticalTest('', 0, 1, False)).statistic

        if mw_sig and abs(effect_size) >= self.min_effect_size:
            direction = "better" if method_better else "worse"
            magnitude = self._interpret_cohens_d(abs(effect_size)).lower()
            return f"Significantly {direction} with {magnitude} effect size"
        elif mw_sig:
            direction = "better" if method_better else "worse"
            return f"Significantly {direction} but small effect size"
        else:
            return "No significant difference"

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible effect"
        elif abs_d < 0.5:
            return "Small effect"
        elif abs_d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"

    @staticmethod
    def _interpret_eta_squared(eta_sq: float) -> str:
        """Interpret eta squared effect size."""
        if eta_sq < 0.01:
            return "No effect"
        elif eta_sq < 0.06:
            return "Small effect"
        elif eta_sq < 0.14:
            return "Medium effect"
        else:
            return "Large effect"