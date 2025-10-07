"""Multi-seed experiment runner with statistical validation.

Runs comprehensive experiments across multiple seeds and methods,
then automatically performs statistical analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json
import concurrent.futures
from tqdm import tqdm
import warnings

from .statistical_validator import StatisticalValidator, ExperimentResult


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    environment: str
    method: str
    seeds: List[int]
    total_timesteps: int
    eval_episodes: int
    save_interval: Optional[int] = None
    config_overrides: Dict[str, Any] = None


class ExperimentRunner:
    """Runs multi-seed experiments and performs statistical analysis."""

    def __init__(
        self,
        output_dir: str = "experiment_results",
        parallel_jobs: int = 1,
        save_individual_runs: bool = True
    ):
        """Initialize experiment runner.

        Args:
            output_dir: Directory for saving results
            parallel_jobs: Number of parallel experiment jobs
            save_individual_runs: Whether to save individual run data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.parallel_jobs = parallel_jobs
        self.save_individual_runs = save_individual_runs

        # Results storage
        self.results: List[ExperimentResult] = []
        self.raw_results: List[Dict] = []

    def run_experiments(
        self,
        experiment_configs: List[ExperimentConfig],
        training_function: Callable,
        baseline_method: str = "student_only"
    ) -> Dict[str, Any]:
        """Run comprehensive experiments with statistical analysis.

        Args:
            experiment_configs: List of experiment configurations
            training_function: Function that runs training (env, method, seed) -> results
            baseline_method: Method to use as baseline for comparisons

        Returns:
            Complete statistical analysis results
        """
        print(f"🚀 Starting comprehensive experiment suite")
        print(f"   Experiments: {len(experiment_configs)}")
        print(f"   Total runs: {sum(len(config.seeds) for config in experiment_configs)}")
        print(f"   Parallel jobs: {self.parallel_jobs}")
        print(f"   Output directory: {self.output_dir}")

        start_time = time.time()

        # Run all experiments
        all_experiment_args = []
        for config in experiment_configs:
            for seed in config.seeds:
                all_experiment_args.append((config, seed, training_function))

        if self.parallel_jobs > 1:
            results = self._run_parallel_experiments(all_experiment_args)
        else:
            results = self._run_sequential_experiments(all_experiment_args)

        # Process and save results
        self._process_results(results)

        # Run statistical analysis
        validator = StatisticalValidator(output_dir=self.output_dir / "statistical_analysis")

        for result in self.results:
            validator.add_result(result)

        statistical_results = validator.run_comprehensive_analysis(baseline_method=baseline_method)

        # Create final summary
        summary = self._create_final_summary(statistical_results, start_time)

        print(f"✅ Experiment suite completed in {time.time() - start_time:.1f}s")
        print(f"📊 Results saved to {self.output_dir}")

        return {
            'experiment_results': self.results,
            'statistical_analysis': statistical_results,
            'summary': summary
        }

    def _run_parallel_experiments(self, experiment_args: List) -> List[Dict]:
        """Run experiments in parallel."""
        results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallel_jobs) as executor:
            # Submit all experiments
            future_to_args = {
                executor.submit(self._run_single_experiment, config, seed, training_func):
                (config, seed) for config, seed, training_func in experiment_args
            }

            # Collect results with progress bar
            with tqdm(total=len(experiment_args), desc="Running experiments") as pbar:
                for future in concurrent.futures.as_completed(future_to_args):
                    config, seed = future_to_args[future]
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.set_postfix(
                            method=config.method,
                            env=config.environment,
                            seed=seed
                        )
                    except Exception as e:
                        print(f"❌ Experiment failed: {config.method}/{config.environment}/seed{seed}: {e}")
                    finally:
                        pbar.update(1)

        return results

    def _run_sequential_experiments(self, experiment_args: List) -> List[Dict]:
        """Run experiments sequentially."""
        results = []

        with tqdm(total=len(experiment_args), desc="Running experiments") as pbar:
            for config, seed, training_func in experiment_args:
                try:
                    result = self._run_single_experiment(config, seed, training_func)
                    results.append(result)
                    pbar.set_postfix(
                        method=config.method,
                        env=config.environment,
                        seed=seed
                    )
                except Exception as e:
                    print(f"❌ Experiment failed: {config.method}/{config.environment}/seed{seed}: {e}")
                finally:
                    pbar.update(1)

        return results

    def _run_single_experiment(
        self,
        config: ExperimentConfig,
        seed: int,
        training_function: Callable
    ) -> Dict:
        """Run a single experiment."""
        # Set random seeds
        np.random.seed(seed)

        # Run training
        training_result = training_function(
            environment=config.environment,
            method=config.method,
            seed=seed,
            total_timesteps=config.total_timesteps,
            eval_episodes=config.eval_episodes,
            config_overrides=config.config_overrides or {}
        )

        # Extract metrics
        result = {
            'experiment_name': config.name,
            'method': config.method,
            'environment': config.environment,
            'seed': seed,
            'total_timesteps': config.total_timesteps,
            'training_time': training_result.get('training_time', 0),
            'final_performance': training_result.get('final_performance', 0),
            'sample_efficiency': training_result.get('sample_efficiency', 0),
            'area_under_curve': training_result.get('area_under_curve', 0),
            'total_reward': training_result.get('total_reward', 0),
            'episode_length_mean': training_result.get('episode_length_mean', 0),
            'teacher_usage_ratio': training_result.get('teacher_usage_ratio', 0),
            'policy_switches': training_result.get('policy_switches', 0),
            'convergence_step': training_result.get('convergence_step'),
            'stability_metric': training_result.get('stability_metric'),
            'raw_data': training_result if self.save_individual_runs else None
        }

        return result

    def _process_results(self, results: List[Dict]) -> None:
        """Process and save experimental results."""
        print("  📊 Processing results...")

        self.raw_results = results

        # Convert to ExperimentResult objects
        for result in results:
            exp_result = ExperimentResult(
                method=result['method'],
                environment=result['environment'],
                seed=result['seed'],
                final_performance=result['final_performance'],
                sample_efficiency=result['sample_efficiency'],
                area_under_curve=result['area_under_curve'],
                total_reward=result['total_reward'],
                episode_length_mean=result['episode_length_mean'],
                teacher_usage_ratio=result['teacher_usage_ratio'],
                policy_switches=result['policy_switches'],
                convergence_step=result['convergence_step'],
                stability_metric=result['stability_metric']
            )
            self.results.append(exp_result)

        # Save results as CSV
        results_df = pd.DataFrame([
            {
                'experiment_name': r['experiment_name'],
                'method': r['method'],
                'environment': r['environment'],
                'seed': r['seed'],
                'final_performance': r['final_performance'],
                'sample_efficiency': r['sample_efficiency'],
                'area_under_curve': r['area_under_curve'],
                'total_reward': r['total_reward'],
                'episode_length_mean': r['episode_length_mean'],
                'teacher_usage_ratio': r['teacher_usage_ratio'],
                'policy_switches': r['policy_switches'],
                'convergence_step': r['convergence_step'],
                'stability_metric': r['stability_metric'],
                'training_time': r['training_time']
            }
            for r in results
        ])

        results_df.to_csv(self.output_dir / 'experiment_results.csv', index=False)

        # Save raw results if requested
        if self.save_individual_runs:
            with open(self.output_dir / 'raw_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)

        print(f"  ✅ Processed {len(results)} experiment results")

    def _create_final_summary(self, statistical_results: Dict, start_time: float) -> Dict[str, Any]:
        """Create final experiment summary."""
        runtime = time.time() - start_time

        # Basic statistics
        df = pd.DataFrame([
            {
                'method': r.method,
                'environment': r.environment,
                'final_performance': r.final_performance,
                'teacher_usage_ratio': r.teacher_usage_ratio
            }
            for r in self.results
        ])

        # Method performance summary
        method_summary = df.groupby('method').agg({
            'final_performance': ['mean', 'std', 'count'],
            'teacher_usage_ratio': ['mean', 'std']
        }).round(4)

        # Environment summary
        env_summary = df.groupby('environment').agg({
            'final_performance': ['mean', 'std', 'count']
        }).round(4)

        # Best method per environment
        best_methods = {}
        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]
            best_method = env_data.groupby('method')['final_performance'].mean().idxmax()
            best_performance = env_data.groupby('method')['final_performance'].mean().max()
            best_methods[env] = {'method': best_method, 'performance': best_performance}

        # Overall best method
        overall_performance = df.groupby('method')['final_performance'].mean()
        best_overall = overall_performance.idxmax()
        best_overall_score = overall_performance.max()

        # Significant improvements count
        sig_improvements = 0
        total_comparisons = 0

        if 'pairwise_comparisons' in statistical_results:
            for metric_results in statistical_results['pairwise_comparisons'].values():
                for method_results in metric_results.values():
                    for test in method_results['tests'].values():
                        if hasattr(test, 'significant'):
                            total_comparisons += 1
                            if test.significant and method_results.get('improvement', 0) > 0:
                                sig_improvements += 1

        summary = {
            'experiment_info': {
                'total_experiments': len(self.raw_results),
                'unique_methods': len(df['method'].unique()),
                'unique_environments': len(df['environment'].unique()),
                'total_seeds': len(df['seed'].unique()) if 'seed' in df.columns else 0,
                'runtime_seconds': runtime,
                'runtime_hours': runtime / 3600
            },
            'method_performance': method_summary.to_dict(),
            'environment_performance': env_summary.to_dict(),
            'best_methods_by_env': best_methods,
            'overall_best': {
                'method': best_overall,
                'mean_performance': best_overall_score
            },
            'statistical_summary': {
                'significant_improvements': sig_improvements,
                'total_comparisons': total_comparisons,
                'significance_rate': sig_improvements / total_comparisons if total_comparisons > 0 else 0
            }
        }

        # Convert pandas objects to serializable format
        def make_serializable(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return str(obj)

        # Save summary
        with open(self.output_dir / 'experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=make_serializable)

        return summary

    def load_results_from_csv(self, csv_path: str) -> None:
        """Load previous experiment results from CSV."""
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
            self.results.append(result)

        print(f"✅ Loaded {len(self.results)} results from {csv_path}")

    def analyze_existing_results(
        self,
        csv_path: str,
        baseline_method: str = "student_only"
    ) -> Dict[str, Any]:
        """Run statistical analysis on existing results."""
        self.load_results_from_csv(csv_path)

        validator = StatisticalValidator(output_dir=self.output_dir / "statistical_analysis")

        for result in self.results:
            validator.add_result(result)

        return validator.run_comprehensive_analysis(baseline_method=baseline_method)