"""Main evaluation script for trained adaptive-rl policies.

Usage:
    uv run adaptive-evaluate --checkpoint path/to/model.pt --config config.yaml
    uv run adaptive-evaluate --experiment-dir results/experiment/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from adaptive_rl.algorithms import AlgorithmRegistry
from adaptive_rl.envs import make_vec_env
from adaptive_rl.evaluation import Evaluator, EvaluationMetrics
from adaptive_rl.schedulers import SCHEDULERS
from adaptive_rl.teachers import create_teacher


console = Console()


def load_trained_model(checkpoint_path: Path, config: DictConfig):
    """Load a trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")

    # Create environment
    env = make_vec_env(
        env_id=config.environment.env_id,
        num_envs=1,  # Single env for evaluation
        seed=config.experiment.seed,
    )

    # Create algorithm
    algorithm_cfg = OmegaConf.to_container(config.algorithm, resolve=True)
    algorithm_name = algorithm_cfg.pop("name")
    algorithm_cfg.pop("_target_", None)

    algorithm = AlgorithmRegistry.create(
        algorithm_name,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=config.experiment.device,
        seed=config.experiment.seed,
        **algorithm_cfg,
    )

    # Load checkpoint
    algorithm.load(str(checkpoint_path))

    # Create teacher if needed
    teacher = None
    if config.get("teacher"):
        teacher = create_teacher(
            teacher_type=config.teacher.name,
            env_id=config.environment.env_id,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

    # Create scheduler if needed
    scheduler = None
    if config.get("scheduler"):
        scheduler_cfg = OmegaConf.to_container(config.scheduler, resolve=True)
        scheduler_name = scheduler_cfg.get("name", "student_only")

        if scheduler_name in SCHEDULERS:
            scheduler_class = SCHEDULERS[scheduler_name]
        else:
            scheduler_class = hydra.utils.get_class(scheduler_cfg["_target_"])

        kwargs = {k: v for k, v in scheduler_cfg.items() if k not in ["name", "_target_"]}
        scheduler = scheduler_class(
            student_policy=algorithm,
            teacher_policy=teacher,
            num_envs=1,
            **kwargs,
        )

    return algorithm, teacher, scheduler, env


def evaluate_single_model(
    checkpoint_path: Path,
    config: DictConfig,
    n_eval_episodes: int = 100,
    save_trajectories: bool = False,
) -> EvaluationMetrics:
    """Evaluate a single trained model."""
    algorithm, teacher, scheduler, env = load_trained_model(checkpoint_path, config)

    evaluator = Evaluator(
        env=env,
        n_eval_episodes=n_eval_episodes,
        device=config.experiment.device,
    )

    metrics = evaluator.evaluate_policy(
        policy=algorithm,
        teacher=teacher,
        scheduler=scheduler,
        save_trajectories=save_trajectories,
    )

    return metrics


def evaluate_experiment_directory(
    experiment_dir: Path,
    strategies: Optional[List[str]] = None,
    n_eval_episodes: int = 100,
) -> Dict[str, List[EvaluationMetrics]]:
    """Evaluate all models in an experiment directory."""
    logger.info(f"Evaluating experiment directory: {experiment_dir}")

    results = {}

    # Find all strategy directories
    strategy_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]

    if strategies:
        strategy_dirs = [d for d in strategy_dirs if d.name in strategies]

    for strategy_dir in strategy_dirs:
        strategy_name = strategy_dir.name
        logger.info(f"Evaluating strategy: {strategy_name}")

        strategy_results = []

        # Find all seed directories
        seed_dirs = [d for d in strategy_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]

        for seed_dir in seed_dirs:
            # Find config file
            config_file = seed_dir / ".hydra" / "config.yaml"
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_file}")
                continue

            # Find checkpoint file
            checkpoint_files = list((seed_dir / "checkpoints").glob("*.pt"))
            if not checkpoint_files:
                logger.warning(f"No checkpoints found in {seed_dir}")
                continue

            # Use the latest checkpoint
            checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

            try:
                # Load config
                config = OmegaConf.load(config_file)

                # Evaluate
                metrics = evaluate_single_model(
                    checkpoint_path=checkpoint_path,
                    config=config,
                    n_eval_episodes=n_eval_episodes,
                )

                strategy_results.append(metrics)
                logger.info(f"  Seed {seed_dir.name}: {metrics.mean_return:.2f} ± {metrics.std_return:.2f}")

            except Exception as e:
                logger.error(f"Failed to evaluate {seed_dir}: {e}")
                continue

        results[strategy_name] = strategy_results

    return results


def print_evaluation_summary(results: Dict[str, List[EvaluationMetrics]]):
    """Print a summary table of evaluation results."""
    table = Table(title="Evaluation Results Summary")

    table.add_column("Strategy", style="cyan")
    table.add_column("Seeds", justify="center")
    table.add_column("Mean Return", justify="right")
    table.add_column("Std Return", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Efficiency", justify="right")
    table.add_column("Teacher Usage", justify="right")

    for strategy_name, strategy_results in results.items():
        if not strategy_results:
            continue

        # Aggregate across seeds
        returns = [m.mean_return for m in strategy_results]
        success_rates = [m.success_rate for m in strategy_results]
        episode_lengths = [m.mean_episode_length for m in strategy_results]
        teacher_usages = [m.teacher_usage_ratio for m in strategy_results if m.teacher_usage_ratio is not None]

        mean_return = f"{np.mean(returns):.1f} ± {np.std(returns):.1f}"
        std_return = f"{np.mean([m.std_return for m in strategy_results]):.1f}"
        success_rate = f"{np.mean(success_rates):.2%}"
        efficiency = f"{np.mean(episode_lengths):.0f}"
        teacher_usage = f"{np.mean(teacher_usages):.2%}" if teacher_usages else "N/A"

        table.add_row(
            strategy_name,
            str(len(strategy_results)),
            mean_return,
            std_return,
            success_rate,
            efficiency,
            teacher_usage,
        )

    console.print(table)


def save_detailed_results(results: Dict[str, List[EvaluationMetrics]], output_path: Path):
    """Save detailed evaluation results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable_results = {}
    for strategy, metrics_list in results.items():
        serializable_results[strategy] = [
            {k: v for k, v in metrics.__dict__.items()}
            for metrics in metrics_list
        ]

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    logger.info(f"Detailed results saved to {output_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained adaptive-rl models")
    parser.add_argument("--checkpoint", type=Path, help="Path to model checkpoint")
    parser.add_argument("--experiment-dir", type=Path, help="Path to experiment directory")
    parser.add_argument("--strategies", nargs="+", help="Specific strategies to evaluate")
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument("--save-trajectories", action="store_true", help="Save trajectory data")

    args = parser.parse_args()

    if args.checkpoint:
        # Evaluate single checkpoint
        logger.info(f"Evaluating single checkpoint: {args.checkpoint}")
        metrics = evaluate_single_model(
            checkpoint_path=args.checkpoint,
            config=cfg,
            n_eval_episodes=args.n_episodes,
            save_trajectories=args.save_trajectories,
        )

        console.print(f"[bold green]Evaluation Results[/bold green]")
        console.print(f"Mean Return: {metrics.mean_return:.2f} ± {metrics.std_return:.2f}")
        console.print(f"Success Rate: {metrics.success_rate:.2%}")
        console.print(f"Mean Episode Length: {metrics.mean_episode_length:.1f}")

        if metrics.teacher_usage_ratio is not None:
            console.print(f"Teacher Usage: {metrics.teacher_usage_ratio:.2%}")

    elif args.experiment_dir:
        # Evaluate entire experiment
        logger.info(f"Evaluating experiment directory: {args.experiment_dir}")
        results = evaluate_experiment_directory(
            experiment_dir=args.experiment_dir,
            strategies=args.strategies,
            n_eval_episodes=args.n_episodes,
        )

        print_evaluation_summary(results)

        if args.output:
            save_detailed_results(results, args.output)

    else:
        parser.error("Must specify either --checkpoint or --experiment-dir")


if __name__ == "__main__":
    import numpy as np
    main()