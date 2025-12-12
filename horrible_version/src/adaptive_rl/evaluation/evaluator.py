"""Comprehensive policy evaluation with multiple metrics.

Provides scientifically rigorous evaluation beyond simple episode returns.
"""

import torch
import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
from loguru import logger


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for a policy."""

    # Performance metrics
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    median_return: float

    # Sample efficiency metrics
    mean_episode_length: float
    std_episode_length: float
    success_rate: float  # Fraction of episodes that "succeed"

    # Stability metrics
    return_variance: float
    coefficient_of_variation: float  # std/mean

    # Advanced metrics
    interquartile_range: float
    percentile_95: float
    percentile_5: float

    # Teacher-specific metrics (if applicable)
    teacher_usage_ratio: Optional[float] = None
    switch_frequency: Optional[float] = None
    teacher_agreement: Optional[float] = None

    # Behavioral metrics
    action_entropy: Optional[float] = None
    state_coverage: Optional[float] = None
    trajectory_diversity: Optional[float] = None

    # Meta information
    n_episodes: int = 0
    total_steps: int = 0
    wall_time: float = 0.0


class Evaluator:
    """Comprehensive policy evaluator with multiple metrics."""

    def __init__(
        self,
        env: gym.Env,
        n_eval_episodes: int = 100,
        max_episode_steps: Optional[int] = None,
        success_threshold: Optional[float] = None,
        device: str = "cpu",
    ):
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.max_episode_steps = max_episode_steps or getattr(env, '_max_episode_steps', 1000)
        self.success_threshold = success_threshold
        self.device = device

        # For behavioral analysis
        self.state_visits = {}
        self.trajectories = []

    def evaluate_policy(
        self,
        policy,
        teacher=None,
        scheduler=None,
        save_trajectories: bool = False,
    ) -> EvaluationMetrics:
        """Comprehensive policy evaluation."""
        import time
        start_time = time.time()

        logger.info(f"Evaluating policy for {self.n_eval_episodes} episodes")

        returns = []
        episode_lengths = []
        successes = []

        # Teacher-specific tracking
        teacher_actions = 0
        student_actions = 0
        switches = 0
        last_policy = None
        teacher_agreements = []

        # Behavioral tracking
        all_actions = []
        all_states = []

        if save_trajectories:
            self.trajectories = []

        for episode in range(self.n_eval_episodes):
            episode_return = 0
            episode_length = 0
            episode_actions = []
            episode_states = []

            obs = self.env.reset()[0]

            for step in range(self.max_episode_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

                # Determine policy source if scheduler available
                if scheduler is not None:
                    policies = scheduler.choose_policy_type(
                        iteration=episode,
                        global_step=episode * self.max_episode_steps + step,
                        steps_since_reset=torch.tensor([step]),
                        prev_reward=torch.tensor([episode_return]),
                    )
                    current_policy = policies[0]

                    # Track switches
                    if last_policy is not None and current_policy != last_policy:
                        switches += 1
                    last_policy = current_policy

                    # Get action from appropriate policy
                    if current_policy == "teacher" and teacher is not None:
                        action = teacher.act(obs)
                        action_tensor = torch.tensor(action, device=self.device)
                        teacher_actions += 1

                        # Check agreement with student
                        if hasattr(policy, 'predict'):
                            student_action, _, _ = policy.predict(obs_tensor, deterministic=True)
                            agreement = self._compute_action_agreement(action_tensor, student_action)
                            teacher_agreements.append(agreement)
                    else:
                        action_tensor, _, _ = policy.predict(obs_tensor, deterministic=True)
                        action = action_tensor.cpu().numpy()
                        student_actions += 1
                else:
                    # No scheduler - use policy directly
                    action_tensor, _, _ = policy.predict(obs_tensor, deterministic=True)
                    action = action_tensor.cpu().numpy()
                    student_actions += 1

                # Store for behavioral analysis
                episode_actions.append(action.copy() if hasattr(action, 'copy') else action)
                episode_states.append(obs.copy() if hasattr(obs, 'copy') else obs)

                # Environment step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated.any() if hasattr(terminated, 'any') else terminated
                if hasattr(truncated, 'any'):
                    done = done or truncated.any()

                episode_return += reward.sum() if hasattr(reward, 'sum') else reward
                episode_length += 1

                obs = next_obs

                if done:
                    break

            returns.append(episode_return)
            episode_lengths.append(episode_length)

            # Determine success
            if self.success_threshold is not None:
                successes.append(episode_return >= self.success_threshold)
            else:
                # Use environment-specific success criteria
                successes.append(self._determine_success(episode_return, episode_length, info))

            all_actions.extend(episode_actions)
            all_states.extend(episode_states)

            if save_trajectories:
                self.trajectories.append({
                    'states': episode_states,
                    'actions': episode_actions,
                    'return': episode_return,
                    'length': episode_length,
                })

            if (episode + 1) % 20 == 0:
                logger.info(f"Evaluated {episode + 1}/{self.n_eval_episodes} episodes")

        # Compute metrics
        returns = np.array(returns)
        episode_lengths = np.array(episode_lengths)

        # Teacher-specific metrics
        total_actions = teacher_actions + student_actions
        teacher_usage = teacher_actions / total_actions if total_actions > 0 else 0
        switch_freq = switches / self.n_eval_episodes if self.n_eval_episodes > 0 else 0
        teacher_agreement = np.mean(teacher_agreements) if teacher_agreements else None

        # Behavioral metrics
        action_entropy = self._compute_action_entropy(all_actions)
        state_coverage = self._compute_state_coverage(all_states)
        trajectory_diversity = self._compute_trajectory_diversity()

        wall_time = time.time() - start_time

        metrics = EvaluationMetrics(
            # Performance
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            min_return=float(np.min(returns)),
            max_return=float(np.max(returns)),
            median_return=float(np.median(returns)),

            # Episode length
            mean_episode_length=float(np.mean(episode_lengths)),
            std_episode_length=float(np.std(episode_lengths)),
            success_rate=float(np.mean(successes)),

            # Stability
            return_variance=float(np.var(returns)),
            coefficient_of_variation=float(np.std(returns) / np.mean(returns)) if np.mean(returns) > 0 else float('inf'),

            # Distribution
            interquartile_range=float(np.percentile(returns, 75) - np.percentile(returns, 25)),
            percentile_95=float(np.percentile(returns, 95)),
            percentile_5=float(np.percentile(returns, 5)),

            # Teacher-specific
            teacher_usage_ratio=teacher_usage if teacher is not None else None,
            switch_frequency=switch_freq if scheduler is not None else None,
            teacher_agreement=teacher_agreement,

            # Behavioral
            action_entropy=action_entropy,
            state_coverage=state_coverage,
            trajectory_diversity=trajectory_diversity,

            # Meta
            n_episodes=self.n_eval_episodes,
            total_steps=int(np.sum(episode_lengths)),
            wall_time=wall_time,
        )

        logger.info(f"Evaluation complete: {metrics.mean_return:.2f} ± {metrics.std_return:.2f}")
        return metrics

    def _compute_action_agreement(self, action1: torch.Tensor, action2: torch.Tensor) -> float:
        """Compute agreement between two actions."""
        if action1.dtype == torch.long or action2.dtype == torch.long:
            # Discrete actions - exact match
            return float(torch.mean((action1 == action2).float()))
        else:
            # Continuous actions - within tolerance
            tolerance = 0.1
            return float(torch.mean((torch.abs(action1 - action2) < tolerance).float()))

    def _determine_success(self, episode_return: float, episode_length: int, info: Dict) -> bool:
        """Determine if episode was successful based on environment."""
        # Default success criteria by environment
        env_id = getattr(self.env, 'spec', None)
        if env_id:
            env_name = env_id.id if hasattr(env_id, 'id') else str(env_id)

            if 'CartPole' in env_name:
                return episode_return >= 195  # CartPole success threshold
            elif 'LunarLander' in env_name:
                return episode_return >= 200  # LunarLander success threshold
            elif 'Acrobot' in env_name:
                return episode_length < 500  # Acrobot success is quick solution

        # Fallback: use median return as threshold
        return episode_return >= np.median([episode_return])  # Trivial fallback

    def _compute_action_entropy(self, actions: List) -> float:
        """Compute entropy of action distribution."""
        if not actions:
            return 0.0

        actions_array = np.array(actions)

        if actions_array.dtype in [np.int32, np.int64]:
            # Discrete actions
            unique, counts = np.unique(actions_array, return_counts=True)
            probabilities = counts / counts.sum()
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            return float(entropy)
        else:
            # Continuous actions - discretize first
            n_bins = 20
            hist, _ = np.histogram(actions_array.flatten(), bins=n_bins)
            probabilities = hist / hist.sum()
            probabilities = probabilities[probabilities > 0]
            entropy = -np.sum(probabilities * np.log(probabilities))
            return float(entropy)

    def _compute_state_coverage(self, states: List) -> float:
        """Compute state space coverage (simplified)."""
        if not states:
            return 0.0

        states_array = np.array(states)

        # Simple coverage: number of unique discretized states
        # This is environment-dependent and simplified
        n_bins_per_dim = 10

        if states_array.ndim == 1:
            states_array = states_array.reshape(-1, 1)

        # Discretize each dimension
        discretized = []
        for dim in range(states_array.shape[1]):
            dim_data = states_array[:, dim]
            bins = np.linspace(dim_data.min(), dim_data.max() + 1e-8, n_bins_per_dim + 1)
            discretized.append(np.digitize(dim_data, bins))

        # Count unique state combinations
        unique_states = set(zip(*discretized))
        total_possible = n_bins_per_dim ** states_array.shape[1]

        coverage = len(unique_states) / total_possible
        return float(coverage)

    def _compute_trajectory_diversity(self) -> float:
        """Compute diversity between trajectories."""
        if not hasattr(self, 'trajectories') or len(self.trajectories) < 2:
            return 0.0

        # Simple diversity: average pairwise distance between trajectory returns
        returns = [traj['return'] for traj in self.trajectories]

        pairwise_distances = []
        for i in range(len(returns)):
            for j in range(i + 1, len(returns)):
                distance = abs(returns[i] - returns[j])
                pairwise_distances.append(distance)

        if not pairwise_distances:
            return 0.0

        return float(np.mean(pairwise_distances))

    def save_results(self, metrics: EvaluationMetrics, output_path: Path):
        """Save evaluation results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable dict
        results = {
            'metrics': {k: v for k, v in metrics.__dict__.items()},
            'trajectories': getattr(self, 'trajectories', [])
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Evaluation results saved to {output_path}")

    def compare_with_baseline(
        self, metrics: EvaluationMetrics, baseline_metrics: EvaluationMetrics
    ) -> Dict[str, float]:
        """Compare metrics with baseline."""
        comparison = {}

        # Performance improvement
        comparison['return_improvement'] = (
            metrics.mean_return - baseline_metrics.mean_return
        ) / abs(baseline_metrics.mean_return) if baseline_metrics.mean_return != 0 else 0

        # Sample efficiency improvement
        comparison['length_improvement'] = (
            baseline_metrics.mean_episode_length - metrics.mean_episode_length
        ) / baseline_metrics.mean_episode_length if baseline_metrics.mean_episode_length > 0 else 0

        # Stability improvement (lower CV is better)
        comparison['stability_improvement'] = (
            baseline_metrics.coefficient_of_variation - metrics.coefficient_of_variation
        ) / baseline_metrics.coefficient_of_variation if baseline_metrics.coefficient_of_variation > 0 else 0

        # Success rate improvement
        comparison['success_rate_improvement'] = (
            metrics.success_rate - baseline_metrics.success_rate
        )

        return comparison