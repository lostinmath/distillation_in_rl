"""Deep analysis of teacher vs student policy behaviors."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


class PolicyAnalyzer:
    """Analyze and compare teacher vs student policies."""

    def __init__(
        self,
        student_checkpoint_path: Path,
        teacher_policy: Any,
        env_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize policy analyzer.

        Args:
            student_checkpoint_path: Path to trained student model
            teacher_policy: Teacher policy instance
            env_id: Environment ID
            device: Device to run analysis on
        """
        self.device = torch.device(device)
        self.env_id = env_id
        self.teacher = teacher_policy

        # Load student model
        self.student = self._load_student_model(student_checkpoint_path)

        # Create test environment
        self.env = gym.make(env_id)

    def _load_student_model(self, checkpoint_path: Path):
        """Load trained student model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Reconstruct model (you'd need to import PPOAgent)
        from adaptive_rl.core.ppo import PPOAgent

        model = PPOAgent(self.env.observation_space, self.env.action_space).to(
            self.device
        )

        model.load_state_dict(checkpoint["agent_state_dict"])
        model.eval()
        return model

    def compute_kl_divergence(
        self, n_episodes: int = 100, n_samples: int = 1000
    ) -> dict[str, float]:
        """Compute KL divergence between teacher and student policies.

        Returns:
            Dictionary with KL statistics
        """
        kl_divergences = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_kls = []

            while not done and len(episode_kls) < n_samples:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                # Get student action distribution
                with torch.no_grad():
                    if hasattr(self.student, "actor"):
                        student_logits = self.student.actor(
                            self.student.shared_net(obs_tensor)
                        )
                        student_probs = F.softmax(student_logits, dim=-1)
                    else:
                        # Handle continuous actions
                        student_mean, student_std = self.student.get_action_dist(
                            obs_tensor
                        )
                        student_dist = torch.distributions.Normal(
                            student_mean, student_std
                        )

                # Get teacher action distribution
                teacher_action = self.teacher.act(obs[np.newaxis, :])[0]

                # For discrete actions, create one-hot distribution for teacher
                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    teacher_probs = torch.zeros_like(student_probs)
                    teacher_probs[0, teacher_action] = 1.0

                    # Compute KL divergence
                    kl = F.kl_div(
                        student_probs.log(), teacher_probs, reduction="batchmean"
                    ).item()
                else:
                    # For continuous, approximate with samples
                    kl = 0.0  # Simplified for now

                episode_kls.append(kl)

                # Step environment
                action = (
                    student_probs.argmax().item()
                    if isinstance(self.env.action_space, gym.spaces.Discrete)
                    else student_dist.sample()
                )
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

            kl_divergences.extend(episode_kls)

        return {
            "mean_kl": np.mean(kl_divergences),
            "std_kl": np.std(kl_divergences),
            "median_kl": np.median(kl_divergences),
            "max_kl": np.max(kl_divergences),
            "min_kl": np.min(kl_divergences),
        }

    def compare_trajectories(self, n_episodes: int = 10) -> dict[str, list[np.ndarray]]:
        """Compare trajectories between teacher and student.

        Returns:
            Dictionary with trajectory data for both policies
        """
        teacher_trajectories = []
        student_trajectories = []

        for _ in range(n_episodes):
            # Collect teacher trajectory
            obs, _ = self.env.reset()
            teacher_traj = {"observations": [obs], "actions": [], "rewards": []}
            done = False

            while not done:
                action = self.teacher.act(obs[np.newaxis, :])[0]
                teacher_traj["actions"].append(action)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                teacher_traj["observations"].append(obs)
                teacher_traj["rewards"].append(reward)
                done = terminated or truncated

            teacher_trajectories.append(teacher_traj)

            # Collect student trajectory
            obs, _ = self.env.reset()
            student_traj = {"observations": [obs], "actions": [], "rewards": []}
            done = False

            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _, _ = self.student.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()[0]

                student_traj["actions"].append(action)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                student_traj["observations"].append(obs)
                student_traj["rewards"].append(reward)
                done = terminated or truncated

            student_trajectories.append(student_traj)

        return {"teacher": teacher_trajectories, "student": student_trajectories}

    def analyze_behavioral_patterns(
        self, trajectories: dict[str, list[dict]]
    ) -> dict[str, Any]:
        """Analyze behavioral patterns in trajectories.

        Returns:
            Analysis of patterns including diversity, consistency, etc.
        """
        results = {}

        for policy_name, trajs in trajectories.items():
            # Extract features
            all_observations = []
            all_actions = []
            episode_returns = []
            episode_lengths = []

            for traj in trajs:
                all_observations.extend(
                    traj["observations"][:-1]
                )  # Exclude terminal state
                all_actions.extend(traj["actions"])
                episode_returns.append(sum(traj["rewards"]))
                episode_lengths.append(len(traj["rewards"]))

            # Convert to arrays
            obs_array = np.array(all_observations)
            action_array = np.array(all_actions)

            # Compute statistics
            results[policy_name] = {
                "mean_return": np.mean(episode_returns),
                "std_return": np.std(episode_returns),
                "mean_length": np.mean(episode_lengths),
                "std_length": np.std(episode_lengths),
                "action_entropy": self._compute_action_entropy(action_array),
                "state_coverage": self._compute_state_coverage(obs_array),
                "action_consistency": self._compute_action_consistency(
                    obs_array, action_array
                ),
            }

        # Compare policies
        results["comparison"] = {
            "return_improvement": results["student"]["mean_return"]
            - results["teacher"]["mean_return"],
            "length_difference": results["student"]["mean_length"]
            - results["teacher"]["mean_length"],
            "entropy_difference": results["student"]["action_entropy"]
            - results["teacher"]["action_entropy"],
            "coverage_difference": results["student"]["state_coverage"]
            - results["teacher"]["state_coverage"],
        }

        return results

    def _compute_action_entropy(self, actions: np.ndarray) -> float:
        """Compute entropy of action distribution."""
        if len(actions) == 0:
            return 0.0

        unique, counts = np.unique(actions, return_counts=True)
        probs = counts / len(actions)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy

    def _compute_state_coverage(self, observations: np.ndarray) -> float:
        """Compute coverage of state space."""
        if len(observations) < 2:
            return 0.0

        # Use PCA to reduce dimensionality if needed
        if observations.shape[1] > 2:
            pca = PCA(n_components=2)
            reduced_obs = pca.fit_transform(observations)
        else:
            reduced_obs = observations

        # Compute coverage as variance in reduced space
        coverage = np.sum(np.var(reduced_obs, axis=0))
        return coverage

    def _compute_action_consistency(
        self, observations: np.ndarray, actions: np.ndarray, n_neighbors: int = 5
    ) -> float:
        """Compute consistency of actions in similar states."""
        if len(observations) < n_neighbors + 1:
            return 0.0

        consistencies = []

        for i in range(len(observations)):
            # Find nearest neighbors
            distances = np.linalg.norm(observations - observations[i], axis=1)
            nearest_indices = np.argsort(distances)[1 : n_neighbors + 1]  # Exclude self

            if len(nearest_indices) > 0:
                # Check action consistency
                neighbor_actions = actions[nearest_indices]
                consistency = np.mean(neighbor_actions == actions[i])
                consistencies.append(consistency)

        return np.mean(consistencies) if consistencies else 0.0

    def detect_novel_solutions(
        self, trajectories: dict[str, list[dict]], similarity_threshold: float = 0.8
    ) -> dict[str, Any]:
        """Detect if student found novel solutions different from teacher.

        Returns:
            Analysis of solution novelty
        """
        teacher_trajs = trajectories["teacher"]
        student_trajs = trajectories["student"]

        # Extract state-action sequences
        teacher_sequences = []
        for traj in teacher_trajs:
            seq = [
                (obs, act)
                for obs, act in zip(
                    traj["observations"][:-1], traj["actions"], strict=False
                )
            ]
            teacher_sequences.append(seq)

        student_sequences = []
        for traj in student_trajs:
            seq = [
                (obs, act)
                for obs, act in zip(
                    traj["observations"][:-1], traj["actions"], strict=False
                )
            ]
            student_sequences.append(seq)

        # Compare sequences
        novel_sequences = []
        similarity_scores = []

        for student_seq in student_sequences:
            max_similarity = 0
            for teacher_seq in teacher_sequences:
                similarity = self._compute_sequence_similarity(student_seq, teacher_seq)
                max_similarity = max(max_similarity, similarity)

            similarity_scores.append(max_similarity)
            if max_similarity < similarity_threshold:
                novel_sequences.append(student_seq)

        return {
            "n_novel_solutions": len(novel_sequences),
            "novelty_rate": len(novel_sequences) / len(student_sequences),
            "mean_similarity": np.mean(similarity_scores),
            "min_similarity": np.min(similarity_scores),
            "novel_sequences": novel_sequences[:3],  # Keep first 3 for visualization
        }

    def _compute_sequence_similarity(
        self, seq1: list[tuple], seq2: list[tuple]
    ) -> float:
        """Compute similarity between two state-action sequences."""
        # Use dynamic time warping or simple overlap
        # Simplified version: compare overlapping portion
        min_len = min(len(seq1), len(seq2))

        if min_len == 0:
            return 0.0

        matches = 0
        for i in range(min_len):
            obs1, act1 = seq1[i]
            obs2, act2 = seq2[i]

            # Check if states and actions are similar
            state_dist = np.linalg.norm(obs1 - obs2)
            action_match = (
                (act1 == act2)
                if isinstance(act1, (int, np.integer))
                else np.allclose(act1, act2)
            )

            if state_dist < 0.5 and action_match:  # Thresholds can be tuned
                matches += 1

        return matches / min_len

    def analyze_failure_patterns(self, n_episodes: int = 50) -> dict[str, Any]:
        """Analyze where and why policies fail.

        Returns:
            Analysis of failure patterns
        """
        teacher_failures = []
        student_failures = []

        for _ in range(n_episodes):
            # Test teacher
            obs, _ = self.env.reset()
            done = False
            steps = 0

            while not done and steps < 500:
                action = self.teacher.act(obs[np.newaxis, :])[0]
                obs, reward, terminated, truncated, _ = self.env.step(action)
                steps += 1
                done = terminated or truncated

                if terminated and steps < 195:  # CartPole "failure" threshold
                    teacher_failures.append({"final_obs": obs, "steps": steps})

            # Test student
            obs, _ = self.env.reset()
            done = False
            steps = 0

            while not done and steps < 500:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _, _ = self.student.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()[0]

                obs, reward, terminated, truncated, _ = self.env.step(action)
                steps += 1
                done = terminated or truncated

                if terminated and steps < 195:  # CartPole "failure" threshold
                    student_failures.append({"final_obs": obs, "steps": steps})

        return {
            "teacher_failure_rate": len(teacher_failures) / n_episodes,
            "student_failure_rate": len(student_failures) / n_episodes,
            "improvement": (len(teacher_failures) - len(student_failures)) / n_episodes,
            "teacher_mean_failure_step": (
                np.mean([f["steps"] for f in teacher_failures])
                if teacher_failures
                else 500
            ),
            "student_mean_failure_step": (
                np.mean([f["steps"] for f in student_failures])
                if student_failures
                else 500
            ),
        }


def plot_policy_comparison(
    kl_stats: dict[str, float],
    behavioral_analysis: dict[str, Any],
    novelty_analysis: dict[str, Any],
    failure_analysis: dict[str, Any],
    save_path: Path | None = None,
):
    """Create comprehensive visualization of policy comparison."""
    fig = plt.figure(figsize=(15, 10))

    # KL Divergence
    ax1 = plt.subplot(2, 3, 1)
    kl_values = [kl_stats["mean_kl"], kl_stats["median_kl"]]
    kl_errors = [kl_stats["std_kl"], 0]
    ax1.bar(["Mean", "Median"], kl_values, yerr=kl_errors, capsize=5)
    ax1.set_title("KL Divergence: Student vs Teacher")
    ax1.set_ylabel("KL Divergence")

    # Performance Comparison
    ax2 = plt.subplot(2, 3, 2)
    policies = ["Teacher", "Student"]
    returns = [
        behavioral_analysis["teacher"]["mean_return"],
        behavioral_analysis["student"]["mean_return"],
    ]
    errors = [
        behavioral_analysis["teacher"]["std_return"],
        behavioral_analysis["student"]["std_return"],
    ]
    ax2.bar(policies, returns, yerr=errors, capsize=5, color=["blue", "orange"])
    ax2.set_title("Performance Comparison")
    ax2.set_ylabel("Episode Return")

    # Behavioral Metrics
    ax3 = plt.subplot(2, 3, 3)
    metrics = ["Entropy", "Coverage", "Consistency"]
    teacher_vals = [
        behavioral_analysis["teacher"]["action_entropy"],
        behavioral_analysis["teacher"]["state_coverage"],
        behavioral_analysis["teacher"]["action_consistency"],
    ]
    student_vals = [
        behavioral_analysis["student"]["action_entropy"],
        behavioral_analysis["student"]["state_coverage"],
        behavioral_analysis["student"]["action_consistency"],
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width / 2, teacher_vals, width, label="Teacher", color="blue")
    ax3.bar(x + width / 2, student_vals, width, label="Student", color="orange")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_title("Behavioral Metrics")
    ax3.legend()

    # Novelty Analysis
    ax4 = plt.subplot(2, 3, 4)
    novelty_data = [
        novelty_analysis["novelty_rate"] * 100,
        (1 - novelty_analysis["novelty_rate"]) * 100,
    ]
    ax4.pie(
        novelty_data,
        labels=["Novel", "Similar"],
        autopct="%1.1f%%",
        colors=["green", "gray"],
    )
    ax4.set_title("Solution Novelty")

    # Failure Analysis
    ax5 = plt.subplot(2, 3, 5)
    failure_rates = [
        failure_analysis["teacher_failure_rate"] * 100,
        failure_analysis["student_failure_rate"] * 100,
    ]
    ax5.bar(policies, failure_rates, color=["blue", "orange"])
    ax5.set_title("Failure Rate Comparison")
    ax5.set_ylabel("Failure Rate (%)")
    ax5.set_ylim(0, max(failure_rates) * 1.2 if max(failure_rates) > 0 else 10)

    # Improvement Summary
    ax6 = plt.subplot(2, 3, 6)
    improvements = {
        "Return": behavioral_analysis["comparison"]["return_improvement"],
        "Entropy": behavioral_analysis["comparison"]["entropy_difference"],
        "Coverage": behavioral_analysis["comparison"]["coverage_difference"],
        "Failure↓": -failure_analysis["improvement"] * 100,
    }

    colors = ["green" if v > 0 else "red" for v in improvements.values()]
    ax6.barh(list(improvements.keys()), list(improvements.values()), color=colors)
    ax6.set_title("Student vs Teacher Improvements")
    ax6.set_xlabel("Difference")
    ax6.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trajectory_comparison(
    trajectories: dict[str, list[dict]], save_path: Path | None = None
):
    """Visualize trajectory differences between teacher and student."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Select first 3 episodes for visualization
    n_episodes = min(3, len(trajectories["teacher"]))

    for i in range(n_episodes):
        ax = axes[0, i]
        teacher_traj = trajectories["teacher"][i]
        student_traj = trajectories["student"][i]

        # Plot state trajectories (first 2 dimensions)
        teacher_states = np.array(teacher_traj["observations"])
        student_states = np.array(student_traj["observations"])

        if teacher_states.shape[1] >= 2:
            ax.plot(
                teacher_states[:, 0],
                teacher_states[:, 1],
                "b-",
                label="Teacher",
                alpha=0.7,
            )
            ax.plot(
                student_states[:, 0],
                student_states[:, 1],
                "r-",
                label="Student",
                alpha=0.7,
            )
            ax.scatter(
                teacher_states[0, 0],
                teacher_states[0, 1],
                color="green",
                s=100,
                marker="o",
                label="Start",
            )
            ax.set_xlabel("State Dim 0")
            ax.set_ylabel("State Dim 1")
        else:
            ax.plot(teacher_states[:, 0], "b-", label="Teacher", alpha=0.7)
            ax.plot(student_states[:, 0], "r-", label="Student", alpha=0.7)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("State Dim 0")

        ax.set_title(f"Episode {i+1} Trajectories")
        ax.legend()

        # Plot actions
        ax = axes[1, i]
        teacher_actions = teacher_traj["actions"]
        student_actions = student_traj["actions"]

        ax.plot(teacher_actions, "b-", label="Teacher", alpha=0.7, marker=".")
        ax.plot(student_actions, "r-", label="Student", alpha=0.7, marker=".")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action")
        ax.set_title(f"Episode {i+1} Actions")
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
