import gymnasium as gym
import numpy as np
import torch
from torch import nn

from adaptive_rl.algorithms.base import Algorithm, AlgorithmRegistry
from adaptive_rl.core.functional import (
    compute_gae,
    compute_ppo_policy_loss,
    compute_ppo_value_loss,
)
from adaptive_rl.networks.mlp import MLPNetwork


@AlgorithmRegistry.register("ppo")
class PPO(Algorithm):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: float | None = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        network: nn.Module | None = None,
        device: str = "cuda",
        seed: int | None = None,
        num_envs: int = 4,
    ):
        super().__init__(observation_space, action_space, device, seed)

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs

        # Handle observation space correctly for vectorized environments
        if len(observation_space.shape) == 2:
            obs_dim = observation_space.shape[1]  # Individual env obs dim
        else:
            obs_dim = np.prod(observation_space.shape)
        if isinstance(action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
            if isinstance(action_space, gym.spaces.Discrete):
                action_dim = action_space.n
            else:  # MultiDiscrete
                action_dim = action_space.nvec[
                    0
                ]  # Assume all envs have same action space
            self.discrete_actions = True
        else:
            action_dim = np.prod(action_space.shape)
            self.discrete_actions = False

        if network is None:
            self.policy = MLPNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=[64, 64],
                activation="tanh",
                discrete=self.discrete_actions,
            ).to(device)
        else:
            self.policy = network.to(device)

        self.value_function = MLPNetwork(
            obs_dim=obs_dim,
            action_dim=1,
            hidden_sizes=[64, 64],
            activation="tanh",
            discrete=True,  # For value function, we want just the output, not a distribution
        ).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_function.parameters()),
            lr=learning_rate,
        )

    def predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.no_grad():
            if self.discrete_actions:
                logits = self.policy(observation)
                distribution = torch.distributions.Categorical(logits=logits)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    action = distribution.sample()
                log_prob = distribution.log_prob(action)
            else:
                mean, std = self.policy(observation)
                distribution = torch.distributions.Normal(mean, std)
                if deterministic:
                    action = mean
                else:
                    action = distribution.sample()
                log_prob = distribution.log_prob(action).sum(dim=-1)

            value = self.value_function(observation)

        return action, value, log_prob

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.discrete_actions:
            logits = self.policy(observations)
            distribution = torch.distributions.Categorical(logits=logits)
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()
        else:
            mean, std = self.policy(observations)
            distribution = torch.distributions.Normal(mean, std)
            log_prob = distribution.log_prob(actions).sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1)

        values = self.value_function(observations)
        return values, log_prob, entropy

    def train_step(
        self,
        rollout_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        observations = rollout_data["observations"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["log_probs"]
        rewards = rollout_data["rewards"]
        dones = rollout_data["dones"]
        values = rollout_data["values"]

        # For GAE computation, we need to reshape back to (n_steps, n_envs)
        # Assume we have n_steps * n_envs total samples
        batch_size = len(observations)
        n_envs = self.num_envs
        n_steps = batch_size // n_envs

        # Reshape to (n_steps, n_envs) for GAE computation
        rewards_2d = rewards.reshape(n_steps, n_envs)
        values_2d = values.reshape(n_steps, n_envs)
        dones_2d = dones.reshape(n_steps, n_envs)

        with torch.no_grad():
            # Get next values for last step
            last_obs = observations[-n_envs:]  # Last observation for each environment
            next_values = self.value_function(last_obs).squeeze(-1)  # Shape: (n_envs,)

            # Compute GAE for each environment separately and then flatten
            all_advantages = []
            all_returns = []

            for env_idx in range(n_envs):
                env_rewards = rewards_2d[:, env_idx]
                env_values = values_2d[:, env_idx]
                env_dones = dones_2d[:, env_idx]
                env_next_value = next_values[env_idx:env_idx+1]

                gae_output = compute_gae(
                    rewards=env_rewards,
                    values=env_values,
                    next_values=env_next_value,
                    dones=env_dones,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                )
                all_advantages.append(gae_output.advantages)
                all_returns.append(gae_output.returns)

            # Flatten back to (batch_size,)
            advantages = torch.cat(all_advantages)
            returns = torch.cat(all_returns)

        total_losses = []
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clipfracs = []
        approx_kl_divs = []

        dataset_size = len(observations)
        indices = np.arange(dataset_size)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_indices = indices[start_idx : start_idx + self.batch_size]

                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = values[batch_indices]

                # Evaluate actions with current policy
                eval_result = self.evaluate_actions(batch_obs, batch_actions)
                if isinstance(eval_result, tuple) and len(eval_result) == 3:
                    batch_values, batch_log_probs, batch_entropy = eval_result
                else:
                    print(
                        f"Unexpected evaluate_actions result: {type(eval_result)}, {eval_result}"
                    )
                    raise ValueError(
                        "evaluate_actions should return (values, log_probs, entropy)"
                    )

                # Normalize advantages
                if self.normalize_advantage:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                        batch_advantages.std() + 1e-8
                    )

                # Compute losses
                pg_loss_result = compute_ppo_policy_loss(
                    logprobs_old=batch_old_log_probs,
                    logprobs_new=batch_log_probs,
                    advantages=batch_advantages,
                    clip_coef=self.clip_range,
                )
                pg_loss, batch_approx_kl, batch_clipfrac = pg_loss_result

                value_loss = compute_ppo_value_loss(
                    values_pred=batch_values.squeeze(-1),  # Remove last dimension
                    returns=batch_returns,
                    values_old=batch_old_values,
                    clip_coef=self.clip_range_vf,
                )

                entropy_loss = batch_entropy.mean()

                # Create loss output object
                class LossOutput:
                    def __init__(
                        self,
                        pg_loss,
                        value_loss,
                        entropy_loss,
                        clipfrac,
                        approx_kl,
                        explained_variance,
                    ):
                        self.pg_loss = pg_loss
                        self.value_loss = value_loss
                        self.entropy_loss = entropy_loss
                        self.clipfrac = clipfrac
                        self.approx_kl = approx_kl
                        self.explained_variance = explained_variance

                # Calculate additional metrics using values we already computed
                with torch.no_grad():
                    y_pred, y_true = batch_values.squeeze(-1), batch_returns
                    var_y = torch.var(y_true)
                    explained_var = (
                        torch.nan
                        if var_y == 0
                        else 1 - torch.var(y_true - y_pred) / var_y
                    )

                loss_output = LossOutput(
                    pg_loss=pg_loss,
                    value_loss=value_loss,
                    entropy_loss=entropy_loss,
                    clipfrac=batch_clipfrac,
                    approx_kl=batch_approx_kl,
                    explained_variance=explained_var,
                )

                loss = (
                    loss_output.pg_loss
                    + self.vf_coef * loss_output.value_loss
                    - self.ent_coef * loss_output.entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.value_function.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_losses.append(loss.item())
                pg_losses.append(loss_output.pg_loss.item())
                value_losses.append(loss_output.value_loss.item())
                entropy_losses.append(loss_output.entropy_loss.item())
                clipfracs.append(loss_output.clipfrac.item())
                approx_kl_divs.append(loss_output.approx_kl.item())

        return {
            "loss/total": np.mean(total_losses),
            "loss/policy": np.mean(pg_losses),
            "loss/value": np.mean(value_losses),
            "loss/entropy": np.mean(entropy_losses),
            "train/clip_fraction": np.mean(clipfracs),
            "train/approx_kl": np.mean(approx_kl_divs),
            "train/explained_variance": loss_output.explained_variance.item(),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_function_state_dict": self.value_function.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hyperparameters": {
                    "learning_rate": self.learning_rate,
                    "n_steps": self.n_steps,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_range": self.clip_range,
                    "clip_range_vf": self.clip_range_vf,
                    "normalize_advantage": self.normalize_advantage,
                    "ent_coef": self.ent_coef,
                    "vf_coef": self.vf_coef,
                    "max_grad_norm": self.max_grad_norm,
                    "discrete_actions": self.discrete_actions,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_function.load_state_dict(checkpoint["value_function_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
