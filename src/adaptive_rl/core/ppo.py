"""PPO implementation for state-based observations.

Adapted from the original RGB-based implementation, simplified for
standard Gymnasium environments like CartPole and LunarLander.
"""

import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical, Normal

from adaptive_rl.core.config import DistillationConfig
from adaptive_rl.core.functional import compute_gae, compute_ppo_losses
from adaptive_rl.schedulers import create_scheduler
from adaptive_rl.teachers import create_teacher
from adaptive_rl.utils.logging import Logger


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize network layers with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    """PPO agent with separate actor and critic networks for state observations."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_size: int = 64,
        n_hidden_layers: int = 2,
        activation: str = "tanh",
    ):
        """Initialize PPO agent.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            hidden_size: Hidden layer size
            n_hidden_layers: Number of hidden layers
            activation: Activation function ("tanh" or "relu")
        """
        super().__init__()

        obs_shape = observation_space.shape
        obs_dim = np.prod(obs_shape)

        # Determine if discrete or continuous action space
        self.discrete_actions = isinstance(action_space, gym.spaces.Discrete)
        if self.discrete_actions:
            action_dim = action_space.n
        else:
            action_dim = np.prod(action_space.shape)

        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU
        else:
            self.activation = nn.Tanh

        # Build shared network layers
        layers = []
        input_dim = obs_dim
        for _ in range(n_hidden_layers):
            layers.append(layer_init(nn.Linear(input_dim, hidden_size)))
            layers.append(self.activation())
            input_dim = hidden_size

        self.shared_net = nn.Sequential(*layers)

        # Actor head
        if self.discrete_actions:
            self.actor = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        else:
            self.actor_mean = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic head
        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def get_value(self, obs):
        """Get value estimate for observations."""
        features = self.shared_net(obs)
        return self.critic(features)

    def get_action_and_value(self, obs, action=None):
        """Get action, log probability, entropy, and value.

        Args:
            obs: Observations
            action: Optional actions to evaluate

        Returns:
            Tuple of (action, logprob, entropy, value)
        """
        features = self.shared_net(obs)
        value = self.critic(features)

        if self.discrete_actions:
            logits = self.actor(features)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), value
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            value,
        )


class PPOTrainer:
    """PPO trainer with teacher-guided scheduling support."""

    def __init__(self, config: DistillationConfig):
        """Initialize PPO trainer.

        Args:
            config: Complete configuration for the experiment
        """
        self.config = config

        # Extract commonly used configs
        self.env_id = config.environment.env_id
        self.num_envs = config.environment.num_envs
        self.num_steps = config.environment.num_steps
        self.num_iterations = config.training.num_iterations
        self.learning_rate = config.training.learning_rate
        self.gamma = config.ppo.gamma
        self.gae_lambda = config.ppo.gae_lambda
        self.clip_coef = config.ppo.clip_coef
        self.vf_coef = config.ppo.vf_coef
        self.ent_coef = config.ppo.ent_coef
        self.max_grad_norm = config.ppo.max_grad_norm
        self.batch_size = config.training.batch_size
        self.n_epochs = config.training.n_epochs
        self.device = torch.device(config.experiment.device)
        self.seed = config.experiment.seed

        # Setup run name and directories
        run_name = config.experiment.name
        if run_name is None:
            run_name = f"{self.env_id}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.run_name = run_name
        self.log_dir = Path(config.experiment.log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        # Create environments
        from adaptive_rl.envs.make_env import make_vec_env

        self.envs = make_vec_env(
            self.env_id,
            self.num_envs,
            self.seed,
            config.experiment.capture_video,
            run_name,
            str(self.log_dir / "videos"),
        )

        # Create agent (student policy)
        self.agent = PPOAgent(
            self.envs.single_observation_space,
            self.envs.single_action_space,
            config.network.hidden_size,
            config.network.n_hidden_layers,
            config.network.activation,
        ).to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.learning_rate, eps=1e-5
        )

        # Create teacher if configured
        self.teacher = None
        teacher_dict = config.teacher.to_dict()
        if teacher_dict is not None and teacher_dict.get("type") is not None:
            teacher_kwargs = {k: v for k, v in teacher_dict.items() if k != "type"}
            self.teacher = create_teacher(
                teacher_type=teacher_dict["type"],
                env_id=self.env_id,
                action_space=self.envs.single_action_space,
                observation_space=self.envs.single_observation_space,
                device=config.experiment.device,
                **teacher_kwargs,
            )

        # Create scheduler
        scheduler_dict = config.scheduler.to_dict()
        self.scheduler = create_scheduler(
            num_envs=self.num_envs,
            device=config.experiment.device,
            log_dir=str(self.log_dir),
            **scheduler_dict,
        )

        # Create logger
        self.logger = Logger(
            run_name=run_name,
            log_dir=str(self.log_dir.parent),
            use_tensorboard=True,
            use_mlflow=False,
        )

        # Log hyperparameters
        self.logger.log_param("env_id", self.env_id)
        self.logger.log_param("num_envs", self.num_envs)
        self.logger.log_param("num_steps", self.num_steps)
        self.logger.log_param("learning_rate", self.learning_rate)
        self.logger.log_param("scheduler_strategy", config.scheduler.strategy)

    def train(self):
        """Main training loop."""
        # Initialize
        global_step = 0
        start_time = time.time()

        # Storage for rollout
        obs_buffer = torch.zeros(
            (self.num_steps, self.num_envs) + self.envs.single_observation_space.shape
        ).to(self.device)
        actions = torch.zeros(
            (self.num_steps, self.num_envs) + self.envs.single_action_space.shape
        ).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        # Start environments
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        # Track metrics
        episode_returns = []
        episode_lengths = []
        steps_since_reset = torch.zeros(self.num_envs)
        prev_reward = torch.full((self.num_envs,), -1.0)

        # Training loop
        for iteration in range(self.num_iterations):
            # Collect rollout
            for step in range(self.num_steps):
                global_step += self.num_envs
                obs_buffer[step] = next_obs

                # Choose policy type (teacher or student) for each env
                policy_types = self.scheduler.choose_policy_type(
                    iteration=iteration,
                    global_step=global_step,
                    steps_since_reset=steps_since_reset,
                    prev_reward=prev_reward,
                )

                # Get actions
                with torch.no_grad():
                    # Get student action and value
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )

                    # Override with teacher actions where specified
                    if self.teacher is not None:
                        teacher_mask = torch.tensor(
                            [p == "teacher" for p in policy_types],
                            dtype=torch.bool,
                            device=self.device,
                        )
                        if teacher_mask.any():
                            teacher_obs = next_obs[teacher_mask]
                            teacher_actions = self.teacher.act(teacher_obs)
                            teacher_actions = torch.tensor(teacher_actions).to(
                                self.device
                            )
                            action[teacher_mask] = teacher_actions

                            # Recalculate logprob for teacher actions
                            _, new_logprob, _, _ = self.agent.get_action_and_value(
                                next_obs, action
                            )
                            logprob = new_logprob

                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Step environment
                next_obs, reward, terminated, truncated, infos = self.envs.step(
                    action.cpu().numpy()
                )
                done = np.logical_or(terminated, truncated)
                rewards[step] = torch.tensor(reward).to(self.device)
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
                next_done = torch.tensor(done, dtype=torch.float32).to(self.device)
                dones[step] = next_done

                # Track metrics
                prev_reward = rewards[step].clone()
                steps_since_reset += 1
                steps_since_reset[done] = 0

                # Log episode statistics
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            episode_returns.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            self.logger.log_metrics(
                                "episode/return", info["episode"]["r"], global_step
                            )
                            self.logger.log_metrics(
                                "episode/length", info["episode"]["l"], global_step
                            )

            # Compute advantages using pure functional approach
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                gae_output = compute_gae(
                    rewards=rewards,
                    values=values,
                    next_values=next_value,
                    dones=dones,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                )
                advantages = gae_output.advantages
                returns = gae_output.returns

            # Flatten batch
            b_obs = obs_buffer.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # PPO update
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.n_epochs):
                np.random.shuffle(b_inds)
                for start in range(
                    0, self.batch_size, self.batch_size // self.n_epochs
                ):
                    end = start + self.batch_size // self.n_epochs
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )

                    # Use pure functional loss computation
                    loss_output = compute_ppo_losses(
                        logprobs_old=b_logprobs[mb_inds],
                        logprobs_new=newlogprob,
                        values_pred=newvalue,
                        advantages=b_advantages[mb_inds],
                        returns=b_returns[mb_inds],
                        entropy=entropy,
                        clip_coef=self.clip_coef,
                        vf_coef=self.vf_coef,
                        ent_coef=self.ent_coef,
                        values_old=b_values[mb_inds],
                        clip_vloss=False,  # Can be made configurable
                    )

                    # Extract losses from output
                    pg_loss = loss_output.policy_loss
                    v_loss = loss_output.value_loss
                    entropy_loss = loss_output.entropy_loss
                    approx_kl = loss_output.approx_kl
                    clipfracs.append(loss_output.clipfrac.item())

                    # Total loss
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

            # Log training metrics
            self.logger.log_metrics("loss/policy", pg_loss.item(), global_step)
            self.logger.log_metrics("loss/value", v_loss.item(), global_step)
            self.logger.log_metrics("loss/entropy", entropy_loss.item(), global_step)
            self.logger.log_metrics("loss/approx_kl", approx_kl.item(), global_step)
            self.logger.log_metrics("loss/clipfrac", np.mean(clipfracs), global_step)

            # Log scheduler statistics
            scheduler_stats = self.scheduler.get_statistics()
            for key, value in scheduler_stats.items():
                self.logger.log_metrics(f"scheduler/{key}", value, global_step)

            # Print progress
            if iteration % 10 == 0:
                if len(episode_returns) > 0:
                    avg_return = np.mean(episode_returns[-100:])
                    print(
                        f"Iteration {iteration}, Step {global_step}, Avg Return: {avg_return:.2f}"
                    )

            # Save checkpoint
            if iteration % self.config.experiment.save_freq == 0:
                self.save_checkpoint(iteration)

        # Final save
        self.save_checkpoint("final")
        self.envs.close()
        self.logger.close()

    def save_checkpoint(self, iteration):
        """Save model checkpoint."""
        checkpoint_path = self.log_dir / f"checkpoint_{iteration}.pt"
        torch.save(
            {
                "agent_state_dict": self.agent.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iteration": iteration,
                "scheduler_stats": self.scheduler.get_statistics(),
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")
