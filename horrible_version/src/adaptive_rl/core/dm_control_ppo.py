"""PPO with dm_control support and teacher-student scheduling.

Extension of ScheduledPPO to support continuous control environments like dm_control.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from src.adaptive_rl.core.scheduled_agent import ScheduledAgent
from src.adaptive_rl.schedulers.base import PolicyScheduler
from src.adaptive_rl.teachers.base import TeacherPolicy


def make_dm_control_env(env_factory, idx, capture_video, run_name):
    """Create dm_control environment thunk."""
    def thunk():
        env = env_factory()
        if capture_video and idx == 0:
            # TODO: Add video recording support for dm_control
            pass
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ContinuousAgent(nn.Module):
    """PPO agent for continuous action spaces."""

    def __init__(self, envs):
        super().__init__()

        # Handle both single env and vector env
        if hasattr(envs, 'single_observation_space'):
            obs_space = envs.single_observation_space
            action_space = envs.single_action_space
        else:
            obs_space = envs.observation_space
            action_space = envs.action_space

        # Network architecture
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_space.shape)), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class DMControlPPOTrainer:
    """PPO trainer for dm_control environments with teacher-student scheduling."""

    def __init__(
        self,
        args,
        scheduler: PolicyScheduler,
        env_factory,
        teacher: Optional[TeacherPolicy] = None,
        writer: Optional[SummaryWriter] = None
    ):
        self.args = args
        self.scheduler = scheduler
        self.teacher = teacher
        self.writer = writer
        self.env_factory = env_factory

        # Setup random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # Setup environment
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv(
            [make_dm_control_env(env_factory, i, args.capture_video, run_name)
             for i in range(args.num_envs)]
        )

        # Verify continuous action space
        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        # Setup student agent (PPO for continuous control)
        student_agent = ContinuousAgent(self.envs).to(self.device)

        # Wrap student agent to match BaseAgent interface
        class ContinuousAgentWrapper:
            def __init__(self, agent):
                self.agent = agent

            def get_action_and_value(self, obs, action=None):
                return self.agent.get_action_and_value(obs, action)

            def get_value(self, obs):
                return self.agent.get_value(obs)

        wrapped_student = ContinuousAgentWrapper(student_agent)

        # Wrap with scheduling capability
        self.agent = ScheduledAgent(
            student_agent=wrapped_student,
            teacher_policy=teacher,
            scheduler=scheduler,
            device=self.device
        )

        self.optimizer = optim.Adam(student_agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # Compute batch size and other derived parameters
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)

        # Setup storage
        obs_shape = self.envs.single_observation_space.shape
        action_shape = self.envs.single_action_space.shape

        self.obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(self.device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(self.device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(self.device)

        self.global_step = 0
        self.start_time = time.time()

    def train(self):
        """Run PPO training with teacher-student scheduling."""
        # Initialize environment
        next_obs = torch.Tensor(self.envs.reset()[0]).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)
        num_updates = self.args.total_timesteps // self.args.batch_size

        for update in range(1, num_updates + 1):
            # Annealing learning rate
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # Collect rollout data
            for step in range(0, self.args.num_steps):
                self.global_step += 1 * self.args.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # Get actions using scheduled policy
                with torch.no_grad():
                    action, logprob, _, value, scheduling_info = self.agent.get_action_and_value(
                        next_obs,
                        iteration=update,
                        global_step=self.global_step,
                        steps_since_reset=torch.zeros(self.args.num_envs),
                        prev_reward=torch.zeros(self.args.num_envs)  # TODO: Track episode rewards
                    )
                    self.values[step] = value.flatten()

                self.actions[step] = action
                self.logprobs[step] = logprob

                # Execute actions in environment
                next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

                # Logging
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={self.global_step}, episodic_return={info['episode']['r']}")
                            if self.writer:
                                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
                                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)

            # Bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.student.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            # Flatten the batch
            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.student.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.student.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

                if self.args.target_kl is not None:
                    if approx_kl > self.args.target_kl:
                        break

            # Logging
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if self.writer:
                self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
                self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
                self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
                self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
                self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
                self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
                self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
                self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)

                # Add scheduling metrics
                metrics = self.agent.get_scheduling_metrics()
                for key, value in metrics.items():
                    self.writer.add_scalar(f"scheduling/{key}", value, self.global_step)

            print(f"SPS: {int(self.global_step / (time.time() - self.start_time))}")

        self.envs.close()
        if self.writer:
            self.writer.close()