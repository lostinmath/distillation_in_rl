"""PPO with algorithm-agnostic teacher-student scheduling.

Minimal modification to CleanRL PPO - only changes action selection
while preserving all PPO training logic exactly.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from src.adaptive_rl.algorithms.cleanrl.ppo import Agent, layer_init, make_env
from src.adaptive_rl.core.scheduled_agent import ScheduledAgent, CleanRLAgentWrapper
from src.adaptive_rl.schedulers.base import PolicyScheduler
from src.adaptive_rl.teachers.base import TeacherPolicy


@dataclass
class ScheduledPPOArgs:
    """PPO + Scheduling configuration."""
    # Experiment
    exp_name: str = "scheduled_ppo"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "adaptive-rl"
    wandb_entity: str = None
    capture_video: bool = False

    # Environment
    env_id: str = "CartPole-v1"
    num_envs: int = 4

    # PPO Algorithm (CleanRL defaults)
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # Scheduling (new)
    use_teacher: bool = True
    scheduler_type: str = "reward_based"
    teacher_type: str = "optimal"

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class ScheduledPPOTrainer:
    """PPO trainer with algorithm-agnostic scheduling.

    Key Design:
    - Minimal changes to CleanRL PPO training loop
    - Scheduling happens at action selection only
    - All PPO loss calculations use student policy
    - Teacher actions applied to environment, not training
    """

    def __init__(
        self,
        args: ScheduledPPOArgs,
        scheduler: PolicyScheduler,
        teacher: Optional[TeacherPolicy] = None,
        writer: Optional[SummaryWriter] = None
    ):
        self.args = args
        self.scheduler = scheduler
        self.teacher = teacher
        self.writer = writer

        # Compute derived args
        self.args.batch_size = int(args.num_envs * args.num_steps)
        self.args.minibatch_size = int(args.batch_size // args.num_minibatches)
        self.args.num_iterations = args.total_timesteps // args.batch_size

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # Setup environment
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        # Setup student agent (CleanRL PPO)
        student_agent = Agent(self.envs).to(self.device)
        wrapped_student = CleanRLAgentWrapper(student_agent)

        # Setup scheduled agent
        self.agent = ScheduledAgent(
            student_agent=wrapped_student,
            teacher_policy=teacher,
            scheduler=scheduler,
            device=self.device
        )

        # Setup optimizer
        self.optimizer = optim.Adam(student_agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # Storage (exactly as in CleanRL)
        self.obs = torch.zeros((args.num_steps, args.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + self.envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(self.device)

        # Scheduling metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.scheduling_metrics = []
        self.start_time = None

        # Track episode rewards for scheduler
        self.current_episode_rewards = torch.zeros(args.num_envs).to(self.device)
        self.last_episode_rewards = torch.full((args.num_envs,), -1.0).to(self.device)  # -1 means no previous episode

    def train(self):
        """Main training loop - minimal modification of CleanRL PPO."""
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        self.start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)

        for iteration in range(1, self.args.num_iterations + 1):
            # Annealing the rate if instructed to do so
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.args.num_steps):
                global_step += self.args.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # MODIFIED: Use scheduled agent instead of direct agent
                with torch.no_grad():
                    # Get scheduled action (teacher or student)
                    action, logprob, _, value, scheduling_info = self.agent.get_action_and_value(
                        next_obs,
                        iteration=iteration,
                        global_step=global_step,
                        steps_since_reset=torch.zeros(self.args.num_envs),  # Track steps since reset if needed
                        prev_reward=self.last_episode_rewards  # Pass actual episode rewards
                    )
                    self.values[step] = value.flatten()

                self.actions[step] = action
                self.logprobs[step] = logprob

                # Execute environment step (action might be from teacher or student)
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                reward_tensor = torch.tensor(reward).to(self.device).view(-1)
                self.rewards[step] = reward_tensor

                # Update episode reward tracking
                self.current_episode_rewards += reward_tensor

                # When episode ends, save the total episode reward
                for env_idx in range(self.args.num_envs):
                    if next_done[env_idx]:
                        self.last_episode_rewards[env_idx] = self.current_episode_rewards[env_idx]
                        self.current_episode_rewards[env_idx] = 0.0

                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

                # Log episode results and scheduling metrics
                if "final_info" in infos:
                    self._log_episode_results(infos, global_step, scheduling_info)

            # PPO Update (UNCHANGED from CleanRL)
            self._ppo_update(global_step, next_obs, next_done)

            # Log scheduling metrics
            if self.writer and iteration % 10 == 0:
                self._log_scheduling_metrics(global_step)

        self.envs.close()
        if self.writer:
            self.writer.close()

    def _prepare_episode_info(self, step: int, iteration: int) -> Dict[str, Any]:
        """Prepare episode information for scheduler."""
        return {
            "step": step,
            "iteration": iteration,
            "recent_rewards": self.episode_rewards[-5:] if self.episode_rewards else [],
            "recent_lengths": self.episode_lengths[-5:] if self.episode_lengths else []
        }

    def _log_episode_results(self, infos: Dict, global_step: int, scheduling_info: Dict[str, Any]):
        """Log episode results and update scheduler."""
        for info in infos["final_info"]:
            if info and "episode" in info:
                episode_return = info["episode"]["r"]
                episode_length = info["episode"]["l"]

                self.episode_rewards.append(episode_return)
                self.episode_lengths.append(episode_length)

                # Log to tensorboard
                if self.writer:
                    self.writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    self.writer.add_scalar("charts/episodic_length", episode_length, global_step)

                print(f"global_step={global_step}, episodic_return={episode_return}")

                # Update scheduler with episode results
                if hasattr(self.scheduler, 'update_episode_results'):
                    self.scheduler.update_episode_results({
                        "reward": episode_return,
                        "length": episode_length,
                        "global_step": global_step
                    })

    def _log_scheduling_metrics(self, global_step: int):
        """Log scheduling-specific metrics."""
        metrics = self.agent.get_scheduling_metrics()
        if not metrics:
            return

        if self.writer:
            self.writer.add_scalar("scheduling/teacher_ratio", metrics.get("teacher_ratio", 0), global_step)
            self.writer.add_scalar("scheduling/student_ratio", metrics.get("student_ratio", 1), global_step)
            self.writer.add_scalar("scheduling/policy_switches", metrics.get("policy_switches", 0), global_step)

    def _ppo_update(self, global_step: int, next_obs: torch.Tensor, next_done: torch.Tensor):
        """PPO update - EXACTLY as in CleanRL (no modifications)."""
        # Bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
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

                # CRITICAL: Use student agent for training (not scheduled agent)
                _, newlogprob, entropy, newvalue = self.agent.student.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
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

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break

        # Logging (same as CleanRL)
        if self.writer:
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)

            print("SPS:", int(global_step / (time.time() - self.start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.start_time)), global_step)