from pathlib import Path

import gymnasium as gym
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from adaptive_rl.algorithms import AlgorithmRegistry
from adaptive_rl.envs import make_vec_env
from adaptive_rl.schedulers import SCHEDULERS
from adaptive_rl.teachers import create_teacher
from adaptive_rl.tracking import ExperimentTracker, TrackerConfig

# TODO: not sure how clean is the builder pattern, need to keep steps functional and to provide and abstaction for env and teachers.
class PipelineBuilder:
    @staticmethod
    def build_env(cfg: DictConfig):
        return make_vec_env(
            env_id=cfg.environment.env_id,
            num_envs=cfg.environment.num_envs,
            seed=cfg.experiment.seed,
        )

    @staticmethod
    def build_algorithm(cfg: DictConfig, env: gym.Env):
        algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
        algorithm_name = algorithm_cfg.pop("name")
        algorithm_cfg.pop("_target_", None)

        network_cfg = algorithm_cfg.pop("network", None)
        if network_cfg and "_target_" in network_cfg:
            network_class = hydra.utils.get_class(network_cfg["_target_"])
            network_cfg.pop("_target_")

            import numpy as np

            # Handle observation space correctly for vectorized environments
            if (
                hasattr(env.observation_space, "shape")
                and len(env.observation_space.shape) > 0
            ):
                # For vectorized envs, we get (n_envs, obs_dim) so take the last dimension
                if len(env.observation_space.shape) == 2:
                    obs_dim = env.observation_space.shape[1]  # Individual env obs dim
                else:
                    obs_dim = int(np.prod(env.observation_space.shape))
            else:
                obs_dim = (
                    env.observation_space.n
                    if hasattr(env.observation_space, "n")
                    else 1
                )

            # Handle action space correctly for vectorized environments
            if hasattr(env.action_space, "n"):
                action_dim = env.action_space.n
                discrete = True
            elif hasattr(env.action_space, "nvec"):
                # MultiDiscrete case - take the first action space
                action_dim = env.action_space.nvec[0]  # All envs should be the same
                discrete = True
            else:
                action_dim = int(np.prod(env.action_space.shape))
                discrete = False

            network = network_class(
                obs_dim=obs_dim, action_dim=action_dim, discrete=discrete, **network_cfg
            )
        else:
            network = None

        algorithm = AlgorithmRegistry.create(
            algorithm_name,
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            device=cfg.experiment.device,
            seed=cfg.experiment.seed,
            **algorithm_cfg,
        )
        return algorithm

    @staticmethod
    def build_teacher(cfg: DictConfig, env: gym.Env):
        if not cfg.get("teacher"):
            return None

        teacher_cfg = OmegaConf.to_container(cfg.teacher, resolve=True)
        teacher_name = teacher_cfg.get("name", "optimal")

        return create_teacher(
            teacher_type=teacher_name,
            env_id=cfg.environment.env_id,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

    @staticmethod
    def build_scheduler(cfg: DictConfig, student, teacher):
        scheduler_cfg = OmegaConf.to_container(cfg.scheduler, resolve=True)
        scheduler_name = scheduler_cfg.get("name", "student_only")

        if scheduler_name in SCHEDULERS:
            scheduler_class = SCHEDULERS[scheduler_name]
        else:
            scheduler_class = hydra.utils.get_class(scheduler_cfg["_target_"])

        kwargs = {
            k: v for k, v in scheduler_cfg.items() if k not in ["name", "_target_"]
        }

        # Check if scheduler needs specific initialization
        init_kwargs = {
            "student_policy": student,
            "teacher_policy": teacher,
            "num_envs": cfg.environment.num_envs,
            **kwargs,
        }

        return scheduler_class(**init_kwargs)

    @staticmethod
    def build_tracker(cfg: DictConfig):
        tensorboard_cfg = dict(cfg.tracker.get("tensorboard", {}))
        tensorboard_cfg["log_dir"] = cfg.paths.log_dir

        csv_cfg = dict(cfg.tracker.get("csv", {}))
        csv_cfg["log_dir"] = cfg.paths.log_dir

        tracker_cfg = TrackerConfig(
            backends=cfg.tracker.get("backends", ["tensorboard"]),
            tensorboard=tensorboard_cfg,
            mlflow=cfg.tracker.get("mlflow", {"enabled": False}),
            wandb=cfg.tracker.get("wandb", {"enabled": False}),
            neptune=cfg.tracker.get("neptune", {"enabled": False}),
            console=cfg.tracker.get("console", {"enabled": True}),
            csv=csv_cfg,
        )

        return ExperimentTracker(
            config=tracker_cfg,
            run_name=cfg.experiment.name,
        )

    @classmethod
    def build_pipeline(cls, cfg: DictConfig):
        components = {}

        components["env"] = cls.build_env(cfg)
        components["algorithm"] = cls.build_algorithm(cfg, components["env"])
        components["teacher"] = cls.build_teacher(cfg, components["env"])
        components["scheduler"] = cls.build_scheduler(
            cfg, components["algorithm"], components["teacher"]
        )
        components["tracker"] = cls.build_tracker(cfg)

        return ExperimentPipeline(**components, config=cfg)


class ExperimentPipeline:
    def __init__(
        self,
        env: gym.Env,
        algorithm,
        scheduler,
        tracker: ExperimentTracker,
        config: DictConfig,
        teacher=None,
    ):
        self.env = env
        self.algorithm = algorithm
        self.scheduler = scheduler
        self.tracker = tracker
        self.config = config
        self.teacher = teacher

        self.device = config.experiment.device
        self.total_timesteps = config.training.total_timesteps
        self.eval_freq = config.training.eval_freq
        self.checkpoint_freq = config.training.checkpoint_freq

        Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        n_envs = self.config.environment.num_envs
        n_steps = self.config.algorithm.n_steps

        obs = torch.tensor(self.env.reset()[0], device=self.device, dtype=torch.float32)
        episode_rewards = torch.zeros(n_envs, device=self.device)
        episode_lengths = torch.zeros(n_envs, device=self.device)
        episode_count = 0

        rollout_buffer = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "log_probs": [],
        }

        timesteps_done = 0

        while timesteps_done < self.total_timesteps:
            for step in range(n_steps):
                # Use the scheduler's choose_policy_type method
                policies = self.scheduler.choose_policy_type(
                    iteration=timesteps_done // n_steps,
                    global_step=timesteps_done,
                    steps_since_reset=torch.zeros(
                        n_envs
                    ),  # We'll track this properly later
                    prev_reward=episode_rewards,
                )
                policy_source = policies[0]  # For now, use same policy for all envs

                if policy_source == "teacher" and self.teacher:
                    action = self.teacher.act(obs.cpu().numpy())
                    action = torch.tensor(action, device=self.device)
                    value = torch.zeros(n_envs, 1, device=self.device)
                    log_prob = torch.zeros(n_envs, device=self.device)
                else:
                    action, value, log_prob = self.algorithm.predict(obs)

                next_obs, reward, terminated, truncated, info = self.env.step(
                    action.cpu().numpy()
                )
                done = terminated | truncated

                rollout_buffer["observations"].append(obs)
                rollout_buffer["actions"].append(action)
                rollout_buffer["rewards"].append(
                    torch.tensor(reward, device=self.device)
                )
                rollout_buffer["dones"].append(torch.tensor(done, device=self.device))
                rollout_buffer["values"].append(
                    value.squeeze(-1)
                )  # Remove last dimension
                rollout_buffer["log_probs"].append(log_prob)

                obs = torch.tensor(next_obs, device=self.device, dtype=torch.float32)
                episode_rewards += torch.tensor(reward, device=self.device)
                episode_lengths += 1

                for i, d in enumerate(done):
                    if d:
                        self.tracker.log_metrics(
                            {
                                "episode/return": float(episode_rewards[i]),
                                "episode/length": float(episode_lengths[i]),
                            },
                            step=timesteps_done,
                        )
                        episode_rewards[i] = 0
                        episode_lengths[i] = 0
                        episode_count += 1

                timesteps_done += n_envs

            # Convert rollout buffer to proper tensor format
            rollout_data = {}
            for key, values in rollout_buffer.items():
                if key == "observations":
                    rollout_data[key] = torch.stack(
                        values
                    )  # (n_steps, n_envs, obs_dim)
                elif (
                    key == "actions"
                    or key == "rewards"
                    or key == "dones"
                    or key == "values"
                    or key == "log_probs"
                ):
                    rollout_data[key] = torch.stack(values)  # (n_steps, n_envs)

            rollout_data["next_observations"] = obs

            train_metrics = self.algorithm.train_step(rollout_data)

            scheduler_metrics = self.scheduler.get_metrics()
            all_metrics = {**train_metrics, **scheduler_metrics}
            self.tracker.log_metrics(all_metrics, step=timesteps_done)

            rollout_buffer = {key: [] for key in rollout_buffer}

            if timesteps_done % self.eval_freq == 0:
                eval_metrics = self.evaluate()
                self.tracker.log_metrics(eval_metrics, step=timesteps_done)

            if timesteps_done % self.checkpoint_freq == 0:
                self.save_checkpoint(timesteps_done)

            progress = timesteps_done / self.total_timesteps
            print(
                f"Progress: {progress:.1%} | Episodes: {episode_count} | Timesteps: {timesteps_done}"
            )

        self.tracker.close()
        print(f"Training complete! Total episodes: {episode_count}")

    def evaluate(self, n_eval_episodes: int = 10):
        eval_rewards = []
        eval_lengths = []

        for _ in range(n_eval_episodes):
            obs = self.env.reset()[0]
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
                action, _, _ = self.algorithm.predict(obs_tensor, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(
                    action.cpu().numpy()
                )
                done = terminated.any() | truncated.any()
                episode_reward += reward.sum()
                episode_length += 1

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

        return {
            "eval/mean_return": float(np.mean(eval_rewards)),
            "eval/std_return": float(np.std(eval_rewards)),
            "eval/mean_length": float(np.mean(eval_lengths)),
        }

    def save_checkpoint(self, timestep: int):
        checkpoint_path = (
            Path(self.config.paths.checkpoint_dir) / f"checkpoint_{timestep}.pt"
        )
        self.algorithm.save(str(checkpoint_path))
        print(f"Saved checkpoint to {checkpoint_path}")
