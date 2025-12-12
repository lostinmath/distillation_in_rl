# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import stable_baselines3 as sb3

# =================================================
# TODO: intergrate with my logging
# TODO: add & test mlflow
# TODO: I need my video recording instead
# TODO: change their eval. I need success rate!!

from octoplus.src.config_dataclasses.env_configs import ManiSkillConfig
from octoplus.src.env.mani_skill_env import get_envs
import mani_skill.envs
import imageio
from copy import deepcopy
from tqdm import tqdm
from octoplus.src.rl_models.nature_cnn import NatureCNN, BasicCritic, BasicActor
from octoplus.src.config_dataclasses.model_configs import TD3Config
from octoplus.src.config_dataclasses.experiments import ExperimentConfig
from octoplus.src.rl_models.shared_model_api import AbstractedRrAlgo
from octoplus.src.utils.general_utils import to_device

# from octoplus.src.rl_models.nature_cnn import NatureCNN
from octoplus.src.utils.custom_logging import Logger
from dataclasses import dataclass, fields
# from octoplus.src.schedulers.basic_scheduler import PolicyScheduler
from octoplus.src.utils.git_utils import write_git_diff_to_file, write_git_id_to_file
import mlflow

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
terminal_logger = logging.getLogger(__name__)


# class TD3Rgb(AbstractedRrAlgo):
#     def __init__(self, envs,
#                  cfg: ExperimentConfig,
#                  hydra_log_dir: str = "run"):
#         self.cfg = cfg
#         self.env_cfg: ManiSkillConfig = cfg.env_config
#         self.model_cfg: TD3Config = cfg.model_config
#         self.envs = envs
#         self.device = cfg.device
#         self.hydra_log_dir = hydra_log_dir

#     def model_setup(self):
#         # put logger here, because should use mlflow for training and should not for eval
#         self.logger = Logger(
#             log_file=f"{self.hydra_log_dir}/{self.model_cfg.run_name}/train_metrics.txt",
#             use_mlflow=True,
#         )
#         # logging hp to mlflow
#         for field in fields(self.cfg.model_config):
#             self.logger.log_param(
#                 field.name, getattr(self.cfg.model_config, field.name)
#             )
#         # seeding
#         random.seed(self.model_cfg.seed)
#         np.random.seed(self.model_cfg.seed)
#         torch.manual_seed(self.model_cfg.seed)
#         torch.backends.cudnn.deterministic = self.model_cfg.torch_deterministic
#         next_obs, _ = self.envs.reset(seed=self.model_cfg.seed)
#         next_obs = to_device(next_obs, self.device)
#         self.scheduler = PolicyScheduler(cfg=self.cfg, envs=self.envs,
#                                          next_obs=next_obs, hydra_log_dir=self.hydra_log_dir)


#     def model_eval_setup(self):
#         # put logger here, becauese should use mlflow for training and should not for eval
#         self.logger = Logger(
#             log_file=f"{self.hydra_log_dir}/{self.model_cfg.run_name}/train_metrics.txt",
#             use_mlflow=False,
#         )
#         random.seed(self.args.seed)
#         np.random.seed(self.args.seed)
#         torch.manual_seed(self.args.seed)
#         torch.backends.cudnn.deterministic = self.args.torch_deterministic
#         self.device = torch.device(self.cfg.device)

#         next_obs, _ = self.envs.reset(seed=self.args.seed)

#         for key in next_obs:
#             next_obs[key] = next_obs[key].to(device=self.device)
#         # for evaluation scheduling_strategy must be internal_only
#         assert self.cfg.scheduler_config.scheduling_strategy == "internal_only"
#         self.scheduler = PolicyScheduler(cfg=self.cfg, envs=self.envs, next_obs=next_obs,
#                                          hydra_log_dir=self.hydra_log_dir)


#     def train(self):
#         pass


SAMPLE_ENV_CONFIG = {
    "obs_mode": "rgb",
    "control_mode": "pd_ee_delta_pose",
    "render_mode": "rgb_array",  # rgb_array
    "sim_backend": "gpu",  # "cpu" # "gpu"
    "reward_mode": "dense",
    "env_id": "PickCube-v1",  # "StackCube-v1",
    # PushCube-v1 # PickSingleYCB-v1 "PushCube-v1" "PokeCube-v1" "PickCube-v1" "StackCube-v1" "PullCube-v1" "TableTopFreeDraw-v1"
    "num_envs": 1,
    # Lets make always 1 for now for debugging, TODO: make the code for multiple env
    # "num_steps": 100,  # we actually don't use it here
    "include_state": False,
    "partial_reset": True,
    "image_height": 256,
    "image_width": 256,
    "use_render_camera_as_input": False,
    "render_mode": "rgb_array",
}
# =================================================

# ENV_CONFIG = ManiSkillConfig(**SAMPLE_ENV_CONFIG)


# def make_env_mani_skill_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         if capture_video and idx == 0:
#             env = get_envs(ENV_CONFIG)
#             # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = get_envs(ENV_CONFIG)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env.action_space.seed(seed)
#         return env

#     return thunk


class NatureCNNEncoder(nn.Module):
    """
    A neural network module that processes image and state observations using a NatureCNN architecture.
    Attributes:
        out_features (int): The total number of output features from the network.
        extractors (nn.ModuleDict): A dictionary containing the feature extractors for different types of observations.
    Methods:
        __init__(sample_obs):
            Initializes the NatureCNN with the given sample observations.
        forward(observations) -> torch.Tensor:
            Forward pass through the network, processing the observations and returning the encoded tensor.
    """

    def __init__(self, sample_obs):
        super().__init__()

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs.shape[-1]
        image_size = (sample_obs.shape[1], sample_obs.shape[2])

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs.float().permute(0, 3, 1, 2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        self.rgb_encoder = nn.Sequential(cnn, fc)
        self.out_features += feature_size

    def forward(self, observations) -> torch.Tensor:
        obs = observations.float().permute(0, 3, 1, 2)
        obs = obs / 255
        obs = self.rgb_encoder(obs)
        # print(f"NatureNN encoder output.shape={obs.shape}")
        return obs


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()
        # we encode only the image
        self.encoder = NatureCNNEncoder(sample_obs=sample_obs)

        latent_size = self.encoder.out_features
        # print(f"latent_size of QNetwork is: {latent_size}")

        self.fc1 = nn.Linear(
            latent_size + 7, 256
        )  # img encoding is concated with action
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, a):
        x = self.encoder(x)
        # x = x.reshape((x.shape[0], -1)).float()
        # a.shape=torch.Size([256, 7]) x.shape=torch.Size([256, 256])
        # print(f"a.shape={a.shape} x.shape={x.shape}")
        x = torch.cat([x, a], 1)  # adding noise? what is a?
        # print(f"x.shape after concat={x.shape}")
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __call__(self, x, a):
        return self.forward(x, a)


class Actor(nn.Module):
    def __init__(self, sample_obs, env):
        super().__init__()
        self.encoder = NatureCNNEncoder(sample_obs=sample_obs)

        latent_size = self.encoder.out_features
        print(f"latent_size of QNetwork is: {latent_size}")

        self.fc1 = nn.Linear(latent_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_mu = nn.Linear(64, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        # x = x.reshape((x.shape[0], -1)).float()
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def run_td3(cfg: ExperimentConfig, hydra_output_dir: str):
    # args = tyro.cli(TD3Config)
    args: TD3Config = cfg.model_config
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = cfg.device
    # torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    logger = Logger(
        log_file=f"{hydra_output_dir}/{args.run_name}/train_metrics.txt",
        use_mlflow=True,
    )

    # env setup
    envs = get_envs(cfg.env_config)

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # about 3 gb of gpu mem for every env?
    obs, _ = envs.reset(seed=args.seed)
    print(f"env action_space: {envs.action_space}")
    print(f"env observation_space: {envs.observation_space}")

    actor = Actor(obs, envs).to(device)
    qf1 = QNetwork(obs).to(device)
    qf2 = QNetwork(obs).to(device)
    qf1_target = QNetwork(obs).to(device)
    qf2_target = QNetwork(obs).to(device)
    target_actor = Actor(obs, envs).to(device)

    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    # this one occupies the most memory!!
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    recording = []
    cur_successes = 0
    cur_finished_runs = 0
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)  # dtype uint8, [b, 256, 256, 3]

    for global_step in tqdm(
        range(args.total_timesteps)
    ):  # range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:  # WHY???
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actions = actor(
                    torch.Tensor(obs).to(device)
                )  # !!!! sometimes instead of actor put the octo actions here
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = (
                    actions.cpu()
                    .numpy()
                    .clip(envs.single_action_space.low, envs.single_action_space.high)
                )

        # TODO: need a 50 steps loop here? so I can create videos and log rewards
        # env will be reset after 50 steps

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TODO: save recording only of cert freq or last steps?
        if global_step > args.total_timesteps - 200:
            recording.append(next_obs[0].cpu())

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            # Adding here custom reward and cucess prossing
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]

            if "episode_len" in final_info["episode"].keys():
                print(
                    f"episode_len",
                    final_info["episode"]["episode_len"][done_mask]
                    .float()
                    .mean()
                    .item(),
                    global_step,
                )
            num_of_new_finished_runs = torch.logical_or(terminations, truncations).sum()
            cur_finished_runs += num_of_new_finished_runs
            cur_successes += terminations.sum()

            cur_reward = 0  # on first iteration set it to unknown
            if "reward" in final_info["episode"].keys():
                cur_reward = (
                    final_info["episode"]["reward"][done_mask].float().mean().item()
                )
            print(
                f"Success rate: {(cur_successes / cur_finished_runs):.3f} Reward: {cur_reward:.3f}"
            )
            # added some loggining
            logger.log_metrics(
                f"training/success_rate", cur_successes / cur_finished_runs, global_step
            )
            logger.log_metrics(f"training/cur_reward", cur_reward, global_step)

            # for info in infos["final_info"]: # infos["final_info"] is a dict in my case
            #     print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #     writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #     writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            #     break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = deepcopy(next_obs.cpu())  # .copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        infos = to_device(infos, "cpu")
        rb.add(
            obs.cpu(), real_next_obs, actions, rewards.cpu(), terminations.cpu(), infos
        )
        # ! here octo actions will go into buffer
        # do we possibly need a separate buffer for them? Than we are sure from where we sample

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # CUDA error: unspecified launch failure ?
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (
                    torch.randn_like(data.actions, device=device) * args.policy_noise
                ).clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale
                # does this noise need to change as training progresses?

                next_state_actions = (
                    target_actor(data.next_observations) + clipped_noise
                ).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                # print(f"next_state_actions.shape={next_state_actions.shape}")
                # print(f"next_state_actions={next_state_actions}")
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 10 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
            if global_step % 100_000 == 0:
                actor_checkpoint_path = f"{hydra_output_dir}/{args.run_name}/{global_step}_actor_checkpoint.pth"
                actor_tgt_checkpoint_path = f"{hydra_output_dir}/{args.run_name}/{global_step}_actor_tgt_checkpoint.pth"
                qf1_checkpoint_path = f"{hydra_output_dir}/{args.run_name}/{global_step}_qf1_checkpoint.pth"
                qf2_checkpoint_path = f"{hydra_output_dir}/{args.run_name}/{global_step}_qf2_checkpoint.pth"
                qf1_tgt_checkpoint_path = f"{hydra_output_dir}/{args.run_name}/{global_step}_qf1_tgt_checkpoint.pth"
                qf2_tgt_checkpoint_path = f"{hydra_output_dir}/{args.run_name}/{global_step}_qf2_tgt_checkpoint.pth"
                torch.save(actor.state_dict(), actor_checkpoint_path)
                torch.save(target_actor.state_dict(), actor_tgt_checkpoint_path)
                torch.save(qf1.state_dict(), qf1_checkpoint_path)
                torch.save(qf2.state_dict(), qf2_checkpoint_path)
                torch.save(qf1_target.state_dict(), qf1_tgt_checkpoint_path)
                torch.save(qf2_target.state_dict(), qf2_tgt_checkpoint_path)
    # time logging
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"Time spent: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"
    )
    # recording
    numpy_images = [tensor.cpu().numpy().astype(np.uint8) for tensor in recording]
    # Save the video using imageio
    with imageio.get_writer(
        f"last_steps_recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4", fps=30
    ) as writer:
        for image in numpy_images:
            writer.append_data(image)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
