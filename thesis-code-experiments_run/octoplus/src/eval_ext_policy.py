import datetime
import math
import os
from collections import defaultdict
from typing import Any

import hydra
import imageio
import mani_skill.envs
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from agents.client import RemoteAgent
from agents.policies import ObsDict
from octoplus.src.config_dataclasses.experiments import ExperimentConfig
from octoplus.src.env.mani_skill_env import get_envs
from octoplus.src.evaluate_main import save_video_from_tensors
from octoplus.src.utils.utils_for_octo import reshape_obs_for_octo

torch.backends.cuda.preferred_linalg_library("magma")

# SUPPORTED_RENDER_MODES = ("human", "rgb_array", "sensors", "all")
PATH_TO_CONFIG = "/home/piscenco/thesis2/thesis-code/configs/ppo_config.yaml"


class EvaluationEnv:
    def __init__(self):
        cfg = OmegaConf.load(PATH_TO_CONFIG)
        del cfg["defaults"]  # this is hydra args
        self.config = ExperimentConfig(**cfg)
        # some constants:
        self.config.model_config.env_id = "PickCube-v1"
        self.external_policy = RemoteAgent.by_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config.env_config.sim_backend = "cpu"  # because locally cuda out of mem

    def get_remote_action(self, next_obs: dict[str, Any]) -> list[torch.Tensor]:
        """
        Get action using Agent package
        """
        reshaped_obs = reshape_obs_for_octo(next_obs["rgb"])
        recommended_actions = torch.zeros((len(reshaped_obs), 7))  # 7 is for franka
        for i, obs_img in enumerate(reshaped_obs):
            # next_obs = ObsDict(image_primary=obs_img, dtype=np.uint8)
            new_next_obs = {"image_primary": obs_img}
            self.external_policy.reset(
                new_next_obs, self.config.model_config.text_instruction
            )
            result = self.external_policy.act(new_next_obs)
            recommended_actions[i] = torch.tensor(
                result[0]["action"][0], device=self.device
            )
        return recommended_actions

    def evaluate_external_policy(
        self,
        total_num_env: int = 1,
        num_parallel: int = 1,
    ):
        """
        Function for evaluating the remote agent performance
        """
        self.config.model_config.num_envs = num_parallel

        eval_envs = get_envs(
            model_config=self.config.model_config,
            env_config=self.config.env_config,
            hydra_log_dir="../../outputs_evaluate",
        )
        # eval_envs.render_mode = "rgb_array"

        # now run the evaluation
        eval_metrics = defaultdict(list)
        metrics_history = []
        # Calculate the number of environments per parallel worker
        total_num_runs = math.ceil(total_num_env / num_parallel)

        total_number_of_finished_runs = 0
        total_number_of_successes = 0

        eval_obs, _ = eval_envs.reset(seed=34)
        recording = []
        recording.append(eval_obs["rgb"][0])  # we save only the first one

        progress_bar = tqdm(range(1, total_num_runs), desc="Evaluation")

        while total_number_of_finished_runs < total_num_runs:
            with torch.no_grad():

                (
                    eval_obs,
                    eval_rew,
                    eval_terminations,
                    eval_truncations,
                    eval_infos,
                ) = eval_envs.step(self.get_remote_action(eval_obs))
                recording.append(eval_obs["rgb"][0])

                if "final_info" in eval_infos:
                    mask = eval_infos["_final_info"]
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)
                    num_of_new_finished_runs = torch.logical_or(
                        eval_terminations, eval_truncations
                    ).sum()
                    total_number_of_finished_runs += num_of_new_finished_runs
                    total_number_of_successes += eval_terminations.sum()
                    postfix_str = f"Success rate: {total_number_of_successes/ total_number_of_finished_runs}. Finished runs: {total_number_of_finished_runs}."
                    progress_bar.set_postfix_str(postfix_str)
                    progress_bar.update(num_of_new_finished_runs.item())
                    # eval_infos.get("success", 0)

        # Save the recording as a video
        current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = "Fine_tuned_octo_upper_camera"
        video_path = f"evaluation_videos/{experiment_name}_{current_datetime_str}.mp4"
        # Ensure the directory exists
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        # Get the dimensions of the images
        height = recording[0].shape[0]
        width = recording[0].shape[1]

        numpy_images = [
            tensor[0].cpu().numpy().astype(np.uint8) for tensor in recording
        ]

        # Save the video using imageio
        with imageio.get_writer(video_path, fps=30) as writer:
            for image in numpy_images:
                writer.append_data(image)

        print(f"Saved evaluation video at {video_path}")

        metrics_history.append(eval_metrics)
        for k, v in eval_metrics.items():
            mean = torch.stack(v).float().mean()
            print(f"eval/{k}", mean.item())
            print(f"eval_{k}_mean={mean.item()}")

        print(
            "Success rate: ",
            (total_number_of_successes / total_number_of_finished_runs).item(),
        )
        print("Total number of finished runs: ", total_number_of_finished_runs.item())
        print("Total number of successes: ", total_number_of_successes.item())
        eval_envs.close()


if __name__ == "__main__":
    env = EvaluationEnv()
    env.evaluate_external_policy()
