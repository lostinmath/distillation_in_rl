######################################################################################
import argparse
import logging
import math
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime

import gymnasium as gym
import imageio
import numpy as np

# import torch
import yaml
from tqdm import tqdm
from agents.client import RemoteAgent
from agents.policies import Obs
import torch
from typing import Any
import mani_skill.envs

from octoplus.src.utils.utils_for_octo import reshape_obs_for_octo
from octoplus.src.config_dataclasses.env_configs import ManiSkillConfig
from octoplus.src.env.mani_skill_env import get_envs

from octoplus.src.rl_models.ppo_rgb_old import OctoPolicy

OCTO_PORT = 10_000
SAMPLE_ENV_CONFIG = {
    "obs_mode": "rgb",
    "control_mode": "pd_ee_delta_pose",
    "render_mode": "rgb_array", # rgb_array
    "sim_backend": "gpu",  # "cpu" # "gpu"
    "reward_mode": "dense",
    "env_id": "PushCube-v1", # "StackCube-v1", 
    # PushCube-v1 # PickSingleYCB-v1 "PushCube-v1" "PokeCube-v1" "PickCube-v1" "StackCube-v1" "PullCube-v1" "TableTopFreeDraw-v1"
    "num_envs": 1,  
    # Lets make always 1 for now for debugging, TODO: make the code for multiple env
    # "num_steps": 100,  # we actually don't use it here
    "include_state": False,
    "partial_reset": True,
    "image_height": 256,
    "image_width": 256,
    "use_render_camera_as_input": True
}

DEFAULT_INSTRUCTION = "pick up the cube"
# "Pick the red cube and move in the position of the green ball."
# "Stack cubes."
#"Poke the cube."
# "Pick the red cube and move in the position of the green ball."
# "Push cube to the red/white circular target."
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


class ExternalPolicyInference:
    def __init__(self):
        
        self.OCTO_INPUT_WIDTH = 256
        self.OCTO_INPUT_HEIGHT = 256
        self.external_policy = None
        
        # RemoteAgent("localhost", OCTO_PORT, "octodist")
        print("Created agent for external policy.")
        self.device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.external_policy_type = "octodist"
        
    def __get_external_policy_action__(
        self, next_obs:torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Returns the action from the external policy.
        """
        if self.external_policy_type == "octodist":
            reshaped_obs = reshape_obs_for_octo(next_obs)
            self.external_policy.reset(reshaped_obs, DEFAULT_INSTRUCTION)
            
            result = self.external_policy.act({"image_primary": reshaped_obs[0]}) # rhis will also break if not seq
            recommended_actions = torch.tensor(np.array(result[0]['action'][0]), device=self.device)
        else:
            raise NotImplementedError
        return recommended_actions


    def forward(self, obs):
        return self.__get_external_policy_action__(obs)

    
    def initialize_envs(self, env_config: dict[str, Any]):
        self.env_config = ManiSkillConfig(**env_config)
        self.envs = get_envs(self.env_config)
        return self.envs

    def evaluate(self, 
                 runs_number: int = 8,
                 experiment_name:str="default_experiment"):
                 
        if not hasattr(self, "envs"):
            raise ValueError("The environment has not been initialized.")

        eval_obs, _= self.envs.reset()
        total_number_of_finished_runs = 0
        total_number_of_successes = 0
        eval_metrics = defaultdict(list)
        total_reward = 0
        total_steps = 0
        cur_step_since_reset = 0
        eval_obs, _ = self.envs.reset()
        recording = []
        progress_bar = tqdm(range(1, runs_number), desc="Evaluation")
        runs_reward_history = np.zeros((runs_number, 50)) -1 # initially fill with -1
        cur_run_id = 0
        env_was_reset_on_prev_step = True
        eval_obs, _ = self.envs.reset(seed=11)
        self.external_policy = OctoPolicy( env_id=self.env_config.env_id, 
                                                instruction="push the cube away from the robot base",
                                                batch_example=eval_obs,
                                                device=self.device)
        
        while total_number_of_finished_runs < runs_number:
            # always reset the agent every step
            # observation = self.envs.render()
            pred_act = self.external_policy.act(eval_obs)
            (
                eval_obs,
                eval_rew,
                eval_terminations,
                eval_truncations,
                eval_infos,
            ) = self.envs.step(pred_act)

            # lets save only 10 first runs recordings, otherwise the video is too long
            # might want to save successes/failures as well
            if total_number_of_finished_runs < 10:
                recording.append(eval_obs['rgb'])
                
            cur_reward = eval_infos['episode']['reward'].item()
            total_reward += cur_reward
            runs_reward_history[cur_run_id][cur_step_since_reset] = cur_reward
            total_steps += 1
            cur_step_since_reset += 1
            
            if "final_info" in eval_infos:                
                for k, v in eval_infos["final_info"]["episode"].items():
                    eval_metrics[k].append(v)
                num_of_new_finished_runs = np.logical_or(
                    np.array(eval_terminations.cpu()), np.array(eval_truncations.cpu())
                ).sum()
                total_number_of_finished_runs += 1
                total_number_of_successes += eval_terminations.sum()
                postfix_str = f"Finished runs: {total_number_of_finished_runs} Cur Reward: {cur_reward} Success rate: {total_number_of_successes / total_number_of_finished_runs}"
                progress_bar.update(num_of_new_finished_runs.item())
                progress_bar.set_postfix_str(postfix_str)
                cur_run_id += 1
                cur_step_since_reset = 0
                
        print("Evaluation finished.")
        print(f"Avg reward per step {total_reward/total_steps}")
        print("Success rate: ", (total_number_of_successes / total_number_of_finished_runs).item())
        current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        recording = recording[:30*30] # lets save only 30 seconds of video
        # ================== Recording saving ==================
        self.save_recording(recording=recording, experiment_name=experiment_name, current_datetime_str=current_datetime_str)
        # ======================================================
        self.aggregate_reward_history(reward_history=runs_reward_history, experiment_name=experiment_name, current_datetime_str=current_datetime_str)
        # ================== Distribution plot ==================
        self.draw_distribution_plot(reward_history=runs_reward_history, experiment_name=experiment_name, current_datetime_str=current_datetime_str)
        # close the envs, or error will be shown during their deletion
        self.envs.close()
        
        
        # Save the recording as a video
    def aggregate_reward_history(self, reward_history: np.ndarray, experiment_name, 
                                 current_datetime_str) -> np.ndarray:
        df_rewards = pd.DataFrame(reward_history)
        df_rewards.to_csv(f"octo_analysis/{experiment_name}_{current_datetime_str}.csv")
        print(f"Saved reward history at octo_analysis/{experiment_name}_{current_datetime_str}.csv")
        print(df_rewards.tail(2))
    
    
    def save_recording(self, recording:list, experiment_name:str, current_datetime_str:str):
        
        
        env_name = self.env_config.env_id
        video_path = (
            f"octo_analysis/{experiment_name}_{current_datetime_str}.mp4"
        )
        # Ensure the directory exists
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        numpy_images = [tensor[0].cpu().numpy().astype(np.uint8) for tensor in recording]

        # Save the video using imageio
        with imageio.get_writer(video_path, fps=30) as writer:
            for image in numpy_images:
                writer.append_data(image)

        print(f"Saved evaluation video at {video_path}")
    
    def draw_distribution_plot(self, reward_history: pd.DataFrame, 
                               experiment_name, current_datetime_str):
        df = pd.DataFrame(reward_history)
        horizon_len = len(df.columns)
        df.reset_index(drop=False, inplace=True)
        df.rename(columns={"index": "step"}, inplace=True)
        
        fig = go.Figure()
        for i in range(horizon_len):
            trace = go.Histogram(x=df[i], nbinsx=50, visible=i==0)
            fig.add_trace(trace)

        steps = []
        for i in range(horizon_len):
            visibility = np.zeros(horizon_len, dtype=bool)
            visibility[i] = True

            step = dict(
                method="update",
                args=[{"visible": visibility}],  # type: ignore
                label=f"reward on step {str(i)}",
            )
            steps.append(step)
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "Step "},
                pad={"t": 100},
                steps=steps,
            )
        ]
        fig.update_layout(sliders=sliders,
                                title=f"Reward over time")
        
        fig.write_html(f"octo_analysis/{experiment_name}_{current_datetime_str}.html")
        print(f"Saved reward distr at octo_analysis/{experiment_name}_{current_datetime_str}.html")
        fig.show()
        
# CUDA_VISIBLE_DEVICES=0 python3 /home/piscenco/thesis_octo/thesis-code/octoplus/src/eval_octo.py
if __name__ == "__main__":
    experiment_name = "octo_26_11_evaluation_100runs"
    octo_inference = ExternalPolicyInference()
    print("Evaluating the external policy.")
    octo_inference.initialize_envs(env_config=SAMPLE_ENV_CONFIG)
    octo_inference.evaluate(runs_number=100, experiment_name=experiment_name)
    
# Avg reward per step 0.7350707302415839
# Success rate:  0.009999999776482582
