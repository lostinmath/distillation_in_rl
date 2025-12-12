"""
Metrics of particular interest for evaluation are:
- Success rate: the percentage of successful episodes (e.g., reaching the goal)
- Avg return per episode: the average return per episode

"""

import argparse
import logging
import math
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime

import cv2
import imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm

from octoplus.src.config_dataclasses.experiments import ExperimentConfig
from octoplus.src.env.mani_skill_env import get_envs
from octoplus.src.rl_models.ppo_rgb import PPORgb
from octoplus.src.rl_models.shared_model_api import load_model
from omegaconf import OmegaConf

# Suppress future warnings
# I know about this warning, need to think about the best solution
warnings.simplefilter(action="ignore", category=FutureWarning)
"""
FutureWarning: You are using `torch.load` with `weights_only=False`
(the current default value), which uses the default pickle module implicitly...
"""
############################ Defaults for 1 of my local models #########################
# /mod_ppo_octo_only_kl_pen
# 
ROOT_PATH = "logs/095_experiment_mae_octo_epsilon_0_5/PushCube-v1/octo_epsilon/2025-05-23-21-13-16/"
#"logs/095_experiment_mae_octo_epsilon_0_5/PushCube-v1/octo_epsilon/2025-04-02-20-52-09/"
#"logs/098_experiment_mae_octo_epsilon_decr_trust_3/PushCube-v1/octo_epsilon_decreasing/2025-03-30-22-20-31/"

#"logs/100_experiment_reward_based/PushCube-v1/octo_reward_based/2025-03-31-11-36-59/"
#"logs/096_experiment_octo_eps_0_5_mae_value/PushCube-v1/octo_epsilon/2025-03-27-12-54-45/"
# "logs/098_experiment_mae_octo_epsilon_decr_trust_3/PushCube-v1/octo_epsilon_decreasing/2025-03-29-06-50-40/"
# "logs/097_experiment_octo_eps_0_5_mae_value_act/PushCube-v1/octo_epsilon/2025-03-27-22-56-54/"
#"logs/099_experiment_act_on_teach_octo_only_mae_val_act/PushCube-v1/octo_only/2025-03-29-06-57-49/"
# 
#"logs/097_experiment_octo_eps_0_5_mae_value_act/PushCube-v1/octo_epsilon/2025-03-27-22-56-54/"
#"/home/piscenco/logs/098_experiment_mae_octo_epsilon_decr_trust_3/PushCube-v1/octo_epsilon_decreasing/2025-03-28-12-34-09/"

#"logs/094_experiment_mae_value_act_octo_only/PushCube-v1/octo_only/2025-03-27-01-16-37/"
#"logs/097_baseline_ppo_kl0_2_steps_2M/PushCube-v1/internal_only/2025-03-26-22-27-14/"
#"logs/debug/PushCube-v1/octo_epsilon/2025-03-24-23-40-28/" very successfull run!!
SAMPLE_MODEL_PATH = (
     ROOT_PATH +"095_experiment_mae_octo_epsilon_0_5/model_ckpt"
)
SAMPLE_PATH_TO_CONFIG = (
    ROOT_PATH + ".hydra/config.yaml"
)
######################### Logging ##################################################
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
######################## Utils for video ####################################

def get_ckpt_steps(path_to_ckpt:str):
    files = os.listdir(path_to_ckpt)
    files = sorted([int(file.replace(".pt", "").split("__")[1]) for file in files])
    return files

def save_video_from_tensors(tensor_list, output_path, fps=30):
    """
    Saves a video from a list of PyTorch tensors representing RGB images.

    Args:
        tensor_list (list of torch.Tensor): List of tensors representing RGB images.
        output_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """
    # Convert tensors to NumPy arrays
    numpy_images = [tensor.cpu().numpy().astype(np.uint8) for tensor in tensor_list]

    # Save the video using imageio
    with imageio.get_writer(output_path, fps=fps) as writer:
        for image in numpy_images:
            writer.append_data(image)


############################ Evaluation ################################################
# >> output_file.txt # appends to file
# time CUDA_VISIBLE_DEVICES=1 python3 /home/piscenco/thesis_octo/thesis-code/octoplus/src/evaluate_main2.py >> eval_result.txt
def evaluate_model(
    model_path: str,
    path_to_config: str,
    num_parallel: int,
    total_num_runs: int,
    print_config:bool=False
)-> dict:
    # Load the configuration file using OmegaConf
    cfg_omega = OmegaConf.load(path_to_config)

    # Resolve interpolations and defaults
    OmegaConf.resolve(cfg_omega)

    # Convert OmegaConf to a dictionary
    cfg_dict = OmegaConf.to_container(cfg_omega, resolve=True)
    
    if "defaults" in cfg_dict.keys():
        del cfg_dict["defaults"]  # whis is only for hydra
    cfg = ExperimentConfig(**cfg_dict)
    if print_config:
        print(f"# seed: {cfg.model_config.seed} target_kl: {cfg.model_config.target_kl}")
        print(f"# Config: {cfg}")
    cfg.model_config.use_external_policy = False
    cfg.scheduler_config.scheduling_strategy = "internal_only"
    

    # ovewrite some values in the config with ones from the command line
    cfg.env_config.num_envs = num_parallel
    cfg.env_config.reward_mode = "dense" # "normalized_dense"
    cfg.env_config.seed = 11  # use another seed for evaluation
    # Create the evaluation environments
    eval_envs = get_envs(
        # model_config=cfg.model_config,
        env_config=cfg.env_config,
        #hydra_log_dir="../../outputs_evaluate",
    )
    

    model = load_model(
        model_path=model_path, cfg=cfg, hydra_log_dir="evaluate_log/.", envs=eval_envs, mode="eval"
    )

    model.eval()
    # print(
    #     f"Model {type(model)} loaded and ready for evaluation in {cfg.env_config.env_id}"
    # )

    # now run the evaluation
    eval_metrics = defaultdict(list)
    metrics_history = []
    # Calculate the number of environments per parallel worker
    # Metrics:
    rewards_since_reset = [[] for _ in range(num_parallel)]
    all_avg_rewards = []
    all_avg_returns = []
    # all_steps_since_reset = []
    all_episode_lengths = []
    

    total_number_of_finished_runs = 0
    total_number_of_successes = 0
    all_reward = 0
    val_steps_count = 0

    eval_obs, _ = eval_envs.reset() # reset to a certain seed
    # unfortunately maniskill did not implement seeding for some reason? 
    # # eval_envs.reset(cfg.env_config.seed)
    # From 310 line of BaseEnv:
    #    # Use a fixed (main) seed to enhance determinism
    #    self._main_seed = None
    # recording = []
    # recording.append(eval_obs["rgb"])

    progress_bar = tqdm(range(1, total_num_runs), desc="Evaluation")

    while total_number_of_finished_runs < total_num_runs:
        with torch.no_grad():
            pred_act = model(eval_obs)
            # print(pred_act)
            (
                eval_obs,
                _,
                eval_terminations,
                eval_truncations,
                eval_infos,
            ) = eval_envs.step(pred_act)

            # recording.append(eval_obs["rgb"])
            all_reward += eval_infos["episode"]["reward"].sum().item()
            for i in range(num_parallel):
                rewards_since_reset[i].append(eval_infos["episode"]["reward"][i].cpu().item())
            val_steps_count += 1

            if "final_info" in eval_infos:
                # mask = eval_infos["_final_info"]
                finished_runs = torch.logical_or(
                    eval_terminations, eval_truncations
                ).cpu()
                
                
                for k, v in eval_infos["final_info"]["episode"].items():
                    eval_metrics[k].extend(v[finished_runs].cpu().tolist())
                    
                
                num_of_new_finished_runs = finished_runs.sum().item()
                # save rewards and returns and reset them if the run has finished
                all_avg_returns.extend(eval_infos["episode"]["return"][finished_runs].cpu().tolist())
                
                all_episode_lengths.extend(eval_infos["episode"]["episode_len"][finished_runs].cpu().tolist())
                
                for i, finished in enumerate(finished_runs):
                    if finished:
                        all_avg_rewards.append(np.mean(rewards_since_reset[i]))
                        rewards_since_reset[i] = []
                        
                        # avg reward is avg over all runs 1/ T * sum_{t=1}^{T} r_t, T - run length
                
                total_number_of_finished_runs += num_of_new_finished_runs
                total_number_of_successes += eval_terminations.sum().item()
                postfix_str = f"Success rate: {total_number_of_successes / total_number_of_finished_runs}. Finished runs: {total_number_of_finished_runs}."
                progress_bar.set_postfix_str(postfix_str)
                progress_bar.update(num_of_new_finished_runs)

    # Save the recording as a video
    # current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # experiment_name = cfg.model_config.exp_name
    # env_name = cfg.env_config.env_id
    
    # Too many videos, I evaluate ckpts each 1600 steps
    # video_path = (
    #     f"evaluation_videos/{env_name}_{experiment_name}_{current_datetime_str}.mp4"
    # )
    # Ensure the directory exists
    # os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # numpy_images = [tensor[0].cpu().numpy().astype(np.uint8) for tensor in recording]

    # # Save the video using imageio
    # with imageio.get_writer(video_path, fps=30) as writer:
    #     for image in numpy_images:
    #         writer.append_data(image)

    # logger.info(f"Saved evaluation video at {video_path}")

    metrics_history.append(eval_metrics)
    # for k, v in eval_metrics.items():
    #     mean = torch.stack(v).float().mean()
    #     print(f"eval/{k}", mean.item())
    #     print(f"eval_{k}_mean={mean.item()}")

    # print(
    #     "Success rate: ",
    #     (total_number_of_successes / total_number_of_finished_runs),
    # )
    # print("Total number of finished runs: ", total_number_of_finished_runs)
    # print("Total number of successes: ", total_number_of_successes)
    # print(f"Avg reward per step {all_reward/args.num_parallel/val_steps_count}")
    all_avg_rewards = np.array(all_avg_rewards)
    all_avg_returns = np.array(all_avg_returns)
    all_episode_lengths = np.array(all_episode_lengths)
    
    reported_metrics = {
        "success_rate": (total_number_of_successes / total_number_of_finished_runs),
        "total_number_of_finished_runs": total_number_of_finished_runs,
        "total_number_of_successes": total_number_of_successes,
        
        "avg_reward_per_episode": all_avg_rewards.mean(), 
        "avg_return_per_episode": all_avg_returns.mean(),         
        "avg_episode_length": all_episode_lengths.mean(), 
        
        "max_reward_per_episode": all_avg_rewards.max(),
        "min_reward_per_episode": all_avg_rewards.min(),
        "std_reward_per_episode": all_avg_rewards.std(),
        "max_return_per_episode": all_avg_returns.max(),
        "min_return_per_episode": all_avg_returns.min(),
        "std_return_per_episode": all_avg_returns.std(),
        "max_episode_length": all_episode_lengths.max(),
        "min_episode_length": all_episode_lengths.min(),
        "std_episode_length": all_episode_lengths.std(),
         
    }
    # We also need min, max, std!
    #print("reported_metrics: ", reported_metrics)
    return reported_metrics


# CUDA_VISIBLE_DEVICES=1 time python3 /home/piscenco/thesis_octo/thesis-code/octoplus/src/evaluate_main2.py >> eval_result_baseline.txt
if __name__ == "__main__":

    if "--help" in sys.argv:
        print(
            """
        Usage: python evaluate_main.py [OPTIONS]

        Options:
        --model_path      Path to the model checkpoint (default: /home/piscenco/thesis2/outputs/2024-09-27/16-41-39/PickCube/model_ckpt/ckpt_491.pt)
        --num_parallel    Number of parallel environments to run (default: 16)
        --total_num_env   Total number of environments to run (default: 32)
        --path_to_config  Path to the configuration file (default: /home/piscenco/thesis2/outputs/2024-09-27/16-41-39/.hydra/config.yaml)
        """
        )
    else:
        # parse arguments from command line
        # (see constants.py for the list of available environments)
        # --model_path: Path to the model checkpoint
        parser = argparse.ArgumentParser(
            description="Evaluate a trained model on a specified environment."
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=SAMPLE_MODEL_PATH,
            help="Path to the model checkpoint",
        )
        parser.add_argument(
            "--num_parallel",
            type=int,
            default=8,
            help="Number of parallel environmens to run",
        )
        parser.add_argument(
            "--total_num_env", type=int, default=500, help="Number of env to run"
        )
        parser.add_argument(
            "--path_to_config",
            type=str,
            default=SAMPLE_PATH_TO_CONFIG,
            help="Path to the configuration file",
        )
        args = parser.parse_args()

        if args.num_parallel < 1:
            raise ValueError("num_parallel must be greater than 0")
        model_path = args.model_path if args.model_path else SAMPLE_MODEL_PATH
        print(f"# model_path: {model_path}")
        
        all_steps = get_ckpt_steps(model_path)
        # all_steps = [all_steps[0], all_steps[15], all_steps[-1]]
        successful_steps = []
        logger.info(f"The evaluated_steps {all_steps}")
        # [1600, 17600, 33600, 49600, 65600, 81600, 97600, 113600, 129600, 145600, 161600, 177600, 193600]
        sc_metrics = []
        # runs evaluation of the model
        for j, ckpt_step in enumerate(all_steps):
            # print("*"*50)
            logger.info(f"Evaluating model at step {ckpt_step}")
            # print("*"*50)
            model_path = os.path.join(SAMPLE_MODEL_PATH, f"ckpt__{ckpt_step}.pt")
            # try:
            eval_metrics = evaluate_model(
                model_path=model_path,
                path_to_config=args.path_to_config,
                num_parallel=args.num_parallel,
                total_num_runs=args.total_num_env,
                print_config=j==0,
            )
            sc_metrics.append(eval_metrics)
            successful_steps.append(ckpt_step)
            # except Exception as error:
            #     print(f"An error occurred during evaluation of step {ckpt_step}:", error)    
        print("# =================================================================== #")
        print(f"all_steps[] = {all_steps}")        
        print(f"sc_metrics[] = {sc_metrics}")
        print("# =================================================================== #\n\n\n")
