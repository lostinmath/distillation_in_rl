import torch
from torch import nn
from collections import defaultdict
from typing import Optional
import argparse
from octoplus.src.env.mani_skill_env import get_envs
from octoplus.rldp_train import Actor
from octoplus.src.config_dataclasses.env_configs import ManiSkillConfig
import numpy as np
from tqdm import tqdm
import os


DEFAULT_CKPT_PATH = "/home/piscenco/runs/baseline_sac_seed_13/final_ckpt.pt"
ROOT_CKPT_PATH = "/mnt/dataset_drive/piscenco/baseline_sac_seed_26_2025-03-29_22-18-29/"

num_parallel=32
total_num_runs=1000


DEFAULT_ENV_CONFIG = ManiSkillConfig(
    obs_mode="rgb",
    control_mode="pd_ee_delta_pose",
    render_mode="rgb_array",
    sim_backend="gpu",
    reward_mode="normalized_dense",
    env_id="PushCube-v1",
    num_envs=32,
    num_steps=50,
    include_state=False,
    partial_reset=True,
    image_height=256,
    image_width=256,
    use_render_camera_as_input=True
)
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_ckpt_steps(path_to_ckpt:str):
    files = os.listdir(path_to_ckpt)
    files = sorted([int(file.replace(".pt", "").split("_")[1]) for file in files if "final" not in file and "pt" in file])
    return files

def evaluate_sac_agent(
    actor: nn.Module,
) -> dict:
    """
    Evaluate the SAC agent on the given evaluation environments.

    Args:
        actor (nn.Module): The SAC actor model.
        eval_envs (ManiSkillVectorEnv): The evaluation environments.
        num_eval_steps (int): Number of steps to run in each evaluation environment.
        device (torch.device): The device to run the evaluation on.
        logger (Optional[Logger]): Logger for recording evaluation metrics.
        global_step (int): Current global training step for logging.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    actor.eval()
    eval_obs, _ = eval_envs.reset(seed=11)
    eval_obs = {k: v.to(device) for k, v in eval_obs.items()}

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

    eval_obs, _ = eval_envs.reset(seed=11) # reset to a certain seed
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
            pred_act = actor.get_action(eval_obs)[0]
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

    metrics_history.append(eval_metrics)
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



if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint", type=str, required=True, help="Path to the SAC model checkpoint.", default=DEFAULT_CKPT_PATH)
    # parser.add_argument("--num_eval_steps", type=int, default=10, help="Number of evaluation steps.")
    # args = parser.parse_args()
    # checkpoint = DEFAULT_CKPT_PATH
    # num_eval_steps = 10
    
    # Initialize environment, actor, and logger
    eval_envs = get_envs(DEFAULT_ENV_CONFIG)  # Example environment
    eval_envs.reset(seed=11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_step = 0

    # Load the SAC actor model
    obs = eval_envs.reset()[0]
    actor = Actor(eval_envs, sample_obs=obs).to(device)
    
    all_steps = get_ckpt_steps(ROOT_CKPT_PATH)
    successful_steps = []
    logger.info(f"The evaluated_steps {all_steps}")
    sc_metrics = []

    for j, ckpt_step in enumerate(all_steps):
        logger.info(f"Evaluating model at step {ckpt_step}")
        checkpoint = os.path.join(ROOT_CKPT_PATH, f"ckpt_{ckpt_step}.pt")
        ckpt = torch.load(checkpoint, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        eval_metrics = evaluate_sac_agent(
            actor=actor,
            )
        sc_metrics.append(eval_metrics)
        successful_steps.append(ckpt_step)
        print(f"Evaluation metrics at step {ckpt_step}: {eval_metrics}")

    print("# =================================================================== #")
    print(f"all_steps[] = {all_steps}")        
    print(f"sc_metrics[] = {sc_metrics}")
    print("# =================================================================== #\n\n\n")