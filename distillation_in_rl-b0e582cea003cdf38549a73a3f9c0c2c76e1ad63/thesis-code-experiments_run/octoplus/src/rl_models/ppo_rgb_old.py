import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Any, Optional
import gymnasium as gym
# ManiSkill specific imports
import mani_skill.envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.distributions.normal import Normal
import imageio
import pandas as pd

from agents.client import RemoteAgent
from agents.policies import Obs

from octoplus.src.config_dataclasses.env_configs import ManiSkillConfig
from octoplus.src.config_dataclasses.experiments import ExperimentConfig
from octoplus.src.config_dataclasses.model_configs import PPORGBConfig
from octoplus.src.utils.custom_logging import Logger
from octoplus.src.rl_models.nature_cnn import Agent
from octoplus.src.rl_models.shared_model_api import AbstractedRrAlgo
from octoplus.src.utils.utils_for_octo import reshape_obs_for_octo
from octoplus.src.config_dataclasses.policy_scheduling import PolicySchedulingConfig

OCTO_PORT = 10_000

# ================== # Logging # ==================================================== #
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# =================================================================================== #

INSTRUCTIONS = {
        "LiftPegUpright-v1": "lift the peg upright",
        "PegInsertionSide-v1": "insert the peg from the side",
        "PickCube-v1": "pick up the cube",
        "PlugCharger-v1": "plug the charger in",
        "PullCube-v1": "pull the cube towards the robot base",
        "PullCubeTool-v1": "pull the cube by using the red tool",
        "PushCube-v1": "push the cube away from the robot base",
        "PushT-v1": "align the T shape",
        "RollBall-v1": "push the ball",
        "StackCube-v1": "stack the red cube on the green cube",
        "PokeCube-v1": "push the cube by using the blue tool",
    }


def to_device(nested_dict, device):
    """
    Converts nested dict with tensors to device
    """
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            to_device(v, device)
        else:
            nested_dict[k] = v.to(device=device)
                            
                            
class DictArray(object):
    """
    A class to manage a dictionary of arrays, where each array can have a different shape.
    This class is useful for handling complex data structures in rl,
    as observations or other data can be represented as dictionaries with nested
    structures.

    Here is used to store observations.

    Attributes:from torch.distributions.normal import Normal
        buffer_shape (tuple): The shape of the buffer.
        data (dict): A dictionary containing the data arrays.
    Methods:
        keys():
            Returns the keys of the data dictionary.
        __getitem__(index):
            Retrieves an item or a sub-dictionary from the data dictionary.
        __setitem__(index, value):
            Sets an item or a sub-dictionary in the data dictionary.
        shape():
            Returns the shape of the buffer.
        reshape(shape):
            Reshapes the data dictionary to a new shape.
    """

    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v)
                else:
                    self.data[k] = torch.zeros(buffer_shape + v.shape).to(device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[: len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


class OctoPolicy:
    """
    Wrapper around Remote Agent, where Octo resides
    """
    def __init__(self, env_id:str, 
                 instruction:str,
                 batch_example:dict,
                 device:str="cuda"):
        # self.octo: RemoteAgent = RemoteAgent.by_config()
        self.octo: RemoteAgent = RemoteAgent("localhost", OCTO_PORT, "octodist")
        self.octo.setup(env_id=env_id, batch_example=batch_example)
        self.octo.reset(obs=Obs(rgb_side=None), instruction=instruction)
    
        self.env_id = env_id
        self.device = device
        
    def act(self, obs, pos:None|int=None):
        # needs resizing to 256x256
        # https://github.com/octo-models/octo/blob/main/examples/01_inference_pretrained.ipynb
        # is it a dict?
        # if type(obs) == dict:
        #     obs = obs["rgb"]
        result = self.octo.act_batch(obs)
        return torch.tensor(result, device=self.device)


class RandomPolicy:
    def __init__(self, device="cuda"):
        self.device = device
    
    def act(self, obs):
        action_mean = torch.zeros(7)
        action_std = torch.ones(7)
        probs = Normal(action_mean, action_std)
        return probs.sample().to(self.device)
    
    
class PolicyScheduler:
    """
    
    
    """
    def __init__(self, cfg:ExperimentConfig, envs, next_obs, hydra_log_dir:str):
        self.cfg: ExperimentConfig = cfg
        self.scheduler_config:PolicyScheduler = cfg.scheduler_config
        self.scheduling_strategy = self.scheduler_config.scheduling_strategy
        self.device = torch.device(self.cfg.device)
        self.hydra_log_dir = hydra_log_dir
        self.max_iter = self.cfg.model_config.num_iterations
        # initialize external policy if scheduling strategy is more complicated than "internal only"
        # if change between policies based on reward
        self.last_used_policy = ["octo"] * self.cfg.model_config.num_envs
        self.steps_taken_on_last_policy = [0] * self.cfg.model_config.num_envs
        self.prev_prev_reward = [-1] * self.cfg.model_config.num_envs
        
        # we always create internal policy
        self.internal_policy = Agent(envs, sample_obs=next_obs,                                     
                                     model_config=self.cfg.model_config)
        self.internal_policy.to(self.cfg.model_config.model_device)
        # loading must be outside!!!
        # if self.cfg.model_config.checkpoint:
        #     self.load(self.cfg.model_config.checkpoint)
        # possible external policies
        if "octo" in self.cfg.scheduler_config.scheduling_strategy:
            self.octo: OctoPolicy = OctoPolicy( env_id=cfg.env_config.env_id, 
                                                instruction=cfg.scheduler_config.text_instruction,
                                                batch_example=next_obs,
                                                device=self.device)
        if "random" in self.cfg.scheduler_config.scheduling_strategy:
            self.random: RandomPolicy = RandomPolicy()
            
    def load(self, checkpoint_path: str):
        self.internal_policy.load_state_dict(torch.load(checkpoint_path))
        Logger.terminal_only_print(f"Loaded model from {self.cfg.model_config.checkpoint}")
        
    def save(self, checkpoint_path: str):
        torch.save(self.internal_policy.state_dict(), checkpoint_path)
            
    def reset(self):
        self.acted_on_external_policy = 0
        self.acted_on_internal_policy = 0
        self.external_policy_is_choosen = False
        self.last_used_policy = ["octo"] * self.cfg.model_config.num_envs
        self.steps_taken_on_last_policy = [0] * self.cfg.model_config.num_envs
        self.prev_prev_reward = [-1] * self.cfg.model_config.num_envs
        
        
    def eval(self):
        self.internal_policy.eval()
    def train(self):
        self.internal_policy.train()    
    
    def to(self, device):
        self.device = device
        self.internal_policy.to(device)
        
    # ===================== epsilon decay strategies ==================
    def _linear_than_const(self, probability_of_external_action:float, 
                           iteration:int, num_iterations:int) -> float:
        """
        
        """
        if iteration > 0.5 * num_iterations:
            return 0.05
        else:
            return (
                probability_of_external_action
                * ((2 * iteration) / num_iterations)
            )
    def _linear(self, probability_of_external_action:float, 
                iteration:int, num_iterations:int) -> float:
        return probability_of_external_action * (1 - (iteration / num_iterations))
    def _constant(self, probability_of_external_action:float, 
                  iteration:int, num_iterations:int) -> float:
        return probability_of_external_action
    
    def get_action_and_value(self, obs, policy_type:list[str]):
        """
        Predict action, logprob and value for given policy type.
        Args:
            obs: DictArray
            policy_type: list[str]
        Returns:
            action: torch.Tensor
            logprob: torch.Tensor
            value: torch.Tensor
        """
        for key in obs.keys():
            obs[key].to(self.device)
        # case all element are the same batched prediction
        if all(policy == "internal" for policy in policy_type):
            action, logprob, _, value = self.internal_policy.get_action_and_value(obs)
            # stud_act = self.internal_policy.get_action(obs, deterministic=True).to(self.device)
            return action, logprob, value, action, np.zeros(len(policy_type), dtype=bool)
        
        if all(policy == "octo" for policy in policy_type):
            action = self.octo.act(obs).to(self.device)
            _, logprob, _, value = self.internal_policy.get_action_and_value(obs, action)
            stud_act, _, _, _ = self.internal_policy.get_action_and_value(obs)
            return action, logprob, value, stud_act, np.ones(len(policy_type), dtype=bool)
        predicted_actions = torch.zeros((len(policy_type), 7)).to(self.device)
        predicted_logprobs = torch.zeros((len(policy_type), 1)).to(self.device)
        predicted_values = torch.zeros((len(policy_type), 1)).to(self.device)
        # if uses internal policy somewhere faster to predict in a batched way for all, cause network is small
        predicted_actions, predicted_logprobs, _, predicted_values = self.internal_policy.get_action_and_value(obs)
        # for octo overwrite internal policy predictions
        use_octo = [ True if policy == "octo" else False for policy in policy_type]
        
        predicted_actions[use_octo] = self.octo.act({k: v[use_octo] for k, v in obs.items()})
        stud_act, _, _, _ = self.internal_policy.get_action_and_value(obs)
        return predicted_actions, predicted_logprobs, predicted_values, stud_act, np.array(use_octo, dtype=bool)
        
    def calculate_epsilon(self, global_step:int) -> float:
        return max(- 1/self.scheduler_config.decrease_until_global_step * global_step + 1, 0)
        
    def choose_policy_type(self, iteration: int, global_step:int,
                           steps_since_reset: torch.Tensor,
                           prev_reward: torch.Tensor)->list[str]:
        """
        This can be fairly complicated strategy to select policy to take next step on.
        """
        match self.scheduling_strategy:
            case "internal_only":
                return ["internal"] * self.cfg.model_config.num_envs
            case "octo_only":
                return ["octo"] * self.cfg.model_config.num_envs
            case "internal_octo_interchangeably":
                if iteration % 2 == 0:
                    return ["internal"] * self.cfg.model_config.num_envs
                else:
                    return ["octo"] * self.cfg.model_config.num_envs
                
            case "octo_than_internal":
                if iteration < self.scheduler_config.iteration_to_switch:
                    list_of_policies = ["octo" if el < self.scheduler_config.step_to_switch 
                                        else "internal" for el in steps_since_reset] 
                    return list_of_policies
                else:
                    return ["internal"] * self.cfg.model_config.num_envs
                
            case "octo_epsilon_decreasing":
                list_of_policies = []
                if iteration >= self.scheduler_config.decrease_until_global_step:
                    list_of_policies = ["internal"] * self.cfg.model_config.num_envs
                else:
                    # after that act like on octo_epsilon
                    cur_epsilon = self.calculate_epsilon(global_step=global_step) # epsilon is prob of octo
                    # print(f"Current epsilon: {cur_epsilon}") # calculate decreasing epsilon and act randomly
                    list_of_policies = []
                    for i in range(self.cfg.model_config.num_envs):
                        if prev_reward[i] == -1:
                            list_of_policies.append("octo") # if env is reset use octo
                            self.steps_taken_on_last_policy[i] = 0 # reset steps, cause env was eret and new epoch
                        # if trust period has expired
                        elif self.steps_taken_on_last_policy[i] >= self.scheduler_config.policy_trust_length:
                                
                                sampled_epsilon = random.random()
                                # with probability epsilon we switch to octo from internal
                                if (self.last_used_policy[i] == "internal" and sampled_epsilon < cur_epsilon):
                                    list_of_policies.append("octo")
                                    self.last_used_policy[i] = "octo"
                                    self.steps_taken_on_last_policy[i] = 0
                                # vice versa with prob 1 - epsilon
                                elif (self.last_used_policy[i] == "octo" and sampled_epsilon < 1 - cur_epsilon):
                                    list_of_policies.append("internal") 
                                    self.last_used_policy[i] = "internal"
                                    self.steps_taken_on_last_policy[i] = 0                                          
                                # keep the policy
                                else:
                                    list_of_policies.append(self.last_used_policy[i])      
                        else:
                            list_of_policies.append(self.last_used_policy[i])
                        
                        self.steps_taken_on_last_policy[i] += 1
                       
            case "octo_reward_based":
                list_of_policies = []                
                for i in range(self.cfg.model_config.num_envs):
                    if prev_reward[i] == -1:
                        list_of_policies.append("internal") # if env is reset use Internal strategy!! Lets give the student posibility!
                        self.last_used_policy[i] = "internal" # 
                        self.steps_taken_on_last_policy[i] = 0
                    # change policy if more than policy_trust_length steps have passed and
                    # reward has decreased
                    elif self.steps_taken_on_last_policy[i] >= self.scheduler_config.policy_trust_length and \
                        prev_reward[i] < self.prev_prev_reward[i]:
                        list_of_policies.append("internal" if self.last_used_policy[i] == "octo" else "octo")
                        self.last_used_policy[i] = "internal" if self.last_used_policy[i] == "octo" else "octo"
                        self.steps_taken_on_last_policy[i] = 0
                    # or if keep the previous one
                    else:
                        list_of_policies.append(self.last_used_policy[i])
                    self.prev_prev_reward[i] = prev_reward[i].item()
                    self.steps_taken_on_last_policy[i] += 1
                # print('Rewards:', self.prev_prev_reward)
                # print(list_of_policies)
            case "octo_epsilon":
                # this policy gives credit to policy for policy_trust_length than switched with probability epsilon
                list_of_policies = []
                for i in range(self.cfg.model_config.num_envs):
                    if prev_reward[i] == -1:
                        list_of_policies.append("octo") # if env is reset use octo
                    # if trust period has expired
                    elif self.steps_taken_on_last_policy[i] >= self.scheduler_config.policy_trust_length:
                            sampled_epsilon = random.random()
                            # with probability epsilon we switch to octo from internal
                            if (self.last_used_policy[i] == "internal" and sampled_epsilon < self.scheduler_config.epsilon):
                                list_of_policies.append("octo")
                                self.last_used_policy[i] = "octo"
                                self.steps_taken_on_last_policy[i] = 0
                            # vice versa with prob 1 - epsilon
                            elif (self.last_used_policy[i] == "octo" and sampled_epsilon < 1 -self.scheduler_config.epsilon):
                                list_of_policies.append("internal") 
                                self.last_used_policy[i] = "internal"
                                self.steps_taken_on_last_policy[i] = 0                                          
                            # keep the policy
                            else:
                                list_of_policies.append(self.last_used_policy[i])
                            
                    else:
                        list_of_policies.append(self.last_used_policy[i])
                    self.prev_prev_reward[i] = prev_reward[i].item()
                    self.steps_taken_on_last_policy[i] += 1
        # save to file used policies
        policy_history_path = f"{self.hydra_log_dir}/{self.cfg.model_config.run_name}/policy_history.csv"
        policy_df = pd.DataFrame([list_of_policies])
        if os.path.exists(policy_history_path):
            policy_df.to_csv(policy_history_path, mode='a', index=False, header=False)
        else:
            policy_df.to_csv(policy_history_path, index=False)
        return list_of_policies
    

class PPORgb(AbstractedRrAlgo):
    def __init__(self, envs, 
                 cfg: ExperimentConfig, 
                 hydra_log_dir: str = "run") -> None:
        self.cfg: ExperimentConfig = cfg
        self.args: PPORGBConfig = cfg.model_config
        self.envs = envs
        self.hydra_log_dir = hydra_log_dir
        self.device = self.args.model_device
        self.log_epsilon = self.cfg.scheduler_config.scheduling_strategy == "octo_epsilon_decreasing"

    def model_setup(self):
        # put logger here, because should use mlflow for training and should not for eval
        self.logger = Logger(
            run_name=self.args.run_name,
            log_file=f"{self.hydra_log_dir}/{self.args.run_name}/train_metrics.txt",
            use_mlflow=True,
        )
        for field in fields(self.cfg.model_config):
            self.logger.log_param(
                field.name, getattr(self.cfg.model_config, field.name)
            )
        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
        self.device = torch.device(self.cfg.device)

        self.max_episode_steps = gym_utils.find_max_episode_steps_value(self.envs._env)

        # set it up here because is needed for both train and eval
        next_obs, _ = self.envs.reset(seed=self.args.seed) # WE use the same sed for model and env initialization!!
        for key in next_obs:
            next_obs[key] = next_obs[key].to(device=self.device)
        self.scheduler = PolicyScheduler(cfg=self.cfg, envs=self.envs, next_obs=next_obs, hydra_log_dir=self.hydra_log_dir)
        

        
    def model_eval_setup(self):
        # put logger here, becauese should use mlflow for training and should not for eval
        # self.logger = Logger(
        #     run_name=self.args.run_name,
        #     log_file=f"{self.hydra_log_dir}/{self.args.run_name}/train_metrics.txt",
        #     use_mlflow=False,
        # )
        self.logger = None
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
        self.device = torch.device(self.cfg.device)

        self.max_episode_steps = gym_utils.find_max_episode_steps_value(self.envs._env)

        next_obs, _ = self.envs.reset(seed=self.args.seed)

        for key in next_obs:
            next_obs[key] = next_obs[key].to(device=self.device)
        # for evaluation scheduling_strategy must be internal_only
        assert self.cfg.scheduler_config.scheduling_strategy == "internal_only"        
        self.scheduler = PolicyScheduler(cfg=self.cfg, envs=self.envs, next_obs=next_obs, hydra_log_dir=self.hydra_log_dir)
        
    
    def start_training(self):
        # ALGO Logic: Storage setup
        
        """
        obs: Stores observations for each eself.eval()nvironment step.
        actions: Stores the actions taken by the agent.
        logprobs: Log probabilities of the actions, important for calculating loss in policy optimization.
        rewards: Stores rewards from the environment.
        dones: Flags that indicate whether an episode ended at a given step.
        values: Stores value function estimates for each step (used for advantage estimation).
        """
        # TRY NOT TO MODIFY: start the game        
        global_step = 0 # is increased only during training!
        # start_time = time.time()

        self.optimizer = optim.Adam(
            self.scheduler.internal_policy.parameters(), 
            lr=self.args.learning_rate, eps=1e-5
        )
        
        print("Training started.")
        print(f"Saving ckpt every {round(self.cfg.save_train_video_freq // self.args.batch_size)} iterations.")
        print(f"Saving video every {round(self.cfg.save_train_video_freq//self.args.batch_size)} iterations.")

        model_ckpt_dir = os.path.join(
            self.hydra_log_dir, self.args.run_name, "model_ckpt"
        )
        if not os.path.exists(model_ckpt_dir):
            os.makedirs(model_ckpt_dir)
        # excluded validation at all, instead saving checkpoints and videos during training and can eval on them
        progress_bar = tqdm(
            range(0, self.args.num_iterations),  desc="Training Loop"
        )
        train_iteration_number = 0 
        
        # ============== debug variables ================= #
        debug_returns = []
        debug_ratio = []
        debug_mb_advantages = []
        debug_approx_kl = []
        debug_v_loss = []
        debug_clipfracs = []
        # components of loss
        debug_pg_loss = []
        debug_entropy_loss = []
        debug_total_loss = []
        
        debug_octo_use_percentage = []
        
        # try:
        if True:
            self.envs.reset(seed=self.args.seed)
            
            # the actual training loop
            for iteration in progress_bar:        
                mode = "Training"
                self.train()
                # we update the clipping threshold
                # this should be done only from the  config, also makes runs not comparable
                # if global_step > self.cfg.scheduler_config.decrease_until_global_step:
                #     self.args.clip_coef = 0.2 # set to default one   
                
                obs = DictArray(
                    (self.args.num_steps, self.args.num_envs),
                    self.envs.single_observation_space,
                    device=self.device,
                )
                actions = torch.zeros(
                    (self.args.num_steps, self.args.num_envs)
                    + self.envs.single_action_space.shape
                ).to(self.device)
                logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
                    self.device
                )
                rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
                reward_history = torch.zeros((self.args.num_envs, self.args.num_steps))            
                
                dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
                values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
                next_done = torch.zeros(self.args.num_envs, device=self.device)
                student_actions = torch.zeros((self.args.num_steps, self.args.num_envs, self.envs.action_space.shape[1])).to(self.device) # as actions have 7 dim
                use_octo_all = np.zeros((self.args.num_steps, self.args.num_envs), dtype=bool)
                recording = []
                
                #TODO:env should be reset here
                next_obs, _ = self.envs.reset(seed=self.args.seed)
                steps_since_reset = torch.zeros(self.args.num_envs)
                
                recording.append(next_obs['rgb'][0])
                for key in next_obs:
                    next_obs[key] = next_obs[key].to(device=self.device)
                
                cur_successes = 0
                cur_finished_runs = 0
                last_step_reward = list()
                
                # for scheduling
                reward = torch.zeros(self.args.num_envs, device=self.device) - 1 # rewards of prev step for scheduler
                
                self.logger.terminal_only_print(
                    f"Epoch: {iteration}, global_step={global_step}"
                )
                final_values = torch.zeros(
                    (self.args.num_steps, self.args.num_envs), device=self.device
                )
                
                # Annealing the rate if instructed to do so.
                if self.args.anneal_lr:
                    frac = 1.0 - iteration / self.args.num_iterations
                    lrnow = frac * self.args.learning_rate
                    self.optimizer.param_groups[0]["lr"] = lrnow
                rollout_time = time.time()
                
                for step in range(0, self.args.num_steps):
                    if mode == "Training":
                        global_step += self.args.num_envs # increase global step only during training
                        if self.log_epsilon: # log epsilon if needed
                            self.logger.log_metrics(
                                f"{mode}/epsilon", self.scheduler.calculate_epsilon(global_step),global_step
                            )

                    obs[step] = next_obs
                    dones[step] = next_done

                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        used_policies:list = self.scheduler.choose_policy_type(
                        iteration=train_iteration_number,
                        global_step=global_step,
                        steps_since_reset=steps_since_reset,
                        prev_reward=reward)
                            
                        action, logprob, value, stud_act, use_octo = self.scheduler.get_action_and_value(obs=next_obs, policy_type=used_policies)
                            
                        """
                        Note:
                        The agent, given the current observation, uses the policy to sample
                        an action and computes its log probability,
                        entropy (for regularization), and the estimated value of the state.
                        """
                        values[step] = value.flatten()
                    actions[step] = action # these are selected by scheduler actions
                    logprobs[step] = logprob
                    student_actions[step] = stud_act
                    use_octo_all[step] = use_octo
    

                    # take the step in the environment
                    action = action if self.cfg.model_config.act_on_teach_actions else stud_act
                    
                    next_obs, reward, terminations, truncations, infos = self.envs.step(
                        action
                    )
                    recording.append(next_obs['rgb'][0])
                    steps_since_reset = infos['elapsed_steps'].cpu()

                    # convert to device, sim may run on cuda
                    for key in next_obs:
                        next_obs[key] = next_obs[key].to(device=self.device)

                    reward = reward.to(self.device)
                    terminations = terminations.to(self.device)
                    truncations = truncations.to(self.device)

                    to_device(nested_dict=infos, device=self.device)

                    next_done = torch.logical_or(terminations, truncations).to(
                        torch.float32
                    )
                    rewards[step] = reward.view(-1) * self.args.reward_scale
                    reward_history[:, step] = rewards[step] # add reward to history
                    # prev_prev_rewards = prev_rewards
                    # prev_rewards = rewards[step].sum().item()

                    if "final_info" in infos:
                        final_info = infos["final_info"]
                        done_mask = infos["_final_info"]
                        # zero out rewards if env will be reset
                        reward[done_mask] = -1
                        
                        if "episode_len" in final_info["episode"].keys():
                            self.logger.log_metrics(
                                f"{mode}/episode_len",
                                final_info["episode"]["episode_len"][done_mask].float().mean().item(),
                                global_step,
                            )                        
                        num_of_new_finished_runs = torch.logical_or(
                            terminations, truncations
                        ).sum()
                        cur_finished_runs += num_of_new_finished_runs
                        cur_successes += terminations.sum()
                        last_step_reward.extend([el.item() for el in final_info["episode"]["reward"][done_mask].float()])

                        # cur reward to realize if it learns anything even if if success does not change
                        cur_reward = "???"  # on first iteration set it to unknown
                        if "reward" in final_info["episode"].keys():
                            cur_reward = (
                                final_info["episode"]["reward"][done_mask]
                                .float()
                                .mean()
                                .item()
                            )
                        # logging reward
                        postfix_str = f"{mode} | Epoch: {iteration} Success rate: {(cur_successes / cur_finished_runs):.3f} Reward: {cur_reward:.3f}"
                        progress_bar.set_postfix_str(postfix_str)
                        log_to_file_ = f"{mode},Epoch,{iteration},Success rate,{cur_successes / cur_finished_runs},Reward,{cur_reward}"
                        self.logger.log_to_file(log_to_file_)

                        for k in infos["final_observation"]:
                            infos["final_observation"][k] = infos["final_observation"][k][
                                done_mask
                            ]
                        with torch.no_grad():
                            final_values[
                                step,
                                torch.arange(self.args.num_envs, device=self.device)[
                                    done_mask
                                ],
                            ] = self.scheduler.internal_policy.get_value(infos["final_observation"]).view(-1)
                rollout_time = time.time() - rollout_time
                # ========== Save the recording ===========
                # if (self.args.evaluate and iteration % self.args.eval_freq == 0 and 
                # self.args.save_eval_video and iteration % self.args.save_eval_video_freq == 0 )
                # or (save_train_video)
                if (mode == "Validation" and self.cfg.save_eval_videos) or (
                    mode == "Training" and self.cfg.save_train_videos and \
                    train_iteration_number % round(self.cfg.save_train_video_freq // self.args.batch_size) == 0
                ):
                    video_path = (f"{self.hydra_log_dir}/{self.args.run_name}/recordings/{mode}_{iteration}_{train_iteration_number}.mp4")
                    os.makedirs(os.path.dirname(video_path), exist_ok=True) # create if does not exist
                    numpy_images = [tensor.cpu().numpy().astype(np.uint8) for tensor in recording]
                    # Save the video using imageio
                    with imageio.get_writer(video_path, fps=30) as writer:
                        for image in numpy_images:
                            writer.append_data(image)
                # print(f"Saved video for mode {mode} at iteration {iteration} and train iteration {train_iteration_number}")
                        
                # ================ Save rewards ===========================
                # beware: ran may finish with success on step n and than later rewards are actually from the reset env
                rewards_df = pd.DataFrame(reward_history.cpu().numpy())
                rewards_df['iteration'] = iteration
                rewards_df['mode'] = mode
                rewards_file_path = f"{self.hydra_log_dir}/{self.args.run_name}/rewards.csv"
                if os.path.exists(rewards_file_path):
                    rewards_df.to_csv(rewards_file_path, mode='a', index=False, header=False)
                rewards_df.to_csv(rewards_file_path, index=False)
                del rewards_df
                # ================ Logging metrics ===========================
                # TODO: here should be train iteration number??
                self.logger.log_metrics(
                    
                    f"{mode}/success_rate", cur_successes / cur_finished_runs if cur_finished_runs>0 else 0, global_step
                )
                self.logger.log_metrics(
                    f"{mode}/reward_avg", reward_history.mean() if len(reward_history)> 0 else 0, global_step
                )
                self.logger.log_metrics(
                    f"{mode}/reward_std", reward_history.std() if len(reward_history)> 0 else 0, global_step
                )
                self.logger.log_metrics(
                    f"{mode}/last_step_reward_avg", np.array(last_step_reward).mean() if len(last_step_reward)> 0 else 0, global_step
                )
                self.logger.log_metrics(
                    f"{mode}/last_step_reward_std", np.array(last_step_reward).std() if len(last_step_reward)> 0 else 0, global_step
                )
                
                # self.cfg.ckpt_save_frequency is in steps but we need iterations so // self.args.batch_size
                if mode == "Training" and train_iteration_number % round(self.cfg.ckpt_save_frequency // self.args.batch_size) == 0:    
                    # TODO: cumulative reward. Its hard because the runs are not the same length and env can be reset after success
                    # update progress bar
                    # lets always save the model during validation
                    model_path = f"{self.hydra_log_dir}/{self.args.run_name}/model_ckpt/ckpt__{global_step}.pt"
                    self.scheduler.save(model_path)
                
                # bootstrap value according to termination and truncation
                with torch.no_grad():
                    next_value = self.scheduler.internal_policy.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1:
                            next_not_done = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            next_not_done = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        real_next_values = (
                            next_not_done * nextvalues + final_values[t]
                        )  # t instead of t+1
                        # next_not_done means nextvalues is computed from the correct next_obs
                        # if next_not_done is 1, final_values is always 0
                        # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                        if self.args.finite_horizon_gae:
                            """
                            See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                            1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                            lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                            lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                            lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                            We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                            """
                            if t == self.args.num_steps - 1:  # initialize
                                lam_coef_sum = 0.0
                                reward_term_sum = 0.0  # the sum of the second term
                                value_term_sum = 0.0  # the sum of the third term
                            lam_coef_sum = lam_coef_sum * next_not_done
                            reward_term_sum = reward_term_sum * next_not_done
                            value_term_sum = value_term_sum * next_not_done

                            lam_coef_sum = 1 + self.args.gae_lambda * lam_coef_sum
                            reward_term_sum = (
                                self.args.gae_lambda * self.args.gamma * reward_term_sum
                                + lam_coef_sum * rewards[t]
                            )
                            value_term_sum = (
                                self.args.gae_lambda * self.args.gamma * value_term_sum
                                + self.args.gamma * real_next_values
                            )

                            advantages[t] = (
                                reward_term_sum + value_term_sum
                            ) / lam_coef_sum - values[t]
                        else:
                            delta = (
                                rewards[t] + self.args.gamma * real_next_values - values[t]
                            )
                            advantages[t] = lastgaelam = (
                                delta
                                + self.args.gamma
                                * self.args.gae_lambda
                                * next_not_done
                                * lastgaelam
                            )  # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
                    returns = advantages + values

                # flatten the batch
                b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)   
                b_st_actions = student_actions.reshape((-1,) + self.envs.single_action_space.shape)         
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)
                use_octo_all = use_octo_all.reshape(-1)
                debug_returns.extend(b_returns.cpu().numpy())
                
                # Optimizing the policy and value network
                self.scheduler.internal_policy.train()
                b_inds = np.arange(self.args.batch_size)
                clipfracs = []
                update_time = time.time()
                for epoch in range(self.args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, self.args.batch_size, self.args.minibatch_size):
                        end = start + self.args.minibatch_size
                        mb_inds = b_inds[start:end]
                        
                        octo_inds = mb_inds[use_octo_all[mb_inds]]
                        int_inds = mb_inds[~use_octo_all[mb_inds]]
                        loss = torch.tensor(0.0).to(self.device)
                        # get actions and values with gradients
                        # than depending on action taken calculate the loss diffewerntly
                        # PPO_loss(student_action) 
                        # # or PPO_loss(teacher action, techers action value) + L2_loss(student_action, teacher_action)
                        calculated_student_loss = False
                        calculated_teacher_loss = False
                        # ==================student actions!!=========================
                        if len(int_inds) > 0:
                            _, newlogprob, entropy, newvalue = self.scheduler.internal_policy.get_action_and_value(
                                obs.reshape((-1,))[int_inds], b_actions[int_inds]
                            )
                            
                            logratio = newlogprob - logprobs.reshape(-1)[int_inds] # we take only student actions for kl
                            ratio = logratio.exp() # == 1 so should learn smth?
                            
                            with torch.no_grad():
                                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean() # 0s here
                                # This clipping coef could be changed
                                clipfracs += [
                                    ((ratio - 1.0).abs() > self.args.clip_coef)
                                    .float()
                                    .mean()
                                    .item()
                                ] # 0.0?
                                
                            debug_ratio.extend(ratio.detach().cpu().numpy())
                            debug_approx_kl.append(approx_kl.cpu().numpy())
                            debug_clipfracs.extend(clipfracs)
                            
                            # print(f"clipfracs {clipfracs}")
                            self.logger.log_metrics("losses/value_to_clip_mean_on_student_actions", (ratio - 1.0).abs().mean().item(), global_step)
                            self.logger.log_metrics("losses/clip_coef", self.args.clip_coef, global_step)
                        

                            if (
                                self.args.target_kl is None
                                or approx_kl <= self.args.target_kl
                            ):
                                mb_advantages = advantages.reshape(-1)[int_inds] # [0.1994]
                                if self.args.norm_adv:
                                    if mb_advantages.shape[0] > 1:
                                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                                            mb_advantages.std() + 1e-8
                                        )
                                # for some reason the advantages are mostly negative?
                                debug_mb_advantages.extend(mb_advantages.cpu().numpy())
                                # Policy loss
                                pg_loss1 = -mb_advantages * ratio
                                pg_loss2 = -mb_advantages * torch.clamp(
                                    ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                                )
                                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                                if pg_loss is None:
                                    print("pg_loss is None")
                                # has gradient
                                # Value loss
                                newvalue = newvalue.view(-1)
                                if self.args.clip_vloss:
                                    v_loss_unclipped = (newvalue - b_returns[int_inds]) ** 2
                                    
                                    v_clipped = b_values[int_inds] + torch.clamp(
                                        newvalue - b_values[int_inds],
                                        -self.args.clip_coef,
                                        self.args.clip_coef,
                                    )
                                    v_loss_clipped = (v_clipped - b_returns[int_inds]) ** 2
                                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                                    v_loss = 0.5 * v_loss_max.mean() # also has gradient
                                else:
                                    # formula: 
                                    v_loss = 0.5 * ((newvalue - b_returns[int_inds]) ** 2).mean()
                                    debug_v_loss.append(v_loss.item())

                                entropy_loss = entropy.mean()
                                #   Loss components:
                                # - Policy loss (pg_loss): Encourages the policy to choose actions that lead to higher rewards.
                                # - Value loss (v_loss): Ensures the value function accurately estimates the expected return.
                                # - Entropy loss (entropy_loss): Encourages exploration by penalizing low entropy.
                                student_loss = (
                                    pg_loss
                                    - self.args.ent_coef * entropy_loss
                                    + v_loss * self.args.vf_coef
                                )
                                debug_pg_loss.append(pg_loss.item())
                                debug_entropy_loss.append(entropy_loss.item())
                                
                                # if student_loss < 10:
                                loss += student_loss 
                                # else:
                                #     print(f"student_loss {student_loss}) is too high")
                                
                                        
                                if self.logger:
                                    self.logger.log_metrics(
                                        "losses/policy_loss_on_student_actions", pg_loss.item(), global_step
                                    )
                                    self.logger.log_metrics(
                                        "losses/entropy_on_student_actions", entropy_loss.item(), global_step
                                    )
                                    self.logger.log_metrics(
                                        "losses/value_loss_on_student_actions", v_loss.item(), global_step
                                    )
                                    self.logger.log_metrics(
                                        "losses/total_loss_on_student_actions", loss.item(), global_step
                                    )
                                    print(f"total_loss_on_student_actions {loss}: policy_loss {pg_loss} entropy_loss {entropy_loss} v_loss comp {v_loss * v_loss}")
                                calculated_student_loss = True
                            else: 
                                print(f"skipping student loss because approx_kl {approx_kl} > self.args.target_kl {self.args.target_kl}")
                                loss = torch.tensor(0.0).to(self.device)
                                
                            
                        # ==================teacher actions!!=========================
                        if len(octo_inds) > 0:
                            # depending on act_on_teach_actions teach or stud actions were exetuted in env,
                            # meaning the rewards are for them
                            executed_actions = b_actions[octo_inds] if self.cfg.model_config.act_on_teach_actions else b_st_actions[octo_inds]
                    
                            
                            # we need means of student with grad to optimize for
                            _, newlogprob, entropy, newvalue = self.scheduler.internal_policy.get_action_and_value(# prob of teach in the agent
                                obs.reshape((-1,))[octo_inds], executed_actions # teach or stud actions here
                            )
                            
                            # _, newlogprob, entropy, newvalue = self.scheduler.internal_policy.get_action_and_value(# prob of teach in the agent
                            #      obs.reshape((-1,))[octo_inds], student_actions[octo_inds]                  )
                            action_with_grad = self.scheduler.internal_policy.get_action(obs.reshape((-1,))[octo_inds], deterministic=True)
                            # if action_with_grad.requires_grad:
                            #     print("action_with_grad requires grad")
                            # else:
                            #     print("action_with_grad does NOT require grad")
                            
                            
                            
                            
                            # !! Teachers actions are too much outside distribution, to the prob needs clipping from below
                            if self.cfg.model_config.act_on_teach_actions:
                                EPS = 1e-6  # Small epsilon to prevent -inf issue
                                logratio = torch.clamp(newlogprob - logprobs.reshape(-1)[octo_inds], min=-10, max=10)
                                # because the teachers actions are very far away at hte begining logratio is a large neg number, and ration becomes 0 or even none
                                # the logprob goes to negative infty making updates unreliable, 
                            else:
                                logratio = newlogprob - logprobs.reshape(-1)[octo_inds]
                            
                            
                            # # we take only student actions for kl
                            
                            # !instead of clipping we could cpmpute KL only between student actions and teacher actions.!
                            ratio = logratio.exp() # == 1 so should learn smth?
                            
                            with torch.no_grad():
                                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean() # KL is calculated only for internal actions!
                                # This clipping coef could be changed
                                clipfracs += [
                                    ((ratio - 1.0).abs() > self.args.clip_coef)
                                    .float()
                                    .mean()
                                    .item()
                                ] # 0.0?
                                
                            # debug_ratio.extend(ratio.detach().cpu().numpy())
                            # debug_approx_kl.append(approx_kl.cpu().numpy())
                            # debug_clipfracs.extend(clipfracs)
                            
                            # print(f"clipfracs {clipfracs}")
                            if self.cfg.model_config.act_on_teach_actions:
                                self.logger.log_metrics("losses/value_to_clip_mean_on_teacher_actions", (ratio - 1.0).abs().mean().item(), global_step)
                                # self.logger.log_metrics("losses/clip_coef", self.args.clip_coef, global_step)
                                self.logger.log_metrics("losses/kl_loss_on_teacher_actions", approx_kl.item(), global_step)
                            else:
                                self.logger.log_metrics("losses/value_to_clip_mean_on_stud_actions2", 0, global_step)
                                self.logger.log_metrics("losses/kl_loss_on_stud_actions2", 0, global_step)
                        
                            # Important mod: On student actions there is a kl check, but on teacher actions we always backprop
                                
                            if self.cfg.model_config.act_on_teach_actions or (
                                self.args.target_kl is None or approx_kl <= self.args.target_kl # check for stud actions
                            ):                           
                                mb_advantages = advantages.reshape(-1)[octo_inds]
                                if self.args.norm_adv:
                                    if mb_advantages.shape[0] > 1:
                                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                                            mb_advantages.std() + 1e-8
                                        )
                                # for some reason the advantages are mostly negative?
                                debug_mb_advantages.extend(mb_advantages.cpu().numpy())
                                # Policy loss
                                pg_loss1 = -mb_advantages * ratio
                                pg_loss2 = -mb_advantages * torch.clamp(
                                    ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                                )
                                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                                # has gradient
                                # Value loss
                                newvalue = newvalue.view(-1)
                                if self.args.clip_vloss:
                                    v_loss_unclipped = (newvalue - b_returns[octo_inds]) ** 2
                                    
                                    v_clipped = b_values[octo_inds] + torch.clamp(
                                        newvalue - b_values[octo_inds],
                                        -self.args.clip_coef,
                                        self.args.clip_coef,
                                    )
                                    v_loss_clipped = (v_clipped - b_returns[octo_inds]) ** 2
                                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                                    v_loss = 0.5 * v_loss_max.mean() # also has gradient
                                else:
                                    v_loss = 0.5 * ((newvalue - b_returns[octo_inds]) ** 2).mean()
                                    debug_v_loss.append(v_loss.item())
                                if self.logger:
                                    self.logger.log_metrics(
                                        "losses/value_loss_on_teacher_actions", v_loss.item(), global_step
                                    )
                                entropy_loss = entropy.mean()
                                #   Loss components:
                                # - Policy loss (pg_loss): Encourages the policy to choose actions that lead to higher rewards.
                                # - Value loss (v_loss): Ensures the value function accurately estimates the expected return.
                                # - Entropy loss (entropy_loss): Encourages exploration by penalizing low entropy.
                                total_loss_on_teacher_act = torch.tensor(0.0).to(self.device)
                                
                                if self.args.use_mae_on_teach_action:
                                    mae_penalty_loss = (b_actions[octo_inds] - action_with_grad).abs().sum() # penalty for being far from teacher
                                else:
                                    mae_penalty_loss = 0
                                if self.args.use_mse_on_teach_action:
                                    mse_penalty_loss = (b_actions[octo_inds] - action_with_grad).pow(2).sum()
                                else:
                                    mse_penalty_loss = 0
                                # solf penalty based on kl, because thrwshold does not work that well
                                kl_penalty = min(1.0, 10.0 / (approx_kl + 1e-10))  # Reduce penalty when KL is high
                                # mechanism to adaptively scale a penalty term based on the KL divergence (approx_kl). Here's what it does:
                                # +1e-10 is added to avoid division by zero
                                
                                if self.args.use_mae_on_teach_action:
                                    coef = kl_penalty if self.args.use_kl_penalty_scaling_on_teach_action else 1
                                    total_loss_on_teacher_act += (mae_penalty_loss * self.args.mae_coef +
                                                                  mse_penalty_loss * self.args.mse_coef) * coef
                                    
                                if self.args.use_value_loss_on_teach_action:
                                    total_loss_on_teacher_act += v_loss * self.args.vf_coef
                                    
                                if self.args.use_action_loss_on_teach_action:
                                    total_loss_on_teacher_act += pg_loss
    
                                
                                # total_loss_on_teacher_act = (
                                #     # pg_loss
                                #     # - self.args.ent_coef * entropy_loss
                                #     + v_loss * self.args.vf_coef
                                #     + self.args.teacher_student_coef * penalty_loss
                                # ) # * 0.1 * kl_penalty
                                
                                # if total_loss_on_teacher_act < 10:
                                #     loss += total_loss_on_teacher_act 
                                # else:
                                #     print(f"total_loss_on_teacher_act {total_loss_on_teacher_act} is too high")
                                
                                # !!! we add teacher's loss 
                                loss += total_loss_on_teacher_act
                                
                                debug_pg_loss.append(pg_loss.item())
                                debug_entropy_loss.append(entropy_loss.item())
                                
                                if self.logger:
                                    self.logger.log_metrics(
                                        "losses/penalty_loss", ((mae_penalty_loss * self.args.mae_coef +
                                                                  mse_penalty_loss * self.args.mse_coef) * coef), global_step
                                    )
                                    self.logger.log_metrics(
                                        "losses/mae", mae_penalty_loss, global_step
                                    )
                                    self.logger.log_metrics(
                                        "losses/mse", mse_penalty_loss, global_step
                                    )
                                    self.logger.log_metrics(
                                        "losses/policy_loss_on_teacher_actions", pg_loss.item(), global_step
                                    )
                                    self.logger.log_metrics(
                                        "losses/entropy_on_teacher_actions", entropy_loss.item(), global_step
                                    )
                                    self.logger.log_metrics(
                                        "losses/value_loss_on_teacher_actions", v_loss.item(), global_step
                                    )
                                    self.logger.log_metrics(
                                        "losses/total_loss_on_teacher_actions", total_loss_on_teacher_act.item(), global_step
                                    )
                                    print(f"total_loss_on_teacher_act {total_loss_on_teacher_act}: policy_loss {pg_loss} entropy_loss {entropy_loss} v_loss comp {v_loss * v_loss} ")
                                calculated_teacher_loss = True
                            # else:
                            #     print(f"skipping teacher loss because approx_kl {approx_kl} > 10 ")
                                
                        if self.logger:
                            self.logger.log_metrics("losses/joint_opimized_loss", loss.item(), global_step)
                              
                        debug_total_loss.append(loss.item())
                        
                        
                        if calculated_student_loss or calculated_teacher_loss: # only if either student or teach loss was not skipped
                            self.optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(
                                self.scheduler.internal_policy.parameters(), self.args.max_grad_norm
                            )
                            self.optimizer.step()

                    # if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    #     print(f"BR1: breaking because approx_kl {approx_kl} > self.args.target_kl {self.args.target_kl}")
                    #     break
                update_time = time.time() - update_time
                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                
                del b_values
                
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )
                if self.logger is not None:
                    
                    # regarding scheduling. in the gloabal step when do we use octo?
                    if self.scheduler.scheduling_strategy == "octo_than_internal":
                        used_octo = 1 if iteration < self.cfg.scheduler_config.iteration_to_switch else 0
                        self.logger.log_metrics(
                            "scheduling/used_octo", used_octo, global_step
                        )                  
                    
                    self.logger.log_metrics(
                        "charts/learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        global_step,
                    ) # for now keep it constant its kinda hard to schedule it because different policies are mixed

                    self.logger.log_metrics(
                        "losses/old_approx_kl_divergence", old_approx_kl.item(), global_step
                    ) # not sure if we need this one?
                    self.logger.log_metrics(
                        "losses/approx_kl_divergence", approx_kl.item(), global_step
                    )
                    self.logger.log_metrics(
                        "losses/clipped_fraction", np.mean(clipfracs), global_step
                    ) # very important how often we clipp octo actions!!!
                    # TODO: add ignored fraction
                    self.logger.log_metrics(
                        "losses/explained_variance", explained_var, global_step
                    )
                # only if it train the exectuon gets here
                train_iteration_number += 1
                # update progress bar
                progress_bar.update(1)    
            # -----------------saving the model------------------------ #
            model_path = f"{self.hydra_log_dir}/{self.args.run_name}/final_ckpt.pt"
            self.scheduler.save(model_path)
            print(f"Final Model saved to {model_path}")
            
        # except Exception as e:
        #     print(f"Exception: {e}")
        #     progress_bar.close()
        #     self.envs.close()
        #     raise e
        # finally:
        if True:
            # we do not save final model in case of failure, makes an easy way to figure if run succeeded
            progress_bar.close()           
          
            # Save debug lists to files
            debug_dir = os.path.join(self.hydra_log_dir, self.args.run_name, "debug_arrays")
            os.makedirs(debug_dir, exist_ok=True)
            
            np.save(os.path.join(debug_dir, "debug_returns.npy"), np.array(debug_returns))
            np.save(os.path.join(debug_dir, "debug_ratio.npy"), np.array(debug_ratio))
            np.save(os.path.join(debug_dir, "debug_mb_advantages.npy"), np.array(debug_mb_advantages))
            np.save(os.path.join(debug_dir, "debug_approx_kl.npy"), np.array(debug_approx_kl))
            np.save(os.path.join(debug_dir, "debug_v_loss.npy"), np.array(debug_v_loss))
            np.save(os.path.join(debug_dir, "debug_clipfracs.npy"), np.array(debug_clipfracs))
            
            self.envs.close()
    
    # ----------- Wrappers to have same api as a model ----------------#
    def to(self, device):
        self.scheduler.to(device)
        self.device = device
        return self
    
    def eval(self):
        self.scheduler.eval()
        
    def train(self):
        self.scheduler.train()
    
    def forward(self, obs, deterministic=True):
        # This is important. We evaluate a deperministic policy, so the mean is returned
        return self.scheduler.internal_policy.get_action(
                obs, deterministic=deterministic
            )
        
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
