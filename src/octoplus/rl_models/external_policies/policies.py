"""
Function to load Octo model from the API or from the local pickle file.
"""
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
from agents.policies import ObsDict

from octoplus.src.config_dataclasses.env_configs import ManiSkillConfig
from octoplus.src.config_dataclasses.experiments import ExperimentConfig
from octoplus.src.config_dataclasses.model_configs import PPORGBConfig
from octoplus.src.utils.custom_logging import Logger
from octoplus.src.rl_models.nature_cnn import Agent
from octoplus.src.rl_models.shared_model_api import AbstractedRrAlgo
from octoplus.src.utils.utils_for_octo import reshape_obs_for_octo
from octoplus.src.config_dataclasses.policy_scheduling import PolicySchedulingConfig


class OctoPolicy:
    """
    Wrapper around Remote Agent, where Octo resides
    """

    def __init__(self, text_instruction: str, device="cuda"):
        self.octo: RemoteAgent = RemoteAgent.by_config()
        self.text_instruction = text_instruction
        self.device = device

    def act(self, obs, pos: None | int = None):
        # needs resizing to 256x256
        # https://github.com/octo-models/octo/blob/main/examples/01_inference_pretrained.ipynb
        # is it a dict?
        if type(obs) == dict:
            obs = obs["rgb"]

        if pos is not None:
            # we take only element in position pos
            reshaped_obs = reshape_obs_for_octo(obs[pos])
        else:
            # we take all elements
            reshaped_obs = reshape_obs_for_octo(obs)
            #
        if len(obs.shape) == 3:
            self.octo.reset(reshaped_obs, self.text_instruction)
            result = self.octo.act_batch([{"image_primary": reshaped_obs[0]}])
            # result = self.octo.act({"image_primary" : reshaped_obs})
        else:
            result = self.octo.act_batch(
                [{"image_primary": obs_img} for obs_img in reshaped_obs]
            )
        recommended_actions = torch.tensor(
            np.array([el[0]["action"][0] for el in result])
        )
        return recommended_actions


class RandomPolicy:
    def __init__(self, device="cuda"):
        self.device = device

    def act(self, obs):
        action_mean = torch.zeros(7)
        action_std = torch.ones(7)
        probs = Normal(action_mean, action_std)
        return probs.sample().to(self.device)
