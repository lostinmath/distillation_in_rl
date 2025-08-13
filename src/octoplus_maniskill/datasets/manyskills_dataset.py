import logging
from functools import partial

import dlimp as dl
import h5py
import jax
import jax.numpy as jnp

# from torch.utils.data import Dataset, DataLoader
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from gymnasium.wrappers import TimeLimit
from mani_skill2.utils.common import flatten_state_dict
from mani_skill2.utils.io_utils import load_json
from tqdm.notebook import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
#############################h5 to rdls########################################

import h5py
import torch as th
from mani_skill2.utils.io_utils import load_json


class DatasetConverter:
    #  it requires _inputs, element_spec methods?
    def __init__(self) -> None:
        pass

    def h5_to_rdls(self, dataset_file: str):
        data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        json_data = load_json(json_path)
        episodes = json_data["episodes"]
        env_info = json_data["env_info"]
        env_id = env_info["env_id"]
        env_kwargs = env_info["env_kwargs"]

        obs_state = []
        obs_rgbd = []
        actions = []
        total_frames = 0
        if load_count == -1:
            load_count = len(episodes)
        for eps_id in range(load_count):
            eps = episodes[eps_id]
            trajectory = data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            # convert the original raw observation with our batch-aware function
            obs = convert_observation(trajectory["obs"])
            # we use :-1 to ignore the last obs as terminal observations are included
            # and they don't have actions
            obs_rgbd.append(obs["rgbd"][:-1])
            obs_state.append(obs["state"][:-1])
            actions.append(trajectory["actions"])
        obs_rgbd = np.vstack(self.obs_rgbd)
        obs_state = np.vstack(self.obs_state)
        actions = np.vstack(self.actions)
        logger.info(f"Convereted h5 dataset with {len(self.episodes)} episodes to rlds")

    def __len__(self):
        return len(self.obs_rgbd)

    def __getitem__(self, idx):
        action = th.from_numpy(self.actions[idx]).float()
        rgbd = self.obs_rgbd[idx]
        rgbd = rescale_rgbd(rgbd)
        # permute data so that channels are the first dimension as PyTorch expects this
        rgbd = th.from_numpy(rgbd).float().permute((2, 0, 1))
        state = th.from_numpy(self.obs_state[idx]).float()
        return dict(rgbd=rgbd, state=state), action


####################################################################
# TODO: need some function to iterate over diff envs and tasks


def convert_observation(observation):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images
    # image data is not scaled here and is kept as uint16 to save space
    image_obs = observation["image"]
    rgb = image_obs["base_camera"]["rgb"]
    depth = image_obs["base_camera"]["depth"]
    rgb2 = image_obs["hand_camera"]["rgb"]
    depth2 = image_obs["hand_camera"]["depth"]

    # we provide a simple tool to flatten dictionaries with state data
    state = np.hstack(
        [
            flatten_state_dict(observation["agent"]),
            flatten_state_dict(observation["extra"]),
        ]
    )

    # combine the RGB and depth images
    rgbd = np.concatenate([rgb, depth, rgb2, depth2], axis=-1)
    obs = dict(rgbd=rgbd, state=state)
    return obs


def rescale_rgbd(rgbd, scale_rgb_only=False):
    # rescales rgbd data and changes them to floats
    rgb1 = rgbd[..., 0:3] / 255.0
    rgb2 = rgbd[..., 4:7] / 255.0
    depth1 = rgbd[..., 3:4]
    depth2 = rgbd[..., 7:8]
    if not scale_rgb_only:
        depth1 = rgbd[..., 3:4] / (2**10)
        depth2 = rgbd[..., 7:8] / (2**10)
    return np.concatenate([rgb1, depth1, rgb2, depth2], axis=-1)


# For future may be moved to utils
# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out
