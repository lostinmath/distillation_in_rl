import logging

import h5py
import numpy as np
from mani_skill2.utils.common import flatten_state_dict
from mani_skill2.utils.io_utils import load_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
#############################h5 to rdls########################################
import h5py
import rlds
import tensorflow as tf


def create_rlds_dataset(episodes):
    def generator():
        for episode in episodes:
            yield {
                rlds.OBSERVATION: np.array([step["observation"] for step in episode]),
                rlds.ACTION: np.array([step["action"] for step in episode]),
                rlds.REWARD: np.array([step["reward"] for step in episode]),
                rlds.DISCOUNT: np.array([step["discount"] for step in episode]),
                rlds.IS_TERMINAL: np.array([step["is_terminal"] for step in episode]),
            }

    return tf.data.Dataset.from_generator(
        generator,
        output_signature={
            rlds.OBSERVATION: tf.TensorSpec(
                shape=(None, *episodes[0][0]["observation"].shape), dtype=tf.uint8
            ),
            rlds.ACTION: tf.TensorSpec(
                shape=(None, *episodes[0][0]["action"].shape), dtype=tf.float32
            ),
            rlds.REWARD: tf.TensorSpec(shape=(None,), dtype=tf.float32),
            rlds.DISCOUNT: tf.TensorSpec(shape=(None,), dtype=tf.float32),
            rlds.IS_TERMINAL: tf.TensorSpec(shape=(None,), dtype=tf.bool),
        },
    )


def h5_to_rdls(
    h5_dataset_path: str = "./demos/v0/rigid_body/LiftCube-v0/trajectory.state.pd_ee_delta_pose.h5",
    rdls_dataset_path: str = "demos_rdls",
    load_count: int = 100,
):
    """Converts h5 dataset to rdls dataset

    Args:
        h5_dataset_path (str): path to h5 dataset
        rdls_dataset_path (str): path to save rdls dataset
        load_count (int, optional): number of episodes to load. Defaults to 100.
    """
    # Load the .h5 file
    data = h5py.File(h5_dataset_path, "r")
    json_path = h5_dataset_path.replace(".h5", ".json")
    json_data = load_json(json_path)
    episodes = json_data["episodes"]
    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    env_kwargs = env_info["env_kwargs"]

    trajectories = []
    for eps_id in range(load_count):
        eps = episodes[eps_id]
        trajectory = data[f"traj_{eps['episode_id']}"]
        trajectories.append(trajectory)
    print(f"number of trajectories: {len(trajectories)}")

    episodes = []

    for trajectory in trajectories:
        # Assuming each trajectory is a group with datasets 'rgb', 'actions', and 'terminals'
        rgb_images = trajectory["obs"][:]  # Shape: (num_steps, height, width, channels)
        actions = trajectory["actions"][:]  # Shape: (num_steps, action_dim)
        terminals = trajectory["success"][:]  # Shape: (num_steps,)

        episode = []
        for i in range(len(actions)):
            step = {
                "observation": rgb_images[i],
                "action": actions[i],
                "reward": 1.0,  # Assuming reward is 0 for all steps (adjust if needed)
                "discount": 0.01,  # Assuming no discounting (adjust if needed)
                "is_terminal": terminals[i],
            }
            episode.append(step)

        episodes.append(episode)
    print(f"number of episodes: {len(episodes)}")

    rlds_dataset = create_rlds_dataset(episodes)
    # Save the dataset
    tf.data.experimental.save(rlds_dataset, rdls_dataset_path)
    return rlds_dataset
