import os
from typing import Tuple

from octoplus.src.config_dataclasses.experiments import ExperimentConfig

# from torch.utils.data import DataLoader
from octoplus.src.datasets.manyskills2_dataset import ManiSkill2DatasetImages


def get_dataloaders(
    config: ExperimentConfig,
) -> Tuple[DataLoader | None, DataLoader | None, DataLoader | None]:
    """
    Creates dataloaders for training, validation and testing

    """
    # thesis-code/octoplus/demos/v0/rigid_body/LiftCube-v0/trajectory.state.pd_ee_delta_pose.h5
    print(f"Current path: {os.getcwd()}")
    dataset = ManiSkill2DatasetImages(config.dataset_path, load_count=1)
    obs, action = dataset[0]
    print("RGBD:", obs["rgbd"].shape)
    print("State:", obs["state"].shape)
    print("Action:", action.shape)
    # TODO: take params from config
    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    return train_dataloader, None, None
