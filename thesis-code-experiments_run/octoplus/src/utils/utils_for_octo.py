import cv2
import numpy as np
import torch

from octoplus.src.constants import OCTO_INPUT_HEIGHT, OCTO_INPUT_WIDTH


def reshape_obs_for_octo(obs: torch.Tensor) -> np.ndarray[float]:
    """
    Reshape the observations for the octo model
    """
    if len(obs.shape) == 4: # batch of images
        obs_numpy = obs.detach().clone().cpu().numpy()
        # would expect shape to be ['batch size', 512, 640, 3] for mani_skills
        # obs_resized = np.array(
        #     [cv2.resize(img, (OCTO_INPUT_WIDTH, OCTO_INPUT_HEIGHT)) for img in obs_numpy]
        # )
    elif len(obs.shape) == 3: # only 1 image
        obs_numpy = obs.detach().clone().cpu().numpy()
        obs_numpy = obs_numpy[None, ...]
        # obs_resized = np.array(cv2.resize(obs_numpy, (OCTO_INPUT_WIDTH, OCTO_INPUT_HEIGHT)))[None, ...]
    else:
        raise ValueError(f"Expected obs to have shape (batch_size, x, y, 3) or (x, y, 3), got {obs.shape}")
    return obs_numpy
