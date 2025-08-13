"""
Visualization utilities
"""
import imageio
import numpy as np
from IPython.display import Video, display
from PIL import Image


def create_recording_with_octo(
    env, octo_model, video_path: str = "Franka_robot_action_octo.mp4", num_steps=500
):

    import jax

    OCTO_INPUT_WIDTH = 256
    OCTO_INPUT_HEIGHT = 256
    task = octo_model.create_tasks(texts=["pick up the cube"])
    images = []
    obs, _ = env.reset()
    for _ in range(num_steps):

        img = env.unwrapped.render_cameras()[:, :512, :]  # We cut the main camera
        images.append(img)
        reshaped_image = np.array(
            Image.fromarray(img).resize((OCTO_INPUT_WIDTH, OCTO_INPUT_HEIGHT))
        )
        image_batch = reshaped_image[np.newaxis, np.newaxis, ...]
        observation = {
            "image_primary": image_batch,
            "timestep_pad_mask": np.array([[True]]),
        }

        octo_action = octo_model.sample_actions(
            observation,
            task,
            unnormalization_statistics=octo_model.dataset_statistics["bridge_dataset"][
                "action"
            ],
            rng=jax.random.PRNGKey(0),
        )
        octo_action = np.array(octo_action[0][0])

        obs, reward, terminated, truncated, info = env.step(octo_action)
        done = truncated or terminated

        if done:
            env.reset()

    imageio.mimsave(video_path, images, fps=30)
    display(Video(video_path, embed=True))
