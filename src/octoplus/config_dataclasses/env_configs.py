from dataclasses import dataclass

import torch

@dataclass
class ManiSkillConfig:
    obs_mode: str = "rgb"
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str = "all"
    sim_backend: str = "gpu"  # "cpu" # "gpu"
    reward_mode: str = "dense"
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    num_envs: int = 256  # 512
    """the number of parallel environments"""
    num_steps: int = 20  # 50
    """the number of steps to run in each environment per policy rollout"""
    include_state: bool = False
    partial_reset: bool = True
    """if True will reset apon reaching max number of steps,
    othewise will reset only upon successful termination"""
    image_height: int = 512
    image_width: int = 512
    use_render_camera_as_input: bool = True #  this the param for my custom wrapper
    use_only_rgb: bool = False
    seed: int = 13

    # completed in runtime
    # env_device: str = "cpu"

    def __post_init__(self):
        if self.sim_backend == "gpu":
            if torch.cuda.is_available():
                self.env_device = "cuda"
            else:
                raise ValueError("CUDA is not available.")
        elif self.sim_backend == "cpu":
            self.env_device = "cpu"
        else:
            raise ValueError("sim_backend must be either 'cpu' or 'gpu'.")

    def __to_dict__(self):
        return {
            "env_config/obs_mode": self.obs_mode,
            "env_config/control_mode": self.control_mode,
            "env_config/render_mode": self.render_mode,
            "env_config/sim_backend": self.sim_backend,
            "env_config/reward_mode": self.reward_mode,
            "env_config/env_id": self.env_id,
            "env_config/num_envs": self.num_envs,
            "env_config/num_steps": self.num_steps,
            "env_config/include_state": self.include_state,
            "env_config/partial_reset": self.partial_reset,
            "env_config/use_render_camera_as_input": self.use_render_camera_as_input,
        }


""""
[mani skills docs] [https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html]
All “pd” controllers translate input actions to target joint positions

(and velocities) for the internal built-in PD controller.
All the controllers have a normalized action space ([-1, 1]),
except arm_pd_joint_pos and arm_pd_joint_pos_vel.

For simplicity, we use the name of the arm controller to
represent each combination of the arm and gripper controllers,
since there is only one gripper controller currently.
For example, pd_joint_delta_pos is short for arm_pd_joint_delta_pos
+ gripper_pd_joint_pos.


from error: ['pd_joint_delta_pos', 'pd_joint_pos', 'pd_ee_delta_pos',
'pd_ee_delta_pose', 'pd_ee_pose', # these 2 have 7 dim
'pd_joint_target_delta_pos',
'pd_ee_target_delta_pos',
'pd_ee_target_delta_pose', # this also has 7 dim
'pd_joint_vel',
'pd_joint_pos_vel', 'pd_joint_delta_pos_vel']


# also from dccs above
# Arm controllers

arm_pd_joint_pos (7-dim): It can be used for motion planning, but note that the target velocity is always 0.

arm_pd_joint_delta_pos (7-dim):

arm_pd_joint_target_delta_pos (7-dim):

arm_pd_ee_delta_pos (3-dim): only control position without rotation

arm_pd_ee_delta_pose (6-dim): both position and rotation  are controlled. Rotation is represented as axis-angle in the end-effector frame.
arm_pd_ee_target_delta_pos (3-dim):
arm_pd_ee_target_delta_pose (6-dim): is the transformation of the end-effector.

is the delta pose induced by the action.

arm_pd_joint_vel (7-dim): only control target joint velocities. Note that the stiffnessis set to 0.

arm_pd_joint_pos_vel (14-dim): the extension of arm_pd_joint_pos to support target velocities

arm_pd_joint_delta_pos_vel (14-dim): the extension of arm_pd_joint_delta_pos to support target velocities
"""
