import os
import time
from dataclasses import dataclass
from typing import Optional

# ===================== TD3 ===================== #
@dataclass
class TD3Config:
    exp_name: str = "debug"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    total_timesteps: int = (
        100_000  # 1_000_000 # 1000000 1k - 0 hours 0 minutes 18 seconds
    )
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 100  # int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 1  # 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    # ------------ Copies of params from env configs----------------#
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    num_envs: int = 256  # 512
    """the number of parallel environments"""
    # render_mode: str = "all"
    """the environment rendering mode"""
    # ---------------------------------------------------------------#

    def __post_init__(self):
        if self.exp_name is None:
            self.exp_name = os.path.basename(__file__)[: -len(".py")]
            self.run_name = (
                f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
            )
        else:
            self.run_name = self.exp_name

    def __to_dict__(self):
        return self.__dict__


# ===================== PPO ===================== #
@dataclass
class PPORGBConfig:
    exp_name: Optional[str] = "default-rgb-pushcube-experiment"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    model_device: str = "cuda"
    # ------------ Copies of params from env configs--------------
    #env_id: str = "PickCube-v1"
    """the id of the environment"""
    num_envs: int = 32  # 512
    """the number of parallel environments"""

    # render_mode: str = "all"
    """the environment rendering mode"""
    ####################################################
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    # checkpoint: str | None = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    num_steps: int = 20  # 50
    """the number of steps to run in each environment per policy rollout"""

    # Algorithm specific arguments
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 250_000  # 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""

    # partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""

    # num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16  # 32
    """the number of mini-batches"""
    update_epochs: int = 8  # 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    teacher_student_coef: float = 0.0
    """coefficient of the teacher-student loss"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    # eval_freq: int = 10  # 25
    """evaluation frequency in terms of iterations"""
    # save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = True

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    run_name: str = "default_run_name"
    # ============= Modifications =================
    use_mae_on_teach_action: bool = True
    mae_coef: float = 0.0005
    use_mse_on_teach_action: bool = False
    mse_coef: float = 0
    use_value_loss_on_teach_action: bool = False
    # vf_coef
    use_action_loss_on_teach_action: bool = False
    
    use_kl_penalty_scaling_on_teach_action: bool = False
    
    act_on_teach_actions: bool = True
    # take teach actions in the env if scheduler selects them
    

    ##################### Agent configs ########num_envs###########
    feature_net_type: str = "NatureCNN"
    feature_net_activation_layer: str = "ReLU"

    critic_type: str = "BasicCritic"
    critic_activation_layer: str = "ReLU"

    actor_type: str = "BasicActor"
    actor_activation_layer: str = "ReLU"

    def __post_init__(self):
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = self.total_timesteps // self.batch_size
        if self.exp_name is None:
            self.exp_name = os.path.basename(__file__)[: -len(".py")]
            self.run_name = (
                f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
            )
        else:
            self.run_name = self.exp_name
    
    def __to_dict__(self):
        return {
            "exp_name": self.exp_name,
            "feature_net_type": self.feature_net_type,
            "feature_net_activation_layer": self.feature_net_activation_layer,
            "critic_type": self.critic_type,
            "critic_activation_layer": self.critic_activation_layer,
            "actor_type": self.actor_type,
            "actor_activation_layer": self.actor_activation_layer,
            "seed": self.seed,
            # "num_envs": self.num_envs, # already in env config, dublicate but we need it for the batch size
            "learning_rate": self.learning_rate,
            "total_timesteps": self.total_timesteps,
            # "num_steps": self.num_steps,
            "num_minibatches": self.num_minibatches,
            "update_epochs": self.update_epochs,
            #"env_id": self.env_id,
            "exp_name": self.exp_name,
            "model_device": self.model_device,
            "clip_coef": self.clip_coef,
            
        }


@dataclass
class PPOStableBaselines3Config:

    policy_type: str = "CNNPolicy"

    def __post_init__(self):
        if self.policy_type not in ["CNNPolicy", "MultiInputPolicy"]:
            raise ValueError(
                "Policy type must be either 'CNNPolicy' or 'MultiInputPolicy'"
            )


@dataclass
class PPOCleanRLConfig:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    checkpoint: str | None = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    # ================== Training related ===================================

    # num_eval_envs: int = 8
    """the number of parallel evaluation environments. Will loo[ untill evaluated on this number of envs"""
    # ===================== Algorithm specific arguments =====================
    # env_id: str = "PickCube-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    # num_envs: int = 512
    """the number of parallel environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    finite_horizon_gae: bool = True
    # ================== Logging ===========================
    # save_train_video: bool = False
    # save_train_video_freq: Optional[int] = None
    # """frequency to save training videos in terms of iterations"""
    # save_eval_video: bool = False
    # save_eval_video_freq: Optional[int] = None
    # =============== to be filled in runtime ===============
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    def __post_init__(self):
        # compute the batch size, mini-batch size, and number of iterations
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = max(1, int(self.batch_size // self.num_minibatches))
        self.num_iterations = self.total_timesteps // self.batch_size
        if self.exp_name is None:
            self.exp_name = f"PPO Experiment on {self.env_id}"
            run_name = (
                f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
            )
        else:
            run_name = self.exp_name
