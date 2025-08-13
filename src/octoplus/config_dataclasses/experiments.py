import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from octoplus.src.config_dataclasses.env_configs import ManiSkillConfig
from octoplus.src.config_dataclasses.model_configs import (
    PPORGBConfig,
)
from octoplus.src.config_dataclasses.policy_scheduling import PolicySchedulingConfig
from octoplus.src.config_dataclasses.model_configs import TD3Config

######################### Logging ########################
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#######################################################

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
    }


@dataclass
class ExperimentConfig:
    train: bool = True
    ###################### model ##################
    model_type: str = "ppo_rgb"
    model_config: Dict[Any, Any] = field(default_factory=dict)
    ################### env #########################
    env_type: str = "mani_skill"
    env_config: Dict[Any, Any] = field(default_factory=dict)

    # ============= Policy scheduling =============
    scheduler_config: PolicySchedulingConfig | None = None
    # field(default_factory=PolicySchedulingConfig)

    ############# training parameters ######## or inside the model params?
    # num_episodes: int = 1
    learning_rate: float = 0.001

    ############# evaluation and video recordings ###########
    evaluate: bool = True
    """if toggled, only runs evaluation with the given model"""
    eval_freq: Optional[int] = 2
    """evaluation frequency in terms of iterations"""
    # checkpoint are saved for every evaluation run
    save_train_videos: bool = True
    save_eval_videos: bool = True

    save_train_video_freq: int = 1
    ckpt_save_frequency: int = 10_000

    ############# cuda stuff ######################

    device: str = "cuda"
    use_jax: bool = False

    ###################################################
    path_to_loaded_model: str | None = None

    def __post_init__(self):
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("Device must be either 'cuda' or 'cpu'")
        
        # Outdated parameters that were used in old runs for backwards compatibility
        if "octo_temperature" in self.scheduler_config.keys():
            del self.scheduler_config["octo_temperature"]
            logging.info("octo_temperature is not used in the current version. Dropping it.")
        if "env_id" in self.model_config.keys():
            del self.model_config["env_id"]
            logging.info("env_id is not used in the current version. Dropping it.")

        # if self.model_type == "ppo_rgb_cleanrl":
        #     logging.info("Casting model_config to PPOCleanRLConfig type")
        #     self.model_config = PPOCleanRLConfig(**self.model_config)
        # elif self.model_type == "ppo_stable_baselines3":
        #     logging.info("Casting model_config to PPOStableBaselines3Config type")
        #     self.model_config = PPOStableBaselines3Config(**self.model_config)
        if self.model_type == "ppo_rgb":
            logging.info("Casting model_config to PPORGBConfig type")
            self.model_config = PPORGBConfig(**self.model_config)
        elif self.model_type == "td3":
            logging.info("Casting model_config to TD3Config type")
            self.model_config = TD3Config(**self.model_config)
        else:
            raise ValueError(
                f"model_type must be 'ppo_rgb'. Got {self.model_type} instead."
            )

        if self.env_type == "mani_skill":
            logging.info("Casting env_config to ManiSkillConfig type")
            self.env_config = ManiSkillConfig(**self.env_config)
        
        if self.scheduler_config is not None and not isinstance(
            self.scheduler_config, PolicySchedulingConfig
        ):
            if "text_instruction" not in self.scheduler_config:
                self.scheduler_config["text_instruction"] = INSTRUCTIONS[self.env_config.env_id]
                logger.info("Text instruction set to default for the env.")
                
            
                
            self.scheduler_config = PolicySchedulingConfig(**self.scheduler_config)
        
        logging.info(
            f"Scheduling scheduling_strategy was set to {self.scheduler_config.scheduling_strategy}."
        )

        if self.model_type == "ppo_rgb" and self.save_train_videos and round(self.save_train_video_freq // self.model_config.batch_size)  == 0:
            raise ValueError("save_train_video_freq is too low, you might not want to save every iteration. n * {}, where n \in [2, {})".format(self.model_config.batch_size, self.model_config.num_iterations))
        if self.model_type == "ppo_rgb" and round(self.ckpt_save_frequency // self.model_config.batch_size) == 0:
            raise ValueError("ckpt_save_frequency is too low, you might not want to save every iteration. n * {}, where n \in [2, {})".format(self.model_config.batch_size, self.model_config.num_iterations))

    def __to_dict__(self):
        base_dict = {
            "model_type": self.model_type,
            "env_type": self.env_type,
            # "num_episodes": self.num_episodes,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "use_jax": self.use_jax,
            "path_to_loaded_model": self.path_to_loaded_model,
            "evaluate": self.evaluate,
            "eval_freq": self.eval_freq,
            "save_train_videos": self.save_train_videos,
            "save_eval_videos": self.save_eval_videos,
            "save_train_video_freq": self.save_train_video_freq,
        }
        model_config_dict = self.model_config.__to_dict__()
        env_config_dict = self.env_config.__to_dict__()
        scheduler_config_dict = self.scheduler_config.__to_dict__()
        print(f"model_config_dict: {model_config_dict}")
        print(f"env_config_dict: {env_config_dict}")
        print(f"scheduler_config_dict: {scheduler_config_dict}")
        print(f"base_dict: {base_dict}")
        return {
            **base_dict,
            **model_config_dict,
            **env_config_dict,
            **scheduler_config_dict,
        }
