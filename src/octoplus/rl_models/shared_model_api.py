import gymnasium as gym
import torch

from octoplus.src.config_dataclasses.experiments import ExperimentConfig
from octoplus.src.rl_models.rl_model_parent import AbstractedRrAlgo


def load_model(
    model_path: str, cfg: ExperimentConfig, hydra_log_dir: str, envs, mode:str = "train"
) -> AbstractedRrAlgo:
    """
    Loads model from saved checkpoint for eval or to continue training.
    Args:
        model_path: Path to the saved model checkpoint.
        cfg: ExperimentConfig object.
        hydra_log_dir: Directory to save logs.
        envs: Environment to train the model in.
    Returns:
        An implemented model with methods of
        octoplus.src.rl_models.rl_model_parent.AbstractedRrAlgo.
    """
    model_state_dict = torch.load(model_path)
    if cfg.model_type == "ppo_rgb":
        from octoplus.src.rl_models.ppo_rgb_old import PPORgb

        model = PPORgb(envs=envs, cfg=cfg, hydra_log_dir=hydra_log_dir)
        if mode == "train":
            model.model_setup()
        elif mode == "eval":
            model.model_eval_setup()
            
        model.scheduler.internal_policy.load_state_dict(model_state_dict)
        model.to(cfg.model_config.model_device)
        return model


def get_model(
    cfg: ExperimentConfig, env: gym.Env, hydra_log_dir: str = "run"
) -> AbstractedRrAlgo:
    """
    Get a new model based from values in config

    Args:
        cfg (ExperimentConfig): The configuration for the experiment.
        env: The environment in which the model will be trained.
        model_cfg: The configuration for the model.
    Returns:
      An implemented model with methods of
    octoplus.src.rl_models.rl_model_parentAbstractedRrAlgo.
    """
    if cfg.path_to_loaded_model is not None:
        return load_model(
            model_path=cfg.path_to_loaded_model,
            cfg=cfg,
            hydra_log_dir=hydra_log_dir,
            envs=env,
        )
    if cfg.model_type == "ppo_rgb":
        from octoplus.src.rl_models.ppo_rgb_old import PPORgb

        model = PPORgb(envs=env, cfg=cfg, hydra_log_dir=hydra_log_dir)
    else:
        raise NotImplementedError(
            f"Only 'ppo_stable_baselines3' and 'ppo_rgb_cleanrl' 'ppo_rgb'\
                                  model types are implemented."
        )

    return model
