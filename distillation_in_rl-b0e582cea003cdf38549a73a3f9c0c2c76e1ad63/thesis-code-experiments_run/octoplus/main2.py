import logging
import time
from datetime import datetime
import hydra
import mlflow
import psutil
import torch
from datetime import datetime
import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from octoplus.src.config_dataclasses.experiments import ExperimentConfig
from octoplus.src.env.mani_skill_env import get_envs
from octoplus.src.utils.git_utils import write_git_diff_to_file, write_git_id_to_file
from octoplus.src.rl_models.td3_cust_cleanrl import run_td3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
terminal_logger = logging.getLogger(__name__)


def timelogging(func):
    def wrapper(*args, **kwargs):
        # log the time and cpu time
        start_time = time.time()
        start_cpu_time = psutil.cpu_times()

        result = func(*args, **kwargs)

        # execution time
        end_time = time.time()
        end_cpu_time = psutil.cpu_times()
        execution_time = end_time - start_time
        cpu_time = end_cpu_time.user - start_cpu_time.user
        hours, rem = divmod(execution_time, 3600)
        minutes, seconds = divmod(rem, 60)

        terminal_logger.info(
            f"Execution time: {int(hours)} hours {int(minutes)} min {int(seconds)} sec"
        )
        cpu_hours, cpu_rem = divmod(cpu_time, 3600)
        cpu_minutes, cpu_seconds = divmod(cpu_rem, 60)

        terminal_logger.info(
            f"CPU time: {int(cpu_hours)} hours {int(cpu_minutes)} min {int(cpu_seconds)} sec"
        )
        return result

    return wrapper


def log_git_info(hydra_output_dir: str):
    ############################# Logging Git info for experiment reproducibility ##########
    git_id_file_path = f"{hydra_output_dir}/git_commit_id.txt"
    # Write the Git commit ID to the file
    write_git_id_to_file(git_id_file_path)
    git_diff_file_path = f"{hydra_output_dir}/git_diff.txt"
    # Write the Git diff to the file
    write_git_diff_to_file(git_diff_file_path)


@timelogging
def run(cfg: ExperimentConfig):
    hydra_output_dir = HydraConfig.get().run.dir

    datetime_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{cfg.model_config.exp_name}_{cfg.env_config.env_id}_{cfg.model_config.total_timesteps}_{datetime_id}"

    if mlflow.active_run() is not None:
        mlflow.end_run()  # kill the run if previous exists

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg.__to_dict__())
        run_td3(cfg, hydra_output_dir)
    log_git_info(hydra_output_dir)


#  CUDA_VISIBLE_DEVICES=0 python3 /home/piscenco/thesis_octo/thesis-code/octoplus/main2.py +experiment=td3_basic
cs = ConfigStore.instance()
cs.store(name="main_config", node=ExperimentConfig)

# python octoplus/main2.py
@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
    # default_experiment_config
    # ppo_config
    # ppo_different_tasks
)
def main(cfg: ExperimentConfig):

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = ExperimentConfig(**cfg)

    terminal_logger.info(f"Starting experiment with config: {cfg}")
    terminal_logger.info(f"Task name {cfg.env_config.env_id}")

    run(cfg)


# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()
