# from octoplus.src.datasets.manyskills2_dataset import ManiSkill2Dataset, ManiSkill2DatasetImages
# from hydra.core.config_store import ConfigStore
import logging
import time
from datetime import datetime
import hydra
import mlflow
import psutil
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from octoplus.src.config_dataclasses.experiments import ExperimentConfig
from octoplus.src.env.mani_skill_env import get_envs
from octoplus.src.rl_models.shared_model_api import get_model
from octoplus.src.utils.git_utils import write_git_diff_to_file, write_git_id_to_file
######################### Logging ##################################################
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

############################ Const ################################################
USE_DEBUG_MODE = False
SKIP_GIT = False
# Set the preferred linear algebra backend
torch.backends.cuda.preferred_linalg_library("magma")
############################## Time logging ############################################
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

        logger.info(
            f"Execution time: {int(hours)} hours {int(minutes)} min {int(seconds)} sec"
        )
        cpu_hours, cpu_rem = divmod(cpu_time, 3600)
        cpu_minutes, cpu_seconds = divmod(cpu_rem, 60)

        logger.info(
            f"CPU time: {int(cpu_hours)} hours {int(cpu_minutes)} min {int(cpu_seconds)} sec"
        )

        return result
    return wrapper


###################################################################################
@timelogging
def run(cfg: ExperimentConfig, debug_mode=USE_DEBUG_MODE):
    # clean_cuda_memory()

    # Get the Hydra log directory
    hydra_output_dir = HydraConfig.get().run.dir
    env = get_envs(
        env_config=cfg.env_config,
        # hydra_log_dir=hydra_output_dir,
    )

    ############################# Pipline run ############################################

    # get the model
    model = get_model(cfg=cfg, env=env, hydra_log_dir=hydra_output_dir)

    if mlflow.active_run() is not None:
        mlflow.end_run()  # kill the run if previous exists

    # Create a concatenated datetime ID
    datetime_id = datetime.now().strftime("%Y%m%d%H%M%S")
    # the name of the run in mlflow
    run_name = (
        f"{cfg.model_config.exp_name}_{cfg.env_config.env_id}_{cfg.model_config.total_timesteps}_{datetime_id}"
    )

    if debug_mode: # just run without mlflow
        logger.warning("Running in debug mode without MLFlow.")
        # do initialization
        model.model_setup()
        # training
        model.start_training()
        # some cleaning of memory at the end if needed
        model.model_post_setup()
        return
        
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(cfg.__to_dict__())  # logging hyperparams
            # do initialization
            model.model_setup()
            # training
            model.start_training()
            # some cleaning of memory at the end if needed
            model.model_post_setup()
            # TODO: eval metrics
    except Exception as e:
        logger.error(f"An error occurred during the MLflow run: {e}")

    if not SKIP_GIT:
        log_git_info(hydra_output_dir)
    

############################# Logging Git info for experiment reproducibility ##########


def log_git_info(hydra_output_dir: str):
    ############################# Logging Git info for experiment reproducibility ##########
    git_id_file_path = f"{hydra_output_dir}/git_commit_id.txt"
    # Write the Git commit ID to the file
    write_git_id_to_file(git_id_file_path)
    git_diff_file_path = f"{hydra_output_dir}/git_diff.txt"
    # Write the Git diff to the file
    write_git_diff_to_file(git_diff_file_path)
    

################################# Main ################################################
# Registering configs
cs = ConfigStore.instance()
cs.store(name="main_config", node=ExperimentConfig)


# TODO:move functions into separate files
# change param logging to metrics logging
# add logically resonable param loggining
# combine runs

# lauching mlflow server
# mlflow server --host 127.0.0.1 --port 8095

# CUDA_VISIBLE_DEVICES=0 python3 /home/piscenco/thesis_octo/thesis-code/octoplus/main.py +experiment=debug
@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="ppo_config"
    # default_experiment_config
    # ppo_config
    # ppo_different_tasks
)
def main(cfg: ExperimentConfig):

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = ExperimentConfig(**cfg)
    logger.info(f"Starting experiment with config: {cfg}")
    logger.info(f"Task name {cfg.env_config.env_id}")
    run(cfg)
    return 0
    


# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()
