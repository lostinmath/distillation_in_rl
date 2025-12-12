import random
import subprocess
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from adaptive_rl.pipelines import PipelineBuilder

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = str(PROJECT_ROOT / "configs")


def get_git_info():
    """Get git commit information for reproducibility tracking."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"commit": commit, "branch": branch}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "branch": "unknown"}


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)

    if torch.cuda.is_available() and cfg.experiment.device == "cuda":
        torch.cuda.manual_seed(cfg.experiment.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    pipeline = PipelineBuilder.build_pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
