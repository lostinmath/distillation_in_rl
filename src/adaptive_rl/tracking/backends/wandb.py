"""Weights & Biases tracking backend."""

from typing import Any

import numpy as np

from ..config import WandbConfig
from .base import TrackerBackend


class WandbBackend(TrackerBackend):
    """Weights & Biases tracking backend."""

    def __init__(self, config: WandbConfig):
        """Initialize W&B backend."""
        self.config = config

        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb")

        # Initialize W&B run
        self.run = self.wandb.init(
            project=config.project,
            entity=config.entity,
            group=config.group,
            tags=config.tags,
            name=config.name,
            notes=config.notes,
            mode=config.mode,
            reinit=True,
        )

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to W&B."""
        # Convert to scalars
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray) or hasattr(value, "item"):
                value = float(value.item())
            clean_metrics[key] = value

        self.wandb.log(clean_metrics, step=step)

    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters to W&B."""
        self.wandb.config.update(params)

    def log_video(self, video: np.ndarray, name: str, step: int, fps: int = 30):
        """Log video to W&B."""
        # W&B expects shape (T, H, W, C)
        if video.shape[-1] not in [1, 3]:
            video = np.transpose(video, (0, 2, 3, 1))

        self.wandb.log({name: self.wandb.Video(video, fps=fps)}, step=step)

    def log_figure(self, figure: Any, name: str, step: int):
        """Log matplotlib figure to W&B."""
        self.wandb.log({name: self.wandb.Image(figure)}, step=step)

    def log_model(self, model_path: str, name: str, metadata: dict | None = None):
        """Log model to W&B."""
        artifact = self.wandb.Artifact(name, type="model", metadata=metadata)
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)

    def log_histogram(self, values: np.ndarray, name: str, step: int):
        """Log histogram to W&B."""
        self.wandb.log({name: self.wandb.Histogram(values)}, step=step)

    def log_table(self, data: dict[str, list], name: str, step: int):
        """Log table to W&B."""
        table = self.wandb.Table(
            columns=list(data.keys()), data=list(zip(*data.values(), strict=False))
        )
        self.wandb.log({name: table}, step=step)

    def log_text(self, text: str, name: str, step: int):
        """Log text to W&B."""
        self.wandb.log({name: self.wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def close(self):
        """Finish W&B run."""
        self.wandb.finish()
