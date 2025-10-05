"""TensorBoard tracking backend."""

from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ..config import TensorBoardConfig
from .base import TrackerBackend


class TensorBoardBackend(TrackerBackend):
    """TensorBoard tracking backend."""

    def __init__(self, config: TensorBoardConfig):
        """Initialize TensorBoard backend."""
        self.config = config
        self.writer = SummaryWriter(
            log_dir=config.log_dir, flush_secs=config.flush_secs
        )

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log scalar metrics to TensorBoard."""
        for key, value in metrics.items():
            # Convert to scalar if needed
            if isinstance(value, np.ndarray) or hasattr(value, "item"):
                value = float(value.item())
            self.writer.add_scalar(key, value, step)

    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters to TensorBoard."""
        # TensorBoard doesn't have native hyperparam support like other tools
        # Log as text in a markdown table
        param_text = "| Parameter | Value |\n|-----------|-------|\n"
        for key, value in params.items():
            param_text += f"| {key} | {value} |\n"
        self.writer.add_text("hyperparameters", param_text)

    def log_video(self, video: np.ndarray, name: str, step: int, fps: int = 30):
        """Log video to TensorBoard."""
        # Expected shape: (T, C, H, W) or (B, T, C, H, W)
        if video.ndim == 4:
            video = video.unsqueeze(0)
        self.writer.add_video(name, video, step, fps=fps)

    def log_figure(self, figure: Any, name: str, step: int):
        """Log matplotlib figure to TensorBoard."""
        self.writer.add_figure(name, figure, step)

    def log_histogram(self, values: np.ndarray, name: str, step: int):
        """Log histogram to TensorBoard."""
        self.writer.add_histogram(name, values, step)

    def log_text(self, text: str, name: str, step: int):
        """Log text to TensorBoard."""
        self.writer.add_text(name, text, step)

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
