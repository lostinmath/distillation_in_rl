"""Neptune.ai tracking backend."""

from typing import Any

import numpy as np

from ..config import NeptuneConfig
from .base import TrackerBackend


class NeptuneBackend(TrackerBackend):
    """Neptune.ai tracking backend."""

    def __init__(self, config: NeptuneConfig):
        """Initialize Neptune backend."""
        self.config = config

        try:
            import neptune

            self.neptune = neptune
        except ImportError:
            raise ImportError(
                "neptune not installed. Install with: pip install neptune"
            )

        # Initialize Neptune run
        self.run = self.neptune.init_run(
            project=config.project,
            api_token=config.api_token,
            name=config.name,
            tags=config.tags,
            source_files=config.source_files,
        )

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to Neptune."""
        for key, value in metrics.items():
            # Convert to scalar
            if isinstance(value, np.ndarray) or hasattr(value, "item"):
                value = float(value.item())

            self.run[f"metrics/{key}"].append(value, step=step)

    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters to Neptune."""
        self.run["parameters"] = params

    def log_video(self, video: np.ndarray, name: str, step: int, fps: int = 30):
        """Log video to Neptune."""
        import tempfile

        import cv2

        # Save video to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            height, width = video.shape[1:3]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))

            for frame in video:
                # Convert to BGR if needed
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            out.release()

            self.run[f"videos/{name}"].upload(tmp.name)

    def log_figure(self, figure: Any, name: str, step: int):
        """Log matplotlib figure to Neptune."""
        self.run[f"figures/{name}"].upload(figure)

    def log_model(self, model_path: str, name: str, metadata: dict | None = None):
        """Log model to Neptune."""
        self.run[f"models/{name}"].upload(model_path)
        if metadata:
            self.run[f"models/{name}/metadata"] = metadata

    def log_histogram(self, values: np.ndarray, name: str, step: int):
        """Log histogram to Neptune."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(values.flatten(), bins=50)
        self.run[f"histograms/{name}"].upload(fig)
        plt.close(fig)

    def log_text(self, text: str, name: str, step: int):
        """Log text to Neptune."""
        self.run[f"text/{name}"].append(text)

    def log_table(self, data: dict[str, list], name: str, step: int):
        """Log table to Neptune."""
        import pandas as pd

        df = pd.DataFrame(data)
        self.run[f"tables/{name}"].upload(df)

    def close(self):
        """Stop Neptune run."""
        self.run.stop()
