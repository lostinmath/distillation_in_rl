"""MLFlow tracking backend."""

from typing import Any

import numpy as np

from ..config import MLFlowConfig
from .base import TrackerBackend


class MLFlowBackend(TrackerBackend):
    """MLFlow tracking backend."""

    def __init__(self, config: MLFlowConfig):
        """Initialize MLFlow backend."""
        self.config = config

        try:
            import mlflow

            self.mlflow = mlflow
        except ImportError:
            raise ImportError("MLFlow not installed. Install with: pip install mlflow")

        # Set tracking URI if provided
        if config.tracking_uri:
            self.mlflow.set_tracking_uri(config.tracking_uri)

        # Set experiment
        if config.experiment_name:
            self.mlflow.set_experiment(config.experiment_name)

        # Start run
        self.run = self.mlflow.start_run(run_name=config.run_name, tags=config.tags)

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to MLFlow."""
        # Convert to scalars
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray) or hasattr(value, "item"):
                value = float(value.item())
            clean_metrics[key] = value

        self.mlflow.log_metrics(clean_metrics, step=step)

    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters to MLFlow."""
        # MLFlow requires string values for params
        for key, value in params.items():
            self.mlflow.log_param(key, str(value))

    def log_model(self, model_path: str, name: str, metadata: dict | None = None):
        """Log model to MLFlow."""
        self.mlflow.log_artifact(model_path, artifact_path=f"models/{name}")
        if metadata:
            for key, value in metadata.items():
                self.mlflow.set_tag(f"model_{name}_{key}", str(value))

    def log_figure(self, figure: Any, name: str, step: int):
        """Log matplotlib figure to MLFlow."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            figure.savefig(tmp.name)
            self.mlflow.log_artifact(tmp.name, artifact_path=f"figures/{name}_{step}")

    def log_text(self, text: str, name: str, step: int):
        """Log text to MLFlow."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write(text)
            tmp.flush()
            self.mlflow.log_artifact(tmp.name, artifact_path=f"text/{name}_{step}")

    def close(self):
        """End MLFlow run."""
        self.mlflow.end_run()
