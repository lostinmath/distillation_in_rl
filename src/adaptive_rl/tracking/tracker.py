"""Unified experiment tracker with pluggable backends."""

import logging
from typing import Any

import numpy as np

from .backends import (
    ConsoleBackend,
    CSVBackend,
    MLFlowBackend,
    NeptuneBackend,
    TensorBoardBackend,
    TrackerBackend,
    WandbBackend,
)
from .config import TrackerConfig


class ExperimentTracker:
    """Unified interface for experiment tracking across multiple backends."""

    BACKEND_REGISTRY = {
        "tensorboard": TensorBoardBackend,
        "mlflow": MLFlowBackend,
        "wandb": WandbBackend,
        "neptune": NeptuneBackend,
        "console": ConsoleBackend,
        "csv": CSVBackend,
    }

    def __init__(self, config: TrackerConfig, run_name: str | None = None):
        """Initialize experiment tracker with configured backends.

        Args:
            config: Tracker configuration
            run_name: Optional run name for the experiment
        """
        self.config = config
        self.run_name = run_name
        self.backends: list[TrackerBackend] = []
        self.logger = logging.getLogger("adaptive_rl.tracking")

        # Initialize enabled backends
        for backend_name in config.backends:
            if config.is_backend_enabled(backend_name):
                try:
                    backend_class = self.BACKEND_REGISTRY.get(backend_name)
                    if backend_class:
                        backend_config = getattr(config, backend_name, None)
                        if backend_config:
                            backend = backend_class(backend_config)
                            self.backends.append(backend)
                            self.logger.info(f"Initialized {backend_name} backend")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize {backend_name}: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log scalar metrics to all backends.

        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
        """
        for backend in self.backends:
            try:
                backend.log_metrics(metrics, step)
            except Exception as e:
                self.logger.debug(
                    f"Failed to log metrics to {backend.__class__.__name__}: {e}"
                )

    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters to all backends.

        Args:
            params: Dictionary of hyperparameter names and values
        """
        for backend in self.backends:
            try:
                backend.log_params(params)
            except Exception as e:
                self.logger.debug(
                    f"Failed to log params to {backend.__class__.__name__}: {e}"
                )

    def log_video(self, video: np.ndarray, name: str, step: int, fps: int = 30):
        """Log video to supporting backends.

        Args:
            video: Video array
            name: Name for the video
            step: Current training step
            fps: Frames per second
        """
        if not self.config.log_videos:
            return

        for backend in self.backends:
            if hasattr(backend, "log_video"):
                try:
                    backend.log_video(video, name, step, fps)
                except Exception as e:
                    self.logger.debug(
                        f"Failed to log video to {backend.__class__.__name__}: {e}"
                    )

    def log_figure(self, figure: Any, name: str, step: int):
        """Log matplotlib figure to supporting backends.

        Args:
            figure: Matplotlib figure
            name: Name for the figure
            step: Current training step
        """
        if not self.config.log_figures:
            return

        for backend in self.backends:
            if hasattr(backend, "log_figure"):
                try:
                    backend.log_figure(figure, name, step)
                except Exception as e:
                    self.logger.debug(
                        f"Failed to log figure to {backend.__class__.__name__}: {e}"
                    )

    def log_model(self, model_path: str, name: str, metadata: dict | None = None):
        """Log model checkpoint to supporting backends.

        Args:
            model_path: Path to model checkpoint
            name: Name for the model
            metadata: Optional metadata dictionary
        """
        if not self.config.log_models:
            return

        for backend in self.backends:
            if hasattr(backend, "log_model"):
                try:
                    backend.log_model(model_path, name, metadata)
                except Exception as e:
                    self.logger.debug(
                        f"Failed to log model to {backend.__class__.__name__}: {e}"
                    )

    def log_histogram(self, values: np.ndarray, name: str, step: int):
        """Log histogram to supporting backends.

        Args:
            values: Values for histogram
            name: Name for the histogram
            step: Current training step
        """
        for backend in self.backends:
            if hasattr(backend, "log_histogram"):
                try:
                    backend.log_histogram(values, name, step)
                except Exception as e:
                    self.logger.debug(
                        f"Failed to log histogram to {backend.__class__.__name__}: {e}"
                    )

    def log_text(self, text: str, name: str, step: int):
        """Log text to supporting backends.

        Args:
            text: Text to log
            name: Name for the text
            step: Current training step
        """
        for backend in self.backends:
            if hasattr(backend, "log_text"):
                try:
                    backend.log_text(text, name, step)
                except Exception as e:
                    self.logger.debug(
                        f"Failed to log text to {backend.__class__.__name__}: {e}"
                    )

    def log_table(self, data: dict[str, list], name: str, step: int):
        """Log tabular data to supporting backends.

        Args:
            data: Dictionary of column names and values
            name: Name for the table
            step: Current training step
        """
        for backend in self.backends:
            if hasattr(backend, "log_table"):
                try:
                    backend.log_table(data, name, step)
                except Exception as e:
                    self.logger.debug(
                        f"Failed to log table to {backend.__class__.__name__}: {e}"
                    )

    def close(self):
        """Close all backends and clean up resources."""
        for backend in self.backends:
            try:
                backend.close()
            except Exception as e:
                self.logger.debug(f"Failed to close {backend.__class__.__name__}: {e}")

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
