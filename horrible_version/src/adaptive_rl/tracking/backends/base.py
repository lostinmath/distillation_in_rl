"""Base class for tracking backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TrackerBackend(ABC):
    """Abstract base class for all tracking backends."""

    @abstractmethod
    def __init__(self, config: Any):
        """Initialize the backend with configuration."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log scalar metrics."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters."""

    def log_video(self, video: np.ndarray, name: str, step: int, fps: int = 30):
        """Log video data (optional for backends)."""

    def log_figure(self, figure: Any, name: str, step: int):
        """Log matplotlib figure (optional for backends)."""

    def log_model(self, model_path: str, name: str, metadata: dict | None = None):
        """Log model checkpoint (optional for backends)."""

    def log_text(self, text: str, name: str, step: int):
        """Log text data (optional for backends)."""

    def log_histogram(self, values: np.ndarray, name: str, step: int):
        """Log histogram data (optional for backends)."""

    def log_table(self, data: dict[str, list], name: str, step: int):
        """Log tabular data (optional for backends)."""

    @abstractmethod
    def close(self):
        """Clean up and close the backend."""

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
