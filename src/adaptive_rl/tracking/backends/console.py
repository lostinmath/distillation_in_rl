"""Console logging backend."""

import logging
from typing import Any

import numpy as np

from ..config import ConsoleConfig
from .base import TrackerBackend


class ConsoleBackend(TrackerBackend):
    """Console logging backend for terminal output."""

    def __init__(self, config: ConsoleConfig):
        """Initialize console backend."""
        self.config = config
        self.step_counter = 0

        # Setup logger
        self.logger = logging.getLogger("adaptive_rl.tracking")
        self.logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)

        # Add console handler if not present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to console."""
        # Only log every log_frequency steps
        if step % self.config.log_frequency != 0:
            return

        # Format metrics for display
        metric_str = ", ".join(
            [f"{k}={self._format_value(v)}" for k, v in metrics.items()]
        )
        self.logger.info(f"Step {step}: {metric_str}")

    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters to console."""
        self.logger.info("Hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

    def log_text(self, text: str, name: str, step: int):
        """Log text to console."""
        self.logger.info(f"{name} (step {step}): {text}")

    def close(self):
        """Clean up console backend."""

    def _format_value(self, value: Any) -> str:
        """Format value for console display."""
        if isinstance(value, np.ndarray):
            value = value.item()
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
