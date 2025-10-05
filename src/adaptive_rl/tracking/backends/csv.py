"""CSV logging backend."""

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CSVConfig
from .base import TrackerBackend


class CSVBackend(TrackerBackend):
    """CSV logging backend for data export."""

    def __init__(self, config: CSVConfig):
        """Initialize CSV backend."""
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / config.filename
        self.metrics_buffer = []
        self.params_logged = False
        self.fieldnames = set(["step"])

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to CSV buffer."""
        # Convert values to scalars
        row = {"step": step}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray) or hasattr(value, "item"):
                value = float(value.item())
            row[key] = value
            self.fieldnames.add(key)

        self.metrics_buffer.append(row)

        # Save periodically
        if len(self.metrics_buffer) >= self.config.save_frequency:
            self._save_metrics()

    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters to separate CSV file."""
        if self.params_logged:
            return

        params_path = self.log_dir / "hyperparameters.csv"
        with open(params_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["parameter", "value"])
            writer.writeheader()
            for key, value in params.items():
                writer.writerow({"parameter": key, "value": value})

        self.params_logged = True

    def close(self):
        """Save remaining metrics and close."""
        if self.metrics_buffer:
            self._save_metrics()

    def _save_metrics(self):
        """Save buffered metrics to CSV file."""
        if not self.metrics_buffer:
            return

        # Check if file exists
        file_exists = self.csv_path.exists()

        # Sort fieldnames for consistent column ordering
        fieldnames_list = ["step"] + sorted([f for f in self.fieldnames if f != "step"])

        # Write to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_list)

            # Write header if new file
            if not file_exists:
                writer.writeheader()

            # Write rows
            for row in self.metrics_buffer:
                # Fill missing fields with None
                for field in fieldnames_list:
                    if field not in row:
                        row[field] = None
                writer.writerow(row)

        # Clear buffer
        self.metrics_buffer = []
