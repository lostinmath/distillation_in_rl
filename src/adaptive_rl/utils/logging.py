"""Logging utilities for tracking training metrics.

Supports logging to:
- Terminal output
- TensorBoard
- MLFlow
- CSV files
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional imports
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class Logger:
    """Unified logger for training metrics.

    Handles logging to multiple backends simultaneously.
    """

    def __init__(
        self,
        run_name: str,
        log_dir: str = "logs",
        use_mlflow: bool = False,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        silence: bool = False,
        log_file: str = "training_metrics.txt",
        wandb_config: dict = None,
    ) -> None:
        """Initialize logger.

        Args:
            run_name: Name of the current run
            log_dir: Directory for logs
            use_mlflow: Whether to use MLFlow logging
            use_tensorboard: Whether to use TensorBoard logging
            silence: Silence terminal output
            log_file: File for text logging
        """
        # Setup terminal logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.terminal_logger = logging.getLogger(__name__)
        self.silence = silence

        # Create log directory
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup log file
        self.log_file = self.log_dir / log_file

        # MLFlow setup
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        if self.use_mlflow:
            try:
                mlflow.set_experiment(run_name)
                mlflow.start_run()
            except Exception as e:
                self.terminal_logger.warning(f"Failed to initialize MLFlow: {e}")
                self.use_mlflow = False
        elif use_mlflow and not MLFLOW_AVAILABLE:
            self.terminal_logger.warning(
                "MLFlow not installed. MLFlow logging disabled."
            )

        # TensorBoard setup
        self.tensorboard_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                tb_dir = self.log_dir / "tensorboard"
                tb_dir.mkdir(exist_ok=True)
                self.tensorboard_writer = SummaryWriter(str(tb_dir))
            except Exception as e:
                self.terminal_logger.warning(f"Failed to initialize TensorBoard: {e}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            self.terminal_logger.warning(
                "TensorBoard not installed. TensorBoard logging disabled."
            )

        # W&B setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            try:
                wandb_config = wandb_config or {}
                wandb.init(
                    name=run_name,
                    config=wandb_config,
                    reinit=True
                )
            except Exception as e:
                self.terminal_logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False
        elif use_wandb and not WANDB_AVAILABLE:
            self.terminal_logger.warning(
                "W&B not installed. W&B logging disabled."
            )

    def log_param(self, key: str, value: Any):
        """Log a parameter (hyperparameter, configuration, etc.).

        Args:
            key: Parameter name
            value: Parameter value
        """
        if self.use_mlflow:
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                self.terminal_logger.debug(f"MLFlow log_param failed: {e}")

        if self.use_wandb:
            try:
                wandb.config[key] = value
            except Exception as e:
                self.terminal_logger.debug(f"W&B log_param failed: {e}")

        if not self.silence:
            self.terminal_logger.info(f"{key}: {value}")

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(key, str(value))

    def log_metrics(self, metric_name: str, metric_value: float, step: int = -1):
        """Log a metric value.

        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            step: Training step
        """
        # Convert to scalar if numpy array
        import numpy as np

        if isinstance(metric_value, np.ndarray) or hasattr(metric_value, "item"):
            metric_value = float(metric_value.item())
        else:
            metric_value = float(metric_value)

        if self.use_mlflow:
            try:
                mlflow.log_metrics({metric_name: metric_value}, step=step)
            except Exception as e:
                self.terminal_logger.debug(f"MLFlow log_metrics failed: {e}")

        if self.use_wandb:
            try:
                wandb.log({metric_name: metric_value}, step=step if step >= 0 else None)
            except Exception as e:
                self.terminal_logger.debug(f"W&B log_metrics failed: {e}")

        if not self.silence:
            self.terminal_logger.info(
                f"Step {step}: {metric_name} = {metric_value:.4f}"
            )

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(
                metric_name, metric_value, global_step=step
            )

    def log_metrics_dict(self, metrics: dict[str, float], step: int = -1):
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step
        """
        for name, value in metrics.items():
            self.log_metrics(name, value, step)

    def terminal_only_print(self, message: str):
        """Print message only to terminal.

        Args:
            message: Message to print
        """
        if not self.silence:
            self.terminal_logger.info(message)

    def log_to_file(self, message: str):
        """Log message to file.

        Args:
            message: Message to log
        """
        with open(self.log_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")

    def close(self):
        """Close all logging resources."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

        if self.use_mlflow:
            try:
                mlflow.end_run()
            except Exception:
                pass

        if self.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
