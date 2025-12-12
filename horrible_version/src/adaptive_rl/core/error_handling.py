"""Comprehensive error handling and recovery for adaptive-rl.

Provides graceful failure modes and detailed error reporting.
"""

import functools
import traceback
from typing import Any, Callable, Dict, Optional, Type, Union
from pathlib import Path
import torch
import numpy as np

from loguru import logger
from rich.console import Console
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)
console = Console()


class AdaptiveRLError(Exception):
    """Base exception for adaptive-rl."""
    pass


class ConfigurationError(AdaptiveRLError):
    """Configuration validation errors."""
    pass


class ModelError(AdaptiveRLError):
    """Model loading/saving errors."""
    pass


class EnvironmentError(AdaptiveRLError):
    """Environment setup errors."""
    pass


class TrainingError(AdaptiveRLError):
    """Training loop errors."""
    pass


class EvaluationError(AdaptiveRLError):
    """Evaluation errors."""
    pass


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_msg: str = "Operation failed",
    reraise: bool = False,
    **kwargs
) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_msg}: {e}")
        if reraise:
            raise
        return default_return


def validate_tensors(*tensors: torch.Tensor, names: Optional[list] = None) -> None:
    """Validate tensor properties to catch issues early."""
    names = names or [f"tensor_{i}" for i in range(len(tensors))]

    for i, (tensor, name) in enumerate(zip(tensors, names)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} is not a tensor: {type(tensor)}")

        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")

        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains infinite values")

        if tensor.numel() == 0:
            raise ValueError(f"{name} is empty")


def validate_numpy_arrays(*arrays: np.ndarray, names: Optional[list] = None) -> None:
    """Validate numpy array properties."""
    names = names or [f"array_{i}" for i in range(len(arrays))]

    for array, name in zip(arrays, names):
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{name} is not a numpy array: {type(array)}")

        if np.isnan(array).any():
            raise ValueError(f"{name} contains NaN values")

        if np.isinf(array).any():
            raise ValueError(f"{name} contains infinite values")

        if array.size == 0:
            raise ValueError(f"{name} is empty")


def handle_training_errors(func: Callable) -> Callable:
    """Decorator for training loop error handling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            logger.error("CUDA out of memory during training")
            logger.error("Try reducing batch_size, n_envs, or network size")
            torch.cuda.empty_cache()
            raise TrainingError(f"CUDA OOM: {e}") from e
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA runtime error: {e}")
                torch.cuda.empty_cache()
                raise TrainingError(f"CUDA error: {e}") from e
            else:
                logger.error(f"Runtime error during training: {e}")
                raise TrainingError(f"Training runtime error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during training: {e}")
            logger.error(traceback.format_exc())
            raise TrainingError(f"Training failed: {e}") from e

    return wrapper


def handle_model_errors(func: Callable) -> Callable:
    """Decorator for model operation error handling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise ModelError(f"Model file missing: {e}") from e
        except torch.serialization.pickle.UnpicklingError as e:
            logger.error(f"Failed to load model checkpoint: {e}")
            raise ModelError(f"Corrupted checkpoint: {e}") from e
        except RuntimeError as e:
            if "size mismatch" in str(e):
                logger.error(f"Model architecture mismatch: {e}")
                raise ModelError(f"Architecture mismatch: {e}") from e
            else:
                logger.error(f"Model runtime error: {e}")
                raise ModelError(f"Model error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected model error: {e}")
            raise ModelError(f"Model operation failed: {e}") from e

    return wrapper


def handle_environment_errors(func: Callable) -> Callable:
    """Decorator for environment operation error handling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            logger.error(f"Environment dependency missing: {e}")
            raise EnvironmentError(f"Missing dependency: {e}") from e
        except Exception as e:
            if "environment" in str(e).lower():
                logger.error(f"Environment error: {e}")
                raise EnvironmentError(f"Environment setup failed: {e}") from e
            else:
                logger.error(f"Unexpected environment error: {e}")
                raise EnvironmentError(f"Environment error: {e}") from e

    return wrapper


class ErrorRecovery:
    """Error recovery utilities."""

    @staticmethod
    def recover_from_nan_gradients(model: torch.nn.Module) -> bool:
        """Attempt to recover from NaN gradients."""
        logger.warning("Attempting to recover from NaN gradients")

        # Zero out NaN gradients
        nan_params = 0
        total_params = 0

        for param in model.parameters():
            if param.grad is not None:
                total_params += 1
                if torch.isnan(param.grad).any():
                    nan_params += 1
                    param.grad = torch.zeros_like(param.grad)

        if nan_params > 0:
            logger.warning(f"Zeroed {nan_params}/{total_params} parameters with NaN gradients")
            return True

        return False

    @staticmethod
    def reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
        """Reset optimizer state to recover from corruption."""
        logger.warning("Resetting optimizer state")
        optimizer.state.clear()

    @staticmethod
    def save_emergency_checkpoint(model: torch.nn.Module, path: Path) -> None:
        """Save emergency checkpoint for debugging."""
        try:
            emergency_path = path.parent / f"emergency_{path.name}"
            torch.save(model.state_dict(), emergency_path)
            logger.info(f"Emergency checkpoint saved: {emergency_path}")
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")


class PerformanceMonitor:
    """Monitor performance and detect issues."""

    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.warning_threshold = max_memory_gb * 0.8

    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage."""
        import psutil

        # System memory
        system_mem = psutil.virtual_memory()
        system_usage_gb = system_mem.used / (1024**3)

        # GPU memory if available
        gpu_usage_gb = 0.0
        if torch.cuda.is_available():
            gpu_usage_gb = torch.cuda.memory_allocated() / (1024**3)

        usage = {
            "system_memory_gb": system_usage_gb,
            "gpu_memory_gb": gpu_usage_gb,
        }

        # Warning thresholds
        if system_usage_gb > self.warning_threshold:
            logger.warning(f"High system memory usage: {system_usage_gb:.1f} GB")

        if gpu_usage_gb > self.warning_threshold:
            logger.warning(f"High GPU memory usage: {gpu_usage_gb:.1f} GB")

        return usage

    def check_training_health(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check if training is healthy."""
        issues = {}

        # Check for NaN/inf values
        for key, value in metrics.items():
            if not np.isfinite(value):
                issues[f"non_finite_{key}"] = True
                logger.error(f"Non-finite value in {key}: {value}")

        # Check for extreme gradients
        if "grad_norm" in metrics:
            if metrics["grad_norm"] > 100:
                issues["large_gradients"] = True
                logger.warning(f"Large gradient norm: {metrics['grad_norm']}")

        # Check for training stagnation
        if "loss" in metrics:
            if metrics["loss"] < 1e-8:
                issues["loss_too_small"] = True
                logger.warning("Loss suspiciously small - potential numerical issues")

        return issues


class ConfigValidator:
    """Validate configurations with helpful error messages."""

    @staticmethod
    def validate_paths(config: dict) -> None:
        """Validate all paths in config."""
        if "paths" in config:
            paths_config = config["paths"]

            # Check if log directory is writable
            log_dir = Path(paths_config.get("log_dir", "logs"))
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                test_file = log_dir / "test_write.tmp"
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                raise ConfigurationError(f"Log directory not writable: {log_dir}. Error: {e}")

    @staticmethod
    def validate_device_compatibility(config: dict) -> None:
        """Validate device configuration."""
        device = config.get("experiment", {}).get("device", "cpu")

        if device == "cuda":
            if not torch.cuda.is_available():
                raise ConfigurationError("CUDA device requested but not available")

            # Check if there's enough GPU memory
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory_gb < 2.0:
                    logger.warning(f"Low GPU memory: {gpu_memory_gb:.1f} GB")

    @staticmethod
    def validate_hyperparameters(config: dict) -> None:
        """Validate hyperparameter ranges."""
        algo_config = config.get("algorithm", {})

        # Check reasonable ranges
        lr = algo_config.get("learning_rate", 3e-4)
        if lr > 1.0 or lr < 1e-6:
            logger.warning(f"Unusual learning rate: {lr}")

        batch_size = algo_config.get("batch_size", 64)
        n_envs = config.get("environment", {}).get("num_envs", 4)

        if batch_size > n_envs * 1000:
            logger.warning(f"Very large batch size ({batch_size}) relative to num_envs ({n_envs})")


def setup_error_handling() -> None:
    """Set up global error handling."""
    import warnings

    # Filter common warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Set torch error handling
    torch.autograd.set_detect_anomaly(False)  # Disable for performance

    logger.info("Error handling configured")


def create_error_report(error: Exception, context: Dict[str, Any]) -> str:
    """Create detailed error report for debugging."""
    report = [
        "=" * 60,
        "ADAPTIVE-RL ERROR REPORT",
        "=" * 60,
        f"Error Type: {type(error).__name__}",
        f"Error Message: {str(error)}",
        "",
        "Context:",
    ]

    for key, value in context.items():
        report.append(f"  {key}: {value}")

    report.extend([
        "",
        "Traceback:",
        traceback.format_exc(),
        "=" * 60,
    ])

    return "\n".join(report)