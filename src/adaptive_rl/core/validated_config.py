"""Pydantic V2 config validation for adaptive-rl.

Provides type-safe, validated configuration with clear error messages.
Replaces fragile YAML parsing with robust schema validation.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pathlib import Path
import torch

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ConfigDict


class NetworkConfig(BaseModel):
    """Neural network configuration."""
    model_config = ConfigDict(extra="forbid")

    _target_: str = "adaptive_rl.networks.mlp.MLPNetwork"
    hidden_sizes: List[int] = Field(default=[64, 64], min_length=1)
    activation: Literal["tanh", "relu", "gelu", "swish"] = "tanh"

    @field_validator("hidden_sizes")
    @classmethod
    def validate_hidden_sizes(cls, v: List[int]) -> List[int]:
        if any(size <= 0 for size in v):
            raise ValueError("All hidden sizes must be positive")
        if any(size > 4096 for size in v):
            raise ValueError("Hidden sizes should be reasonable (<=4096)")
        return v


class AlgorithmConfig(BaseModel):
    """PPO algorithm configuration with validation."""
    model_config = ConfigDict(extra="forbid")

    name: Literal["ppo"] = "ppo"
    _target_: str = "adaptive_rl.algorithms.ppo.PPO"

    # Learning parameters
    learning_rate: float = Field(default=3e-4, gt=0, le=1.0)
    n_steps: int = Field(default=2048, ge=64, le=32768)
    batch_size: int = Field(default=64, ge=8, le=8192)
    n_epochs: int = Field(default=10, ge=1, le=100)

    # PPO-specific parameters
    gamma: float = Field(default=0.99, gt=0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0, le=1.0)
    clip_range: float = Field(default=0.2, gt=0, le=1.0)
    clip_range_vf: Optional[float] = Field(default=None, gt=0, le=1.0)
    normalize_advantage: bool = True
    ent_coef: float = Field(default=0.0, ge=0, le=1.0)
    vf_coef: float = Field(default=0.5, ge=0, le=10.0)
    max_grad_norm: float = Field(default=0.5, gt=0, le=100.0)

    # Network configuration
    network: Optional[NetworkConfig] = Field(default_factory=NetworkConfig)

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int, info) -> int:
        # Ensure batch_size divides evenly into n_steps
        if hasattr(info.data, 'n_steps') and info.data.get('n_steps'):
            n_steps = info.data['n_steps']
            if n_steps % v != 0:
                raise ValueError(f"batch_size ({v}) must divide n_steps ({n_steps}) evenly")
        return v


class EnvironmentConfig(BaseModel):
    """Environment configuration."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    env_id: str = Field(..., min_length=1)
    num_envs: int = Field(default=4, ge=1, le=256)
    max_episode_steps: Optional[int] = Field(default=None, gt=0)
    normalize_obs: bool = False
    normalize_reward: bool = False

    @field_validator("env_id")
    @classmethod
    def validate_env_id(cls, v: str) -> str:
        # Basic validation for common environments
        known_envs = {
            "CartPole-v1", "LunarLander-v2", "Acrobot-v1",
            "MountainCar-v0", "Pendulum-v1"
        }
        if v not in known_envs:
            # Just warn, don't fail - might be custom env
            pass
        return v


class SchedulerConfig(BaseModel):
    """Scheduler configuration with strategy-specific validation."""
    model_config = ConfigDict(extra="forbid")

    name: Literal[
        "student_only", "teacher_only", "epsilon", "epsilon_decreasing",
        "interchangeably", "teacher_then_student", "reward_based"
    ]
    _target_: str

    # Common scheduler parameters
    num_envs: Optional[int] = None  # Filled automatically

    # Strategy-specific parameters (validated based on name)
    epsilon: Optional[float] = Field(default=None, ge=0, le=1.0)
    initial_epsilon: Optional[float] = Field(default=None, ge=0, le=1.0)
    final_epsilon: Optional[float] = Field(default=None, ge=0, le=1.0)
    epsilon_decay_steps: Optional[int] = Field(default=None, gt=0)

    # Reward-based specific
    trust_period: Optional[int] = Field(default=5, ge=1, le=1000)
    policy_trust_threshold: Optional[float] = Field(default=0.6, ge=0, le=1.0)
    internal_policy_warmup_length: Optional[int] = Field(default=5, ge=0, le=1000)
    initial_policy: Optional[Literal["teacher", "student"]] = "teacher"

    # Teacher-then-student specific
    teacher_steps: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_strategy_params(self) -> "SchedulerConfig":
        """Validate parameters based on strategy name."""
        name = self.name

        if name in ["epsilon", "epsilon_decreasing"]:
            if self.epsilon is None:
                raise ValueError(f"epsilon required for {name} scheduler")

        if name == "epsilon_decreasing":
            if self.epsilon_decay_steps is None:
                raise ValueError("epsilon_decay_steps required for epsilon_decreasing")

        if name == "teacher_then_student":
            if self.teacher_steps is None:
                raise ValueError("teacher_steps required for teacher_then_student")

        if name == "reward_based":
            if self.trust_period is None:
                self.trust_period = 5

        return self


class TeacherConfig(BaseModel):
    """Teacher policy configuration."""
    model_config = ConfigDict(extra="forbid")

    name: Literal["optimal", "random", "pretrained"]
    _target_: str

    # Pretrained teacher specific
    model_path: Optional[Path] = None

    # Teacher quality/noise parameters
    action_noise: float = Field(default=0.0, ge=0, le=1.0)
    success_rate: float = Field(default=1.0, gt=0, le=1.0)

    @model_validator(mode="after")
    def validate_teacher_params(self) -> "TeacherConfig":
        """Validate teacher-specific parameters."""
        if self.name == "pretrained":
            if self.model_path is None:
                raise ValueError("model_path required for pretrained teacher")
            if not self.model_path.exists():
                raise ValueError(f"Teacher model not found: {self.model_path}")

        return self


class TrackerConfig(BaseModel):
    """Experiment tracking configuration."""
    model_config = ConfigDict(extra="forbid")

    backends: List[Literal["tensorboard", "wandb", "mlflow", "console", "csv"]] = ["console"]

    # Backend-specific configs
    tensorboard: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"enabled": True})
    wandb: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"enabled": False})
    mlflow: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"enabled": False})
    console: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"enabled": True, "verbose": True})
    csv: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"enabled": True})

    @field_validator("backends")
    @classmethod
    def validate_backends(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one tracking backend required")
        return v


class PathsConfig(BaseModel):
    """Path configuration with validation."""
    model_config = ConfigDict(extra="forbid")

    log_dir: Path
    checkpoint_dir: Path
    video_dir: Optional[Path] = None

    @model_validator(mode="after")
    def create_directories(self) -> "PathsConfig":
        """Ensure directories exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.video_dir:
            self.video_dir.mkdir(parents=True, exist_ok=True)
        return self


class ExperimentConfig(BaseModel):
    """Experiment metadata and settings."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100)
    seed: int = Field(default=42, ge=0, le=2**32-1)
    device: Literal["cpu", "cuda", "auto"] = "auto"
    num_seeds: int = Field(default=3, ge=1, le=50)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if v == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA not available but cuda device requested")
        return v


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    model_config = ConfigDict(extra="forbid")

    total_timesteps: int = Field(..., gt=0, le=100_000_000)
    eval_freq: int = Field(default=10000, gt=0)
    checkpoint_freq: int = Field(default=50000, gt=0)
    video_freq: int = Field(default=0, ge=0)

    @field_validator("eval_freq", "checkpoint_freq")
    @classmethod
    def validate_freq(cls, v: int, info) -> int:
        if hasattr(info.data, 'total_timesteps') and info.data.get('total_timesteps'):
            total = info.data['total_timesteps']
            if v > total:
                raise ValueError(f"Frequency ({v}) cannot exceed total_timesteps ({total})")
        return v


class AdaptiveRLConfig(BaseModel):
    """Complete adaptive-rl configuration with full validation."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    experiment: ExperimentConfig
    algorithm: AlgorithmConfig
    environment: EnvironmentConfig
    scheduler: SchedulerConfig
    teacher: Optional[TeacherConfig] = None
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    training: TrainingConfig
    paths: PathsConfig

    @model_validator(mode="after")
    def validate_cross_config_consistency(self) -> "AdaptiveRLConfig":
        """Validate consistency across config sections."""

        # Validate teacher requirement
        if self.scheduler.name in ["teacher_only", "epsilon", "epsilon_decreasing",
                                  "reward_based", "teacher_then_student", "interchangeably"]:
            if self.teacher is None:
                raise ValueError(f"Teacher required for {self.scheduler.name} scheduler")

        # Validate environment-algorithm compatibility
        if self.environment.num_envs < self.algorithm.batch_size:
            raise ValueError(
                f"num_envs ({self.environment.num_envs}) should be >= batch_size ({self.algorithm.batch_size})"
            )

        # Set scheduler num_envs
        self.scheduler.num_envs = self.environment.num_envs

        # Validate paths consistency
        if not str(self.paths.checkpoint_dir).startswith(str(self.paths.log_dir)):
            self.paths.checkpoint_dir = self.paths.log_dir / "checkpoints"

        return self


def validate_config_from_dict(config_dict: Dict[str, Any]) -> AdaptiveRLConfig:
    """Validate and parse configuration dictionary."""
    try:
        return AdaptiveRLConfig.model_validate(config_dict)
    except Exception as e:
        from loguru import logger
        logger.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e


def validate_config_from_omegaconf(omega_config) -> AdaptiveRLConfig:
    """Validate OmegaConf configuration."""
    from omegaconf import OmegaConf

    config_dict = OmegaConf.to_container(omega_config, resolve=True)
    return validate_config_from_dict(config_dict)