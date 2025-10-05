"""Configuration dataclasses for distillation experiments."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackerConfig:
    """Experiment tracking configuration."""

    backends: list[str] = field(
        default_factory=lambda: ["tensorboard", "console", "csv"]
    )
    log_frequency: int = 10
    log_videos: bool = False
    log_models: bool = True


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    env_id: str = "CartPole-v1"
    num_envs: int = 8
    num_steps: int = 128


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    num_iterations: int = 500
    learning_rate: float = 2.5e-4
    batch_size: int = 256
    n_epochs: int = 4


@dataclass
class PPOConfig:
    """PPO algorithm parameters."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5


@dataclass
class NetworkConfig:
    """Neural network architecture."""

    hidden_size: int = 64
    n_hidden_layers: int = 2
    activation: str = "tanh"


@dataclass
class SchedulerConfig:
    """Policy scheduling configuration."""

    strategy: str = "student_only"
    trust_length: int = 5
    epsilon: float | None = None
    policy_trust_threshold: float | None = None
    internal_policy_warmup_length: int | None = None
    decrease_until_global_step: int | None = None
    iteration_to_switch: int | None = None
    step_to_switch: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for compatibility."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class TeacherConfig:
    """Teacher policy configuration."""

    type: str | None = None
    checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for compatibility."""
        if self.type is None:
            return None
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ExperimentConfig:
    """Experiment metadata and settings."""

    name: str | None = None
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "logs"
    capture_video: bool = False
    save_freq: int = 100


@dataclass
class DistillationConfig:
    """Complete configuration for distillation experiments."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "DistillationConfig":
        """Create config from dictionary (e.g., from YAML)."""
        return cls(
            environment=EnvironmentConfig(**config_dict.get("environment", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            ppo=PPOConfig(**config_dict.get("ppo", {})),
            network=NetworkConfig(**config_dict.get("network", {})),
            scheduler=SchedulerConfig(**config_dict.get("scheduler", {})),
            teacher=TeacherConfig(**config_dict.get("teacher", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
            tracker=TrackerConfig(**config_dict.get("tracker", {})),
        )

    def update_from_dict(self, updates: dict[str, Any]):
        """Update config fields from a dictionary of updates."""
        for section_name, section_updates in updates.items():
            if hasattr(self, section_name) and isinstance(section_updates, dict):
                section = getattr(self, section_name)
                for key, value in section_updates.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
