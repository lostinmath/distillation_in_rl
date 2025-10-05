"""Configuration for experiment tracking."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TensorBoardConfig:
    """TensorBoard backend configuration."""

    enabled: bool = True
    log_dir: str = "logs/tensorboard"
    flush_secs: int = 10


@dataclass
class MLFlowConfig:
    """MLFlow backend configuration."""

    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class WandbConfig:
    """Weights & Biases backend configuration."""

    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    tags: list[str] | None = None
    name: str | None = None
    notes: str | None = None
    mode: str = "online"  # online, offline, or disabled


@dataclass
class NeptuneConfig:
    """Neptune.ai backend configuration."""

    enabled: bool = False
    project: str | None = None
    api_token: str | None = None
    name: str | None = None
    tags: list[str] | None = None
    source_files: list[str] | None = None


@dataclass
class ConsoleConfig:
    """Console logging configuration."""

    enabled: bool = True
    log_frequency: int = 10
    verbose: bool = False


@dataclass
class CSVConfig:
    """CSV logging configuration."""

    enabled: bool = True
    log_dir: str = "logs/csv"
    filename: str = "metrics.csv"
    save_frequency: int = 10


@dataclass
class TrackerConfig:
    """Complete tracking configuration."""

    backends: list[str] = field(
        default_factory=lambda: ["tensorboard", "console", "csv"]
    )
    log_frequency: int = 10
    log_videos: bool = False
    log_figures: bool = True
    log_models: bool = True
    save_frequency: int = 100

    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    mlflow: MLFlowConfig = field(default_factory=MLFlowConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    neptune: NeptuneConfig = field(default_factory=NeptuneConfig)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    csv: CSVConfig = field(default_factory=CSVConfig)

    def is_backend_enabled(self, backend_name: str) -> bool:
        """Check if a backend is enabled."""
        if backend_name not in self.backends:
            return False

        backend_config = getattr(self, backend_name.lower(), None)
        if backend_config and hasattr(backend_config, "enabled"):
            return backend_config.enabled

        return backend_name in self.backends
