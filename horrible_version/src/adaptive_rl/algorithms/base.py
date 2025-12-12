from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class Algorithm(ABC):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cuda",
        seed: int | None = None,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    @abstractmethod
    def predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        pass

    @abstractmethod
    def train_step(
        self,
        rollout_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    def get_policy(self) -> nn.Module:
        return self.policy

    def get_value_function(self) -> nn.Module | None:
        return getattr(self, "value_function", None)


class AlgorithmRegistry:
    _algorithms: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(algorithm_cls):
            cls._algorithms[name] = algorithm_cls
            return algorithm_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._algorithms:
            raise ValueError(
                f"Algorithm '{name}' not registered. Available: {list(cls._algorithms.keys())}"
            )
        return cls._algorithms[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> Algorithm:
        algorithm_cls = cls.get(name)
        return algorithm_cls(**kwargs)
