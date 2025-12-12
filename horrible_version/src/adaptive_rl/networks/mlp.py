import torch
from torch import nn


class MLPNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] = [64, 64],
        activation: str = "tanh",
        discrete: bool = True,
        std_init: float = 1.0,
    ):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim

        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }.get(activation, nn.Tanh)

        layers = []
        input_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation_fn())
            input_dim = hidden_size

        self.base = nn.Sequential(*layers)

        if discrete:
            self.output_layer = nn.Linear(input_dim, action_dim)
        else:
            self.mean_layer = nn.Linear(input_dim, action_dim)
            self.log_std = nn.Parameter(
                torch.ones(1, action_dim) * torch.log(torch.tensor(std_init))
            )

    def forward(
        self, obs: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.base(obs)

        if self.discrete:
            return self.output_layer(features)
        mean = self.mean_layer(features)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

# TODO: need to add more teachers, including ones for which weights are publicly available