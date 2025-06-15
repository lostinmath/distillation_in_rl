import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributions.normal import Normal

from octoplus.src.constants import ACTIVATION_LAYER_MAPPING

####################### ResNetNatureCNN ############################
# need my custom architectures. Could add Dillated conv.
# TODO: calculate layer hyperparams: stride, in/out channels, ...

################################################################


class CustomNatureCNN(nn.Module):
    """
    A neural network module that processes image and state observations using a NatureCNN architecture.
    Attributes:
        out_features (int): The total number of output features from the network.
        extractors (nn.ModuleDict): A dictionary containing the feature extractors for different types of observations.
    Methods:
        __init__(sample_obs):
            Initializes the NatureCNN with the given sample observations.
        forward(observations) -> torch.Tensor:
            Forward pass through the network, processing the observations and returning the encoded tensor.
    """

    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}
        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here

        conv1 = nn.Conv2d(
            in_channels, 3, kernel_size=7, stride=2, padding=3, bias=False
        )

        resnet = models.resnet18(pretrained=True)
        resnet = nn.Sequential(
            *list(resnet.children())[:-2]
        )  # Keep up to the conv layer
        cnn = nn.Sequential(
            conv1,
            resnet,
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),  # Flatten to get the feature vector
        )

        # To easily figure out the dimensions after flattening, pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[
                1
            ]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256
            print("Beware. The model Uses state of the environment.")

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


#################################################################


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes the weights and biases of a given neural network layer.

    Parameters:
    layer (torch.nn.Module): The neural network layer to initialize.
    std (float, optional): The standard deviation for the orthogonal initialization of the weights. Default is sqrt(2).
    bias_const (float, optional): The constant value to initialize the biases. Default is 0.0.

    Returns:
    torch.nn.Module: The initialized neural network layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """
    A neural network module that processes image and state observations using a NatureCNN architecture.
    Attributes:
        out_features (int): The total number of output features from the network.
        extractors (nn.ModuleDict): A dictionary containing the feature extractors for different types of observations.
    Methods:
        __init__(sample_obs):
            Initializes the NatureCNN with the given sample observations.
        forward(observations) -> torch.Tensor:
            Forward pass through the network, processing the observations and returning the encoded tensor.
    """

    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[
                1
            ]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        # ignore state
        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256
            print("Beware. The model Uses state of the environment.")

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


####################### More complicated Actor Critic ############################
# Critic Network (Value function)
class SimpleCritic(nn.Module):
    def __init__(self, latent_size, activation=nn.ReLU):
        super(SimpleCritic, self).__init__()
        self.fc1 = layer_init(nn.Linear(latent_size, 1024))  # Increased hidden size
        self.fc2 = layer_init(nn.Linear(1024, 512))
        self.fc3 = layer_init(nn.Linear(512, 1))  # Single output for the value
        self.activation = activation()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def __call__(self, x):
        return self.forward(x)


class SimpleActor(nn.Module):
    def __init__(self, latent_size, action_shape, activation=nn.ReLU):
        super(SimpleActor, self).__init__()
        self.fc1 = layer_init(nn.Linear(latent_size, 1024))  # Increased hidden size
        self.fc2 = layer_init(nn.Linear(1024, 512))
        self.fc3 = layer_init(
            nn.Linear(512, np.prod(action_shape)),
            std=0.01 * np.sqrt(2),
        )
        self.activation = activation()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def __call__(self, x):
        return self.forward(x)


########Basic actor & critic from the original implemetation##################3
# just made it a separate class############################################
class BasicActor(nn.Module):
    def __init__(self, latent_size, action_shape):
        super(BasicActor, self).__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(
                nn.Linear(512, np.prod(action_shape)),
                std=0.01 * np.sqrt(2),
            ),
        )

    def forward(self, x):
        return self.actor(x)

    def __call__(self, x):
        return self.forward(x)


class BasicCritic(nn.Module):
    def __init__(self, latent_size):
        super(BasicCritic, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )

    def forward(self, x):
        return self.critic(x)

    def __call__(self, x):
        return self.forward(x)


#################################################################
class Agent(nn.Module):
    def __init__(self, envs, sample_obs, model_config):
        super().__init__()

        if model_config.feature_net_type == "NatureCNN":
            self.feature_net = NatureCNN(sample_obs=sample_obs)
        elif model_config.feature_net_type == "CustomNatureCNN":
            self.feature_net = CustomNatureCNN(sample_obs=sample_obs)
        else:
            raise ValueError(
                f"Unknown feature_net_type: {model_config.feature_net_type}"
            )

        latent_size = self.feature_net.out_features

        if model_config.critic_type == "BasicCritic":
            self.critic = BasicCritic(
                latent_size=latent_size,
            )
        elif model_config.critic_type == "SimpleCritic":
            activation = ACTIVATION_LAYER_MAPPING[model_config.critic_activation_layer]
            self.critic = SimpleCritic(latent_size=latent_size, activation=activation)
        else:
            raise ValueError(f"Unknown critic_type: {model_config.critic_type}")

        if model_config.actor_type == "BasicActor":
            self.actor_mean = BasicActor(
                latent_size=latent_size,
                action_shape=envs.unwrapped.single_action_space.shape,
            )
        elif model_config.actor_type == "SimpleActor":
            activation = ACTIVATION_LAYER_MAPPING[model_config.critic_activation_layer]
            self.actor_mean = SimpleActor(
                latent_size=latent_size,
                action_shape=envs.unwrapped.single_action_space.shape,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown actor_type: {model_config.actor_type}")
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5
        )
        self.device = model_config.model_device

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd).to(action_mean.device) # changed this for some reason was on cpu
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        """
        Compute the action, log probability, entropy, and value for a given input state/observation.
        Note: this function is needed to optimize for the action mean.
        Args:
            x (torch.Tensor): The input tensor, typically representing the state or observation.
            action (torch.Tensor, optional): An optional tensor representing a specific action. 
                If not provided, an action will be sampled from the distribution.
        Returns:
            tuple: A tuple containing the following elements:
                - action (torch.Tensor): The actor predicted mean for the observation or provided action.
                - log_prob (torch.Tensor): The log probability of the action under the policy.
                - entropy (torch.Tensor): The entropy of the action distribution, representing 
                  the randomness of the policy.
                - value (torch.Tensor): The value of the input state as estimated by the critic.
        
        """
        x = self.feature_net(x)
        # Is is reasonable to add discretization here?
        # What should be a reasonable encoding dim? Using 512 for now
        action_mean = self.actor_mean(x)  # batch size, action dim
        action_logstd = self.actor_logstd.expand_as(
            action_mean
        )  # batch size, action dim
        action_logstd = action_logstd.to(self.device) # true here 
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        # TODO:should it be normal distr?
        if action is None:
            action = probs.sample()
        return (
            action,#f
            probs.log_prob(action).sum(1),#f
            probs.entropy().sum(1),# f
            self.critic(x), # false
        )

    def to(self, device: str):
        self.feature_net.to(device)
        self.critic.to(device)
        self.actor_mean.to(device)
        return self
