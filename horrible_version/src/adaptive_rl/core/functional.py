"""Pure functional computations for PPO training.

This module contains stateless, pure functions for calculations that don't
require side effects. All functions are deterministic given the same inputs.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GAEOutput:
    """Output of GAE computation."""

    advantages: Tensor
    returns: Tensor


@dataclass(frozen=True)
class PPOLossOutput:
    """Output of PPO loss computation."""

    policy_loss: Tensor
    value_loss: Tensor
    entropy_loss: Tensor
    approx_kl: Tensor
    clipfrac: Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    dones: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> GAEOutput:
    """Compute Generalized Advantage Estimation (GAE).

    Pure function - no side effects, deterministic output.

    Args:
        rewards: Rewards for each timestep (T, B)
        values: Value estimates (T, B)
        next_values: Next value estimates (T, B)
        dones: Episode termination flags (T, B)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        GAEOutput with advantages and returns
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - dones[t].float()
            nextvalues = next_values
        else:
            nextnonterminal = 1.0 - dones[t + 1].float()
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        )

    returns = advantages + values
    return GAEOutput(advantages=advantages, returns=returns)


def compute_ppo_policy_loss(
    logprobs_old: Tensor,
    logprobs_new: Tensor,
    advantages: Tensor,
    clip_coef: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute PPO clipped policy loss.

    Pure function for policy gradient calculation.

    Args:
        logprobs_old: Log probabilities from old policy
        logprobs_new: Log probabilities from new policy
        advantages: Advantage estimates
        clip_coef: PPO clipping coefficient

    Returns:
        Tuple of (policy_loss, approx_kl, clipfrac)
    """
    logratio = logprobs_new - logprobs_old
    ratio = logratio.exp()

    # Normalize advantages
    advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss with clipping
    pg_loss1 = -advantages_norm * ratio
    pg_loss2 = -advantages_norm * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Metrics
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

    return pg_loss, approx_kl, clipfrac


def compute_ppo_value_loss(
    values_pred: Tensor,
    returns: Tensor,
    values_old: Tensor | None = None,
    clip_coef: float | None = None,
) -> Tensor:
    """Compute PPO value function loss.

    Pure function for value loss calculation.

    Args:
        values_pred: Predicted values from current network
        returns: Target returns
        values_old: Old value predictions (for clipping)
        clip_coef: Optional clipping coefficient

    Returns:
        Value loss
    """
    values_pred = values_pred.view(-1)
    returns = returns.view(-1)

    if clip_coef is not None and values_old is not None:
        # Clipped value loss
        values_old = values_old.view(-1)
        v_loss_unclipped = (values_pred - returns) ** 2
        v_clipped = values_old + torch.clamp(
            values_pred - values_old,
            -clip_coef,
            clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        # Simple MSE loss
        v_loss = 0.5 * ((values_pred - returns) ** 2).mean()

    return v_loss


def compute_ppo_losses(
    logprobs_old: Tensor,
    logprobs_new: Tensor,
    values_pred: Tensor,
    advantages: Tensor,
    returns: Tensor,
    entropy: Tensor,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    values_old: Tensor | None = None,
    clip_vloss: bool = False,
) -> PPOLossOutput:
    """Compute all PPO losses in a single pure function.

    Args:
        logprobs_old: Log probabilities from old policy
        logprobs_new: Log probabilities from new policy
        values_pred: Predicted values
        advantages: Advantage estimates
        returns: Target returns
        entropy: Entropy of current policy
        clip_coef: PPO clipping coefficient
        vf_coef: Value function loss coefficient
        ent_coef: Entropy bonus coefficient
        values_old: Old value predictions (for clipping)
        clip_vloss: Whether to clip value loss

    Returns:
        PPOLossOutput with all loss components
    """
    # Policy loss
    pg_loss, approx_kl, clipfrac = compute_ppo_policy_loss(
        logprobs_old, logprobs_new, advantages, clip_coef
    )

    # Value loss
    v_loss = compute_ppo_value_loss(
        values_pred,
        returns,
        values_old if clip_vloss else None,
        clip_coef if clip_vloss else None,
    )

    # Entropy loss (negative for bonus)
    entropy_loss = entropy.mean()

    return PPOLossOutput(
        policy_loss=pg_loss,
        value_loss=v_loss,
        entropy_loss=entropy_loss,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
    )


def aggregate_episode_metrics(
    episode_returns: list, episode_lengths: list, window_size: int = 100
) -> dict:
    """Aggregate episode metrics with moving averages.

    Pure function for metric aggregation.

    Args:
        episode_returns: List of episode returns
        episode_lengths: List of episode lengths
        window_size: Size of moving average window

    Returns:
        Dictionary of aggregated metrics
    """
    import numpy as np

    if not episode_returns:
        return {
            "return_mean": 0.0,
            "return_std": 0.0,
            "length_mean": 0.0,
            "length_std": 0.0,
        }

    # Take last window_size episodes
    recent_returns = episode_returns[-window_size:]
    recent_lengths = episode_lengths[-window_size:]

    return {
        "return_mean": np.mean(recent_returns),
        "return_std": np.std(recent_returns),
        "return_min": np.min(recent_returns),
        "return_max": np.max(recent_returns),
        "length_mean": np.mean(recent_lengths),
        "length_std": np.std(recent_lengths),
    }


def normalize_observations(
    obs: Tensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    epsilon: float = 1e-8,
    training: bool = True,
    momentum: float = 0.99,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize observations with running statistics.

    Pure function that returns updated statistics without modifying inputs.

    Args:
        obs: Observations to normalize
        running_mean: Current running mean
        running_var: Current running variance
        epsilon: Small value for numerical stability
        training: Whether to update statistics
        momentum: Momentum for running statistics

    Returns:
        Tuple of (normalized_obs, new_running_mean, new_running_var)
    """
    if running_mean is None:
        running_mean = torch.zeros(obs.shape[-1], device=obs.device)
    if running_var is None:
        running_var = torch.ones(obs.shape[-1], device=obs.device)

    if training:
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)

        new_running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        new_running_var = momentum * running_var + (1 - momentum) * batch_var
    else:
        new_running_mean = running_mean
        new_running_var = running_var

    normalized_obs = (obs - new_running_mean) / torch.sqrt(new_running_var + epsilon)

    return normalized_obs, new_running_mean, new_running_var
