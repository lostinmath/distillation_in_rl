#!/usr/bin/env python3
"""Real training script for adaptive RL experiments."""

import argparse
import sys
import time
from pathlib import Path
import traceback

import torch
import numpy as np
import gymnasium as gym

# Import our adaptive RL components
from adaptive_rl.algorithms.ppo import PPO
from adaptive_rl.envs import make_vec_env
from adaptive_rl.schedulers import create_scheduler
from adaptive_rl.teachers import create_teacher
from adaptive_rl.utils.logging import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Adaptive RL Training Script")
    parser.add_argument("--config-path", type=str, help="Configuration path")
    parser.add_argument("--config-name", type=str, help="Configuration name")

    args, unknown = parser.parse_known_args()

    # Parse Hydra-style overrides
    overrides = {}
    for arg in unknown:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to convert to appropriate type
            if value.lower() in ('true', 'false'):
                overrides[key] = value.lower() == 'true'
            else:
                try:
                    overrides[key] = int(value)
                except ValueError:
                    try:
                        overrides[key] = float(value)
                    except ValueError:
                        overrides[key] = value

    return args, overrides


def get_config_from_args(args, overrides):
    """Extract configuration from arguments and overrides."""
    # Extract experiment details from config name
    config_name = args.config_name or "unknown"
    parts = config_name.split("_")

    # Default configuration
    config = {
        'experiment_name': config_name,
        'environment': 'CartPole-v1',
        'scheduler': 'student_only',
        'teacher': 'optimal',
        'seed': 42,
        'total_timesteps': 50000,
        'eval_frequency': 5000,
        'num_envs': 4,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'device': 'cpu'
    }

    # Extract from config name if possible
    if 'reward_based' in config_name:
        config['scheduler'] = 'reward_based'
    elif 'epsilon' in config_name:
        if 'decreasing' in config_name:
            config['scheduler'] = 'epsilon_decreasing'
        else:
            config['scheduler'] = 'epsilon'
    elif 'teacher_only' in config_name:
        config['scheduler'] = 'teacher_only'
    elif 'student_only' in config_name:
        config['scheduler'] = 'student_only'

    if 'cartpole' in config_name:
        config['environment'] = 'CartPole-v1'
    elif 'lunarlander' in config_name:
        config['environment'] = 'LunarLander-v2'

    # Apply overrides
    config.update(overrides)

    return config


def setup_experiment(config):
    """Setup all components for the experiment."""
    print(f"Setting up experiment: {config['experiment_name']}")
    print(f"Environment: {config['environment']}")
    print(f"Scheduler: {config['scheduler']}")
    print(f"Teacher: {config['teacher']}")
    print(f"Total timesteps: {config['total_timesteps']}")

    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Create vectorized environment
    env = make_vec_env(
        config['environment'],
        num_envs=config['num_envs'],
        seed=config['seed']
    )

    # Create PPO agent
    ppo = PPO(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        device=config['device'],
        num_envs=config['num_envs']
    )

    # Create teacher (if needed)
    teacher = None
    if config['scheduler'] != 'student_only':
        teacher = create_teacher(
            teacher_type=config['teacher'],
            env_id=config['environment'],
            action_space=env.action_space,
            observation_space=env.observation_space,
            device=config['device']
        )

    # Create scheduler
    scheduler = create_scheduler(
        config['scheduler'],
        student_policy=ppo,
        teacher_policy=teacher,
        num_envs=config['num_envs']
    )

    # Create logger
    logger = Logger(
        run_name=config['experiment_name'],
        log_dir="logs",
        use_tensorboard=True,
        use_mlflow=False,
        silence=False
    )

    return env, ppo, teacher, scheduler, logger


def run_training(env, ppo, teacher, scheduler, logger, config):
    """Run the main training loop."""
    print("Starting training...")
    start_time = time.time()

    # Initialize
    obs, _ = env.reset()
    episode_rewards = np.zeros(config['num_envs'])
    episode_lengths = np.zeros(config['num_envs'])
    step = 0
    episode_count = 0

    # Track learning curves
    learning_curve_data = []
    recent_rewards = []
    eval_rewards = []

    # Log experiment parameters
    logger.log_param("environment", config['environment'])
    logger.log_param("scheduler", config['scheduler'])
    logger.log_param("teacher", config['teacher'])
    logger.log_param("total_timesteps", config['total_timesteps'])
    logger.log_param("learning_rate", config['learning_rate'])

    while step < config['total_timesteps']:
        # Collect rollouts
        rollout_data = []
        for rollout_step in range(config['n_steps']):
            if step >= config['total_timesteps']:
                break
            # Get policy decisions from scheduler
            policy_choices = scheduler.choose_policy_type(
                iteration=step // config['n_steps'],
                global_step=step,
                steps_since_reset=rollout_step,
                prev_reward=episode_rewards
            )

            # Get actions from appropriate policies
            actions = []
            values = []
            log_probs = []

            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            for env_idx in range(config['num_envs']):
                if policy_choices[env_idx] == "teacher" and teacher is not None:
                    action_scalar = teacher.act(obs[env_idx:env_idx+1])
                    # Convert scalar action to proper tensor format
                    if isinstance(action_scalar, np.ndarray):
                        action_scalar = action_scalar.item() if action_scalar.size == 1 else action_scalar[0]
                    action = torch.tensor(action_scalar, dtype=torch.long).unsqueeze(0)
                    value = torch.zeros(1, 1)  # Match PPO's value shape (batch_size, 1)
                    log_prob = torch.zeros(1)
                else:
                    # Use student policy (PPO)
                    action, value, log_prob = ppo.predict(obs_tensor[env_idx:env_idx+1])

                actions.append(action)
                values.append(value)
                log_probs.append(log_prob)

            actions = torch.cat(actions)
            values = torch.cat(values)
            log_probs = torch.cat(log_probs)

            # Environment step
            next_obs, rewards, terminated, truncated, infos = env.step(actions.cpu().numpy())

            # Store rollout data
            rollout_data.append({
                'obs': obs.copy(),
                'actions': actions.clone(),
                'values': values.clone(),
                'log_probs': log_probs.clone(),
                'rewards': rewards.copy(),
                'terminated': terminated.copy(),
                'policy_choices': policy_choices.copy()
            })

            # Update tracking
            episode_rewards += rewards
            episode_lengths += 1

            # Handle episode endings
            for env_idx in range(config['num_envs']):
                if terminated[env_idx] or truncated[env_idx]:
                    logger.log_metrics(f"episode_reward_env_{env_idx}", episode_rewards[env_idx], step)
                    logger.log_metrics(f"episode_length_env_{env_idx}", episode_lengths[env_idx], step)
                    recent_rewards.append(episode_rewards[env_idx])
                    episode_rewards[env_idx] = 0
                    episode_lengths[env_idx] = 0
                    episode_count += 1

            obs = next_obs
            step += config['num_envs']  # Correct: each rollout step = num_envs environment steps

        # Update PPO with rollout data
        if len(rollout_data) > 0:
            # Convert rollout data to tensors for PPO training
            # Flatten (num_steps, num_envs) -> (num_steps * num_envs,) for batch processing
            all_obs = []
            all_actions = []
            all_values = []
            all_log_probs = []
            all_rewards = []
            all_dones = []

            for r in rollout_data:
                # Each rollout step has data for all environments
                all_obs.append(r['obs'])  # shape: (num_envs, obs_dim)
                all_actions.append(r['actions'].cpu().numpy())  # shape: (num_envs,)
                all_values.append(r['values'].squeeze(-1).cpu().numpy())  # shape: (num_envs,)
                all_log_probs.append(r['log_probs'].cpu().numpy())  # shape: (num_envs,)
                all_rewards.append(r['rewards'])  # shape: (num_envs,)
                all_dones.append(r['terminated'])  # shape: (num_envs,)

            # Stack and flatten for batch processing
            rollout_tensor_data = {
                'observations': torch.tensor(np.concatenate(all_obs, axis=0), dtype=torch.float32),  # (total_samples, obs_dim)
                'actions': torch.tensor(np.concatenate(all_actions, axis=0), dtype=torch.long),  # (total_samples,)
                'values': torch.tensor(np.concatenate(all_values, axis=0), dtype=torch.float32),  # (total_samples,)
                'log_probs': torch.tensor(np.concatenate(all_log_probs, axis=0), dtype=torch.float32),  # (total_samples,)
                'rewards': torch.tensor(np.concatenate(all_rewards, axis=0), dtype=torch.float32),  # (total_samples,)
                'dones': torch.tensor(np.concatenate(all_dones, axis=0), dtype=torch.float32),  # (total_samples,)
            }

            # Train PPO
            training_metrics = ppo.train_step(rollout_tensor_data)

            # Log training metrics
            for metric_name, metric_value in training_metrics.items():
                logger.log_metrics(metric_name, metric_value, step)

            print(f"Step {step}: PPO updated - Policy loss: {training_metrics.get('loss/policy', 0):.4f}")

        # Logging every 1000 steps
        if step % 1000 == 0:
            teacher_usage = np.mean([np.mean([p == 'teacher' for p in r['policy_choices']])
                                   for r in rollout_data[-10:]] if rollout_data else [0])
            logger.log_metrics("teacher_usage_ratio", teacher_usage, step)

            # Average reward over recent episodes
            avg_recent_reward = np.mean(recent_rewards[-20:]) if recent_rewards else 0
            logger.log_metrics("avg_recent_reward", avg_recent_reward, step)

            # Store learning curve data
            learning_curve_data.append({
                'step': step,
                'avg_reward': avg_recent_reward,
                'teacher_usage': teacher_usage,
                'episode_count': episode_count
            })

        # Evaluation
        if step % config['eval_frequency'] == 0:
            eval_reward = evaluate_policy(env, ppo, teacher, scheduler, config)
            logger.log_metrics("eval_reward", eval_reward, step)
            eval_rewards.append({'step': step, 'reward': eval_reward})
            print(f"Step {step}: Eval reward = {eval_reward:.2f}")

    # Training complete
    duration = time.time() - start_time
    print(f"Training completed in {duration:.1f}s ({step} steps, {episode_count} episodes)")

    # Final evaluation
    eval_reward = evaluate_policy(env, ppo, teacher, scheduler, config)
    logger.log_metrics("final_reward", eval_reward, step)
    logger.log_param("training_duration", duration)

    return {
        'final_performance': eval_reward,
        'training_time': duration,
        'total_episodes': episode_count,
        'steps': step,
        'learning_curve': learning_curve_data,
        'eval_rewards': eval_rewards
    }


def evaluate_policy(env, ppo, teacher, scheduler, config, num_episodes=5):
    """Evaluate the current policy."""
    total_reward = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 1000:  # Max episode length
            # Use student policy for evaluation
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action, _, _ = ppo.predict(obs_tensor)

            obs, rewards, terminated, truncated, _ = env.step(action.cpu().numpy())
            episode_reward += np.mean(rewards)
            done = np.any(terminated | truncated)
            step += 1

        total_reward += episode_reward

    return total_reward / num_episodes


def main():
    """Main training function."""
    try:
        # Parse arguments
        args, overrides = parse_args()
        config = get_config_from_args(args, overrides)

        # Setup experiment
        env, ppo, teacher, scheduler, logger = setup_experiment(config)

        # Run training
        results = run_training(env, ppo, teacher, scheduler, logger, config)

        # Cleanup
        logger.close()
        env.close()

        print("Training completed successfully!")
        print(f"Final performance: {results['final_performance']:.2f}")

        return 0

    except Exception as e:
        print(f"Training failed with error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())