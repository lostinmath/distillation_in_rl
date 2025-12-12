#!/usr/bin/env python3
"""
BASIC SMOKE TEST: Does our PPO converge on CartPole?
This should solve CartPole in ~200-500 episodes if working correctly.
"""

import torch
import gymnasium as gym
import numpy as np
from adaptive_rl.algorithms.ppo import PPO
from adaptive_rl.envs import make_vec_env

def test_basic_ppo():
    print("🧪 SMOKE TEST: Basic PPO on CartPole")

    # Simple setup
    env = make_vec_env("CartPole-v1", num_envs=4, seed=42)

    # Create PPO
    ppo = PPO(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=3e-4,
        n_steps=128,  # Small for quick test
        batch_size=64,
        n_epochs=4,
        device="cpu"  # Force CPU for debugging
    )

    print(f"✅ PPO created")
    print(f"✅ Obs space: {env.observation_space}")
    print(f"✅ Action space: {env.action_space}")

    # Test single prediction
    obs = env.reset()[0]
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    action, value, log_prob = ppo.predict(obs_tensor)

    print(f"✅ Single prediction works:")
    print(f"   Action: {action}")
    print(f"   Value: {value}")
    print(f"   Log prob: {log_prob}")

    # Test single environment step
    env_action = action.cpu().numpy()
    next_obs, reward, terminated, truncated, info = env.step(env_action)
    print(f"✅ Environment step works:")
    print(f"   Reward: {reward}")
    print(f"   Done: {terminated | truncated}")

    # Test mini training loop (just 3 iterations)
    print("\n🏃 Testing mini training loop...")
    episode_rewards = []
    current_episode_reward = np.zeros(4)

    for iteration in range(3):
        print(f"\nIteration {iteration + 1}/3")

        # Collect rollout
        rollout_buffer = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "log_probs": [],
        }

        obs = torch.tensor(env.reset()[0], dtype=torch.float32)

        for step in range(ppo.n_steps):
            action, value, log_prob = ppo.predict(obs)
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated | truncated

            rollout_buffer["observations"].append(obs)
            rollout_buffer["actions"].append(action)
            rollout_buffer["rewards"].append(torch.tensor(reward))
            rollout_buffer["dones"].append(torch.tensor(done))
            rollout_buffer["values"].append(value.squeeze(-1))
            rollout_buffer["log_probs"].append(log_prob)

            obs = torch.tensor(next_obs, dtype=torch.float32)
            current_episode_reward += reward

            # Track completed episodes
            for i, d in enumerate(done):
                if d:
                    episode_rewards.append(current_episode_reward[i])
                    current_episode_reward[i] = 0

        # Convert to training format
        rollout_data = {}
        for key, values in rollout_buffer.items():
            rollout_data[key] = torch.stack(values)
        rollout_data["next_observations"] = obs

        # Try training step
        try:
            metrics = ppo.train_step(rollout_data)
            print(f"   ✅ Training step completed")
            print(f"   📊 Loss: {metrics.get('loss/total', 'N/A'):.4f}")
            if episode_rewards:
                recent_rewards = episode_rewards[-10:]
                print(f"   🎯 Recent avg reward: {np.mean(recent_rewards):.2f}")
        except Exception as e:
            print(f"   ❌ Training step failed: {e}")
            raise

    print(f"\n🎉 SMOKE TEST PASSED!")
    print(f"📈 Total episodes completed: {len(episode_rewards)}")
    if episode_rewards:
        print(f"📊 Average reward: {np.mean(episode_rewards):.2f}")
    print("\n✅ Your PPO implementation seems to work!")

if __name__ == "__main__":
    test_basic_ppo()