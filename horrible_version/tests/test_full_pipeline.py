#!/usr/bin/env python3
"""
PIPELINE SMOKE TEST: Does the full training pipeline work?
Test student_only, teacher_only, and reward_based scheduling
"""

import torch
from omegaconf import DictConfig, OmegaConf
from adaptive_rl.pipelines import PipelineBuilder

def create_minimal_config():
    """Create a minimal working config for testing"""
    config = {
        "experiment": {
            "name": "smoke_test",
            "seed": 42,
            "device": "cpu",
        },
        "environment": {
            "name": "cartpole",
            "env_id": "CartPole-v1",
            "num_envs": 2,  # Small for quick test
        },
        "algorithm": {
            "name": "ppo",
            "_target_": "adaptive_rl.algorithms.ppo.PPO",
            "learning_rate": 3e-4,
            "n_steps": 64,  # Small for quick test
            "batch_size": 32,
            "n_epochs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "network": {
                "_target_": "adaptive_rl.networks.mlp.MLPNetwork",
                "hidden_sizes": [64, 64],
                "activation": "tanh"
            }
        },
        "training": {
            "total_timesteps": 500,  # Very small for quick test
            "eval_freq": 250,
            "checkpoint_freq": 1000,
        },
        "paths": {
            "log_dir": "test_logs",
            "checkpoint_dir": "test_logs/checkpoints",
        },
        "tracker": {
            "backends": ["console"],
            "console": {"enabled": True},
        }
    }
    return OmegaConf.create(config)

def test_scheduler(scheduler_config, test_name):
    """Test a specific scheduler configuration"""
    print(f"\n🧪 Testing {test_name}")

    config = create_minimal_config()
    config.scheduler = scheduler_config
    config.experiment.name = f"smoke_test_{test_name}"

    if "teacher" in scheduler_config:
        config.teacher = {
            "name": "optimal",
            "_target_": "adaptive_rl.teachers.optimal.OptimalPolicy"
        }
    else:
        config.teacher = None

    try:
        # Build pipeline
        pipeline = PipelineBuilder.build_pipeline(config)
        print(f"   ✅ Pipeline built successfully")

        # Test a few steps (not full run)
        print(f"   🏃 Running mini training...")

        # Just test the pipeline components work
        n_envs = config.environment.num_envs
        n_steps = config.algorithm.n_steps

        obs = torch.tensor(pipeline.env.reset()[0], device=pipeline.device, dtype=torch.float32)

        # Test 2 training iterations
        for iteration in range(2):
            rollout_buffer = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "values": [],
                "log_probs": [],
            }

            for step in range(n_steps):
                # Test scheduler
                policies = pipeline.scheduler.choose_policy_type(
                    iteration=iteration,
                    global_step=iteration * n_steps + step,
                    steps_since_reset=torch.zeros(n_envs),
                    prev_reward=torch.zeros(n_envs),
                )
                policy_source = policies[0]

                # Get action
                if policy_source == "teacher" and pipeline.teacher:
                    action = pipeline.teacher.act(obs.cpu().numpy())
                    action = torch.tensor(action, device=pipeline.device)
                    value = torch.zeros(n_envs, 1, device=pipeline.device)
                    log_prob = torch.zeros(n_envs, device=pipeline.device)
                else:
                    action, value, log_prob = pipeline.algorithm.predict(obs)

                # Environment step
                next_obs, reward, terminated, truncated, info = pipeline.env.step(action.cpu().numpy())
                done = terminated | truncated

                # Store data
                rollout_buffer["observations"].append(obs)
                rollout_buffer["actions"].append(action)
                rollout_buffer["rewards"].append(torch.tensor(reward, device=pipeline.device))
                rollout_buffer["dones"].append(torch.tensor(done, device=pipeline.device))
                rollout_buffer["values"].append(value.squeeze(-1))
                rollout_buffer["log_probs"].append(log_prob)

                obs = torch.tensor(next_obs, device=pipeline.device, dtype=torch.float32)

            # Convert to training format
            rollout_data = {}
            for key, values in rollout_buffer.items():
                rollout_data[key] = torch.stack(values)
            rollout_data["next_observations"] = obs

            # Train
            metrics = pipeline.algorithm.train_step(rollout_data)
            print(f"   📊 Iteration {iteration + 1}: Loss = {metrics.get('loss/total', 0):.3f}")

        print(f"   ✅ {test_name} completed successfully!")
        return True

    except Exception as e:
        print(f"   ❌ {test_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 FULL PIPELINE SMOKE TEST")
    print("Testing if schedulers + teachers + PPO work together")

    # Test configurations
    tests = [
        {
            "config": {
                "name": "student_only",
                "_target_": "adaptive_rl.schedulers.simple.StudentOnlyScheduler"
            },
            "name": "student_only"
        },
        {
            "config": {
                "name": "teacher_only",
                "_target_": "adaptive_rl.schedulers.simple.TeacherOnlyScheduler"
            },
            "name": "teacher_only"
        },
        {
            "config": {
                "name": "reward_based",
                "_target_": "adaptive_rl.schedulers.reward_based.RewardBasedScheduler",
                "trust_period": 5,
                "initial_policy": "teacher"
            },
            "name": "reward_based"
        }
    ]

    results = []
    for test in tests:
        success = test_scheduler(test["config"], test["name"])
        results.append((test["name"], success))

    print(f"\n🎯 PIPELINE TEST RESULTS:")
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name}: {status}")

    all_passed = all(success for _, success in results)
    if all_passed:
        print(f"\n🎉 ALL PIPELINE TESTS PASSED!")
        print(f"✅ Your full system is working!")
    else:
        print(f"\n💥 SOME TESTS FAILED - need to debug")

if __name__ == "__main__":
    main()