import gymnasium as gym
import numpy as np


from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def run_default_script():
    from stable_baselines3 import TD3
    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

    model.learn(total_timesteps=100, log_interval=10)
    print(model)
    vec_env = model.get_env() # does not work if model is loaded `!!!`

    obs = vec_env.reset()
    # This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        # vec_env.render("human")
        
def try_our_env():
    from octoplus.src.rl_models.td3_modification import TD3
    from octoplus.src.config_dataclasses.env_configs import ManiSkillConfig
    env_cfg = ManiSkillConfig(obs_mode='rgb', control_mode='pd_ee_delta_pose',
                              render_mode='rgb_array', sim_backend='cpu',
                              reward_mode='dense', env_id='PickCube-v1', 
                              num_envs=1, num_steps=50, include_state=False, 
                              partial_reset=True, image_height=20, image_width=20, 
                              use_render_camera_as_input=False)# 256
    from octoplus.src.env.mani_skill_env import get_envs

    envs = get_envs(
            env_config=env_cfg,
        )
    n_actions = envs.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model2 = TD3("MultiInputPolicy", envs, action_noise=action_noise, verbose=1)    

    model2.learn(total_timesteps=100, log_interval=10)
    
    
# home/piscenco/thesis_octo/octoplus_env/lib/python3.10/site-packages/stable_baselines3/common/buffers.py:605: UserWarning: This system does not have apparently enough memory to store the complete replay buffer 393.25GB > 243.28GB
try_our_env()