# from mani_skill2.utils.wrappers import RecordEpisode
# # to make it look a little more realistic, we will enable shadows, and record the "cameras" render mode
# env = gym.make(env_id, render_mode="cameras", enable_shadow=True)
# env = RecordEpisode(
#     env,
#     "./videos", # the directory to save replay videos and trajectories to
#     info_on_video=True # when True, will add informative text onto the replay video such as step counter, reward, and other metrics
# )

# # step through the environment with random actions
# obs, _ = env.reset()
# for i in range(100):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.human_render() # will render with a window if possible
# env.flush_video() # Save the video
# env.close()
# from IPython.display import Video
# Video("./videos/0.mp4", embed=True) # Watch our replay
