import gymnasium as gym, ale_py
gym.register_envs(ale_py)
env = gym.make("ALE/Boxing-v5")
print(env.unwrapped.get_action_meanings())
