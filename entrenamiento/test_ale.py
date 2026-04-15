import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
env = gym.make("ALE/Boxing-v5")
env.reset()
ale = env.unwrapped.ale
print("Available modes:", ale.getAvailableModes())
print("Available difficulties:", ale.getAvailableDifficulties())
