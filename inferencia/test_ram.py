import gymnasium as gym
import ale_py
import numpy as np
import time

gym.register_envs(ale_py)
env = gym.make("ALE/Boxing-v5", obs_type="ram", render_mode=None)
obs, info = env.reset()

print("Buscando las posiciones X e Y...")
for i in range(100):
    obs, r, term, trunc, info = env.step(env.action_space.sample())

print(f"P1 X: {obs[32]}, Y: {obs[34]}")
print(f"P2 X: {obs[33]}, Y: {obs[35]}")
print(f"P1 Score: {obs[18]}, P2 Score: {obs[19]}")
print(f"Reloj: {obs[17]}")

env.close()
