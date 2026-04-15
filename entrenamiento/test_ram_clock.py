import gymnasium as gym, ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Boxing-v5", obs_type="ram")
obs, _ = env.reset()

history = []
for i in range(10):
    obs, _, _, _, _ = env.step(0)
    history.append(obs)

import numpy as np

history = np.array(history)

# Find columns that change every frame
for col in range(128):
    if len(set(history[:, col])) > 3:
        print(f"RAM[{col}] oscillates: {history[:, col]}")
