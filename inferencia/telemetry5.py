import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from arena import extraer_ram_segura
from pettingzoo.atari import boxing_v2
import numpy as np

env = boxing_v2.env(render_mode=None, obs_type="rgb_image")
env.reset()

history = []
first_scores = []
second_scores = []

for i in range(2000):
    for agent_id in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue

        ram = extraer_ram_segura(env)

        if agent_id == "second_0":
            history.append(ram.copy())

        if reward > 0:
            # Rebobinar 5 frames para ver que cambio
            if len(history) > 5:
                if agent_id == "first_0":
                    first_scores.append(history[-5:])
                else:
                    second_scores.append(history[-5:])

        action = env.action_space(agent_id).sample()
        env.step(action)

env.close()

if first_scores and second_scores:
    fs = np.array(first_scores)  # shape (N_scores, 5_frames, 128)
    ss = np.array(second_scores)

    diff = np.mean(fs, axis=(0, 1)) - np.mean(ss, axis=(0, 1))
    print("Top bytes that differ between P1 scoring and P2 scoring:")
    indices = np.argsort(np.abs(diff))[::-1][:10]
    for i in indices:
        print(
            f"Byte {i}: P1 scored mean={np.mean(fs, axis=(0, 1))[i]:.1f}, P2 scored mean={np.mean(ss, axis=(0, 1))[i]:.1f}"
        )
