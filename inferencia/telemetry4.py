import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from arena import extraer_ram_segura
from pettingzoo.atari import boxing_v2
import numpy as np

env = boxing_v2.env(render_mode=None, obs_type="rgb_image")
env.reset()

for i in range(2000):
    for agent_id in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue

        if reward > 0:
            ram = extraer_ram_segura(env)
            print(f"{agent_id} SCORED! Byte 14: {ram[14]}, Byte 15: {ram[15]}")

        action = env.action_space(agent_id).sample()
        env.step(action)

env.close()
