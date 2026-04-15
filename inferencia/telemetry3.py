import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from arena import extraer_ram_segura
from pettingzoo.atari import boxing_v2
import time
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
            dist_x = abs(int(ram[32]) - int(ram[33]))
            dist_y = abs(int(ram[34]) - int(ram[35]))
            print(
                f"PUNTAJE: {agent_id} gano {reward}. Distancia X: {dist_x}, Y: {dist_y}"
            )

        action = env.action_space(agent_id).sample()
        env.step(action)

env.close()
