from pettingzoo.atari import boxing_v2
import sys
import os

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

steps = 0
score = 0
for agent_id in env.agent_iter():
    obs, reward, term, trunc, info = env.last()

    if agent_id == "first_0":
        score += reward

    if term or trunc:
        action = None
    else:
        ram = env.unwrapped.ale.getRAM()
        mi_x, mi_y = ram[32], ram[34]
        su_x, su_y = ram[33], ram[35]

        if agent_id == "first_0":
            action = 11  # RIGHTFIRE
            if steps % 10 < 5:
                action = 14  # UPRIGHTFIRE
            else:
                action = 16  # DOWNRIGHTFIRE
        else:
            action = 0  # NOOP

    env.step(action)
    steps += 1
    if steps >= 800:
        break

print("Final Score Diagonal Fire:", score)
