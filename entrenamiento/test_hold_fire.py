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
            dist_x = abs(su_x - mi_x)
            if dist_x > 26:
                action = 3  # Move right
            else:
                action = 1  # HOLD FIRE
        else:
            action = 0  # NOOP

    env.step(action)
    steps += 1
    if steps >= 400:
        break

print("Final Score with Holding Fire:", score)
