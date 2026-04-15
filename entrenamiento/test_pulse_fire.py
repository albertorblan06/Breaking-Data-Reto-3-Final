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
            dist_y = abs(su_y - mi_y)

            if dist_x > 27:
                action = 3  # Move right
            elif dist_y > 2:
                action = 5 if su_y > mi_y else 2
            else:
                cycle = steps % 16
                if cycle < 4:
                    action = 1  # Fire for 4 frames
                else:
                    action = 0  # NOOP for 12 frames
        else:
            action = 0  # NOOP

    env.step(action)
    steps += 1
    if steps >= 800:
        break

print("Final Score Pulsed Fire:", score)
