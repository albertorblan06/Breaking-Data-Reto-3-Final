from pettingzoo.atari import boxing_v2
import sys
import os

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

steps = 0
for agent_id in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        action = None
    else:
        action = 3 if agent_id == "first_0" else 0
    env.step(action)
    steps += 1
    if steps >= 20:
        break

ram = env.unwrapped.ale.getRAM()
print("After 10 frames of P1 moving RIGHT:")
print("RAM[32] (White X):", ram[32])
print("RAM[33] (Black X):", ram[33])
