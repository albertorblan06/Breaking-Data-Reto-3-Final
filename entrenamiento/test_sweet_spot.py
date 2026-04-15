import os, sys, gymnasium as gym, ale_py
import numpy as np

gym.register_envs(ale_py)

print("Starting sweet spot search...")

for dist_test_x in range(22, 34, 1):
    for dist_test_y in range(0, 5, 1):
        env = gym.make("ALE/Boxing-v5", obs_type="rgb")
        obs, _ = env.reset()

        score = 0
        jab_timer = 0
        for i in range(2000):
            ram = env.unwrapped.ale.getRAM()
            mi_x, mi_y = int(ram[32]), int(ram[34])
            su_x, su_y = int(ram[33]), int(ram[35])

            dist_x = abs(su_x - mi_x)
            dist_y = abs(su_y - mi_y)

            action = 0

            # 1. Align Y exactly
            if su_y > mi_y + dist_test_y:
                action = 5  # DOWN
            elif su_y < mi_y - dist_test_y:
                action = 2  # UP
            # 2. Get close in X
            elif dist_x > dist_test_x:
                action = 3 if su_x > mi_x else 4  # RIGHT or LEFT
            elif dist_x < dist_test_x:
                action = 4 if su_x > mi_x else 3  # Flee
            # 3. Punch if perfectly aligned
            else:
                jab_timer += 1
                if jab_timer >= 3:
                    action = 1
                    jab_timer = 0
                else:
                    action = 0

            obs, reward, term, trunc, _ = env.step(action)
            score += reward
            if term or trunc:
                break

        print(f"Sweet Spot X={dist_test_x} Y={dist_test_y}: Score = {score}")
