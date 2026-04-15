import os
import sys
import numpy as np
import gymnasium as gym
from pettingzoo.atari import boxing_v2


def heuristic_predict(ram, soy_blanco):
    ram_model = ram.copy()
    if not soy_blanco:
        ram_model[32], ram_model[33] = ram_model[33], ram_model[32]
        ram_model[34], ram_model[35] = ram_model[35], ram_model[34]
        ram_model[18], ram_model[19] = ram_model[19], ram_model[18]
        ram_model[107], ram_model[109] = ram_model[109], ram_model[107]
        ram_model[111], ram_model[113] = ram_model[113], ram_model[111]
        ram_model[101], ram_model[105] = ram_model[105], ram_model[101]
        ram_model[103], ram_model[105] = ram_model[105], ram_model[103]

    mi_x = int(ram_model[32])
    mi_y = int(ram_model[34])
    su_x = int(ram_model[33])
    su_y = int(ram_model[35])

    dist_x = abs(su_x - mi_x)
    dist_y = abs(su_y - mi_y)
    action = 0

    is_pinned_left = (mi_x < 35) and (su_x > mi_x)
    is_pinned_right = (mi_x > 115) and (su_x < mi_x)

    if (is_pinned_left or is_pinned_right) and dist_x < 35:
        if dist_y <= 15:
            if mi_y < 50:
                action = 5
            else:
                action = 2
        else:
            if is_pinned_left:
                action = 3
            else:
                action = 4
    elif dist_y > 8:
        if su_y > mi_y:
            action = 5
        else:
            action = 2
    elif dist_x > 28:
        if su_x > mi_x:
            action = 3
        else:
            action = 4
    elif dist_x < 24:
        if su_x > mi_x:
            action = 4
        else:
            action = 3
    else:
        action = 1

    return action


def generate_data(num_steps=100000):
    env = boxing_v2.env(obs_type="rgb_image")

    observations = []
    actions = []
    jab_timer = 0

    print(f"Generating {num_steps} steps of expert data...")
    env.reset()

    while len(observations) < num_steps:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
                env.step(action)
                continue

            try:
                ram = env.unwrapped.ale.getRAM()
            except AttributeError:
                ram = np.zeros(128, dtype=np.uint8)

            if agent == "first_0":
                action = heuristic_predict(ram, soy_blanco=True)

                observations.append(ram.copy())
                actions.append(action)

                if len(observations) % 10000 == 0:
                    print(f"Collected {len(observations)} samples...")

                if len(observations) >= num_steps:
                    env.step(action)
                    break
            else:
                action = env.action_space(agent).sample()

            env.step(action)

        if len(observations) >= num_steps:
            break

        if not env.agents:
            env.reset()
            jab_timer = 0

    obs_array = np.array(observations, dtype=np.uint8)
    act_array = np.array(actions, dtype=np.uint8)

    out_file = os.path.join(os.path.dirname(__file__), "expert_dataset.npz")
    np.savez_compressed(out_file, obs=obs_array, acts=act_array)
    print(f"Dataset generated with {len(obs_array)} samples at {out_file}")


if __name__ == "__main__":
    generate_data(num_steps=100000)
