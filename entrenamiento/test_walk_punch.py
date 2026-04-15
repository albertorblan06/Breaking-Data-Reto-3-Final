import sys
import os
import gymnasium as gym
import ale_py
import numpy as np

ruta_inferencia = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "inferencia")
)
sys.path.append(ruta_inferencia)

from pettingzoo.atari import boxing_v2
import arena

env = boxing_v2.env(obs_type="rgb_image")
env.reset()
a2 = arena.cargar_agente_desde_carpeta(
    os.path.join(ruta_inferencia, "modelos", "equipo_onnx")
)
a2.configurar()

jab_timer = 0
score = 0
for agent_id in env.agent_iter():
    obs, reward, term, trunc, info = env.last()

    if agent_id == "first_0":
        score += reward

    if term or trunc:
        action = None
        env.step(action)
        continue

    ram = arena.extraer_ram_segura(env)

    if agent_id == "first_0":
        mi_x, mi_y = ram[32], ram[34]
        su_x, su_y = ram[33], ram[35]

        dist_x = abs(int(su_x) - int(mi_x))
        dist_y = abs(int(su_y) - int(mi_y))

        jab_timer += 1

        # JUST walk right and punch right
        if dist_x > 27:
            action = 3  # Walk right
        elif dist_y > 2:
            action = 5 if int(su_y) > int(mi_y) else 2
        else:
            if jab_timer >= 10:
                action = 11  # RIGHTFIRE
                jab_timer = 0
            else:
                action = 0  # NOOP

        env.step(action)
    else:
        action = a2.predict({"imagen": obs, "ram": ram, "soy_blanco": False})
        env.step(action)

print("Final Score:", score)
