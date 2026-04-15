from pettingzoo.atari import boxing_v2
import sys
import os
import numpy as np

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

ruta_inferencia = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "inferencia")
)
sys.path.append(ruta_inferencia)
import arena

a2 = arena.cargar_agente_desde_carpeta(
    os.path.join(ruta_inferencia, "modelos", "equipo_onnx")
)
a2.configurar()

steps = 0
score = 0
for agent_id in env.agent_iter():
    obs, reward, term, trunc, info = env.last()

    if agent_id == "first_0":
        score += reward

    if term or trunc:
        action = None
    else:
        ram = arena.extraer_ram_segura(env)
        mi_x, mi_y = ram[32], ram[34]
        su_x, su_y = ram[33], ram[35]

        if agent_id == "first_0":
            action = np.random.randint(0, 18)
        else:
            action = a2.predict({"imagen": obs, "ram": ram, "soy_blanco": False})

    env.step(action)

print("Final Score Spam Punches:", score)
