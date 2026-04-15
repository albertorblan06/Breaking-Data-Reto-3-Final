import sys
import os
import gymnasium as gym
import numpy as np
from pettingzoo.atari import boxing_v2
from stable_baselines3 import PPO

ruta_entrenamiento = os.path.dirname(os.path.abspath(__file__))
ruta_inferencia = os.path.abspath(os.path.join(ruta_entrenamiento, "..", "inferencia"))
sys.path.append(ruta_inferencia)
import arena

model = PPO.load(os.path.join(ruta_entrenamiento, "boxing_model_league_final.zip"))

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

a2 = arena.cargar_agente_desde_carpeta(
    os.path.join(ruta_inferencia, "modelos", "equipo_random")
)
a2.configurar()

score = 0
for agent_id in env.agent_iter():
    obs, reward, term, trunc, info = env.last()

    if agent_id == "first_0":
        score += reward

    if term or trunc:
        action = None
    else:
        ram = arena.extraer_ram_segura(env)
        if agent_id == "first_0":
            action, _ = model.predict(ram, deterministic=True)
            action = int(action)
        else:
            action = a2.predict({"imagen": obs, "ram": ram, "soy_blanco": False})

    env.step(action)

print("Final Score PPO vs Random:", score)
