import sys
import os
import gymnasium as gym
import ale_py
import numpy as np

# Use absolute paths
ruta_entrenamiento = os.path.dirname(os.path.abspath(__file__))
ruta_inferencia = os.path.abspath(os.path.join(ruta_entrenamiento, "..", "inferencia"))
sys.path.append(ruta_inferencia)

import importlib.util


def cargar_mi_agente():
    archivo_submission = os.path.join(
        ruta_entrenamiento, "submission_aggressive_rhythm.py"
    )
    spec = importlib.util.spec_from_file_location("modulo_agente", archivo_submission)
    modulo_agente = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo_agente)
    agente = modulo_agente.AgenteInferencia()
    agente.ruta_modelo = ruta_entrenamiento
    agente.configurar()
    return agente


gym.register_envs(ale_py)
env = gym.make("ALE/Boxing-v5", obs_type="rgb")
obs, _ = env.reset()

agente = cargar_mi_agente()

actions_taken = []
for i in range(500):
    ram = env.unwrapped.ale.getRAM()
    estado = {"imagen": obs, "ram": ram, "soy_blanco": True}
    action = agente.predict(estado)
    actions_taken.append(action)
    obs, reward, term, trunc, _ = env.step(action)
    if term or trunc:
        break

from collections import Counter

print("Action counts:", Counter(actions_taken))
