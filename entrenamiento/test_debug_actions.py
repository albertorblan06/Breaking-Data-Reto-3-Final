import sys
import os
import time
import numpy as np

ruta_entrenamiento = os.path.dirname(os.path.abspath(__file__))
ruta_inferencia = os.path.abspath(os.path.join(ruta_entrenamiento, "..", "inferencia"))
sys.path.append(ruta_inferencia)

from arena import torneo
import importlib.util

def cargar_mi_agente():
    archivo_submission = os.path.join(ruta_entrenamiento, "submission_rf.py")
    spec = importlib.util.spec_from_file_location("modulo_agente", archivo_submission)
    modulo_agente = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo_agente)
    agente = modulo_agente.AgenteInferencia()
    agente.ruta_modelo = ruta_entrenamiento
    agente.configurar()
    return agente

import arena
a1 = cargar_mi_agente()
a2 = arena.cargar_agente_desde_carpeta(os.path.join(ruta_inferencia, "modelos", "equipo_random"))
a2.configurar()

from pettingzoo.atari import boxing_v2
env = boxing_v2.env(obs_type="rgb_image")

env.reset()
actions = []
for agent_id in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        action = None
    else:
        ram = arena.extraer_ram_segura(env)
        estado = {"imagen": obs, "ram": ram, "soy_blanco": (agent_id == "first_0")}
        if agent_id == "first_0":
            action = a1.predict(estado)
            actions.append(action)
        else:
            action = a2.predict(estado)
    env.step(action)
    if len(actions) > 50:
        break
print("My first 50 actions:", actions)
