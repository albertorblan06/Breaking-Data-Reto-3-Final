import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
import numpy as np
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura
import importlib.util

# Load our agent
ruta_ent = os.path.dirname(os.path.abspath(__file__))
archivo = os.path.join(ruta_ent, "submission_rf.py")
spec = importlib.util.spec_from_file_location("modulo_agente", archivo)
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)
a1 = modulo.AgenteInferencia()
a1.ruta_modelo = ruta_ent
a1.configurar()

# Load random opponent
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))
sys.path.insert(0, ruta_inf)
import arena
a2 = arena.cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_random"))
a2.configurar()

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

score_us = 0
score_them = 0
last_30_actions = []

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    
    if agent_id == "first_0":
        score_us += reward
    else:
        score_them += reward
    
    if term or trunc:
        break
    
    ram = extraer_ram_segura(env)
    
    if agent_id == "first_0":
        estado = {"imagen": obs, "ram": ram, "soy_blanco": True}
        action = a1.predict(estado)
        last_30_actions.append(action)
        if len(last_30_actions) > 30:
            last_30_actions.pop(0)
    else:
        estado = {"imagen": obs, "ram": ram, "soy_blanco": False}
        action = a2.predict(estado)
    
    env.step(action)
    
    # Print detailed state every 20 frames
    if i > 0 and i % 20 == 0 and agent_id == "first_0":
        my_x, my_y = int(ram[32]), int(ram[34])
        opp_x, opp_y = int(ram[33]), int(ram[35])
        dist_x = abs(opp_x - my_x)
        dist_y = abs(opp_y - my_y)
        print(f"Frame {i}: me=({my_x},{my_y}) opp=({opp_x},{opp_y}) dist=({dist_x},{dist_y}) last_actions={last_30_actions[-10:]} score={score_us}")

print(f"\nFinal score: US={score_us} THEM={score_them}")

# Action distribution
from collections import Counter
if last_30_actions:
    print(f"Last 30 action distribution: {dict(Counter(last_30_actions))}")

env.close()
