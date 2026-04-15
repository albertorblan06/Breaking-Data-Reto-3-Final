import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import importlib.util
import time

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

# Load our agent
spec = importlib.util.spec_from_file_location("modulo", os.path.join(ruta_ent, "submission_rf.py"))
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)

# Test against ONNX
a1 = modulo.AgenteInferencia()
a1.ruta_modelo = ruta_ent
a1.configurar()

a2 = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_onnx"))
a2.configurar()

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

score_w = 0
score_b = 0
actions_w = []

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if agent_id == "first_0":
        score_w += reward
    else:
        score_b += reward
    
    if term or trunc or i > 6000:
        break
    
    ram = extraer_ram_segura(env)
    
    if agent_id == "first_0":
        estado = {"imagen": obs, "ram": ram, "soy_blanco": True}
        action = a1.predict(estado)
        actions_w.append(action)
    else:
        estado = {"imagen": obs, "ram": ram, "soy_blanco": False}
        action = a2.predict(estado)
    
    if i % 200 == 0:
        my_x, my_y = int(ram[32]), int(ram[34])
        opp_x, opp_y = int(ram[33]), int(ram[35])
        dist_x = abs(opp_x - my_x)
        dist_y = abs(opp_y - my_y)
        score_w_ram = int(ram[18])
        score_b_ram = int(ram[19])
        print(f"Frame {i}: me=({my_x},{my_y}) opp=({opp_x},{opp_y}) dist=({dist_x},{dist_y}) score_w={score_w} score_b={score_b} ram_w={score_w_ram} ram_b={score_b_ram}")
    
    env.step(action)

env.close()

from collections import Counter
action_dist = Counter(actions_w)
print(f"\nFinal: W={score_w} B={score_b}")
print(f"Action distribution: {dict(sorted(action_dist.items()))}")
print(f"Total our actions: {len(actions_w)}")
