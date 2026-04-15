import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import importlib.util

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

spec = importlib.util.spec_from_file_location("modulo", os.path.join(ruta_ent, "submission_rf.py"))
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)

a1 = modulo.AgenteInferencia()
a1.ruta_modelo = ruta_ent
a1.configurar()

a2 = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_vision"))
a2.configurar()

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

score_w = 0
min_dist = 999
opp_x_hist = []

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if agent_id == "first_0":
        score_w += reward
    if term or trunc or i > 8000:
        break
    
    ram = extraer_ram_segura(env)
    my_x, my_y = int(ram[32]), int(ram[34])
    opp_x, opp_y = int(ram[33]), int(ram[35])
    dist_x = abs(opp_x - my_x)
    dist_y = abs(opp_y - my_y)
    
    if dist_x < min_dist:
        min_dist = dist_x
    
    if agent_id == "first_0":
        estado = {"imagen": obs, "ram": ram, "soy_blanco": True}
        action = a1.predict(estado)
    else:
        estado = {"imagen": obs, "ram": ram, "soy_blanco": False}
        action = a2.predict(estado)
        opp_x_hist.append(opp_x)
    
    if i % 500 == 0 and i > 0:
        print(f"Frame {i}: me=({my_x},{my_y}) opp=({opp_x},{opp_y}) dist=({dist_x},{dist_y}) score_w={score_w}")

# Check if vision opponent moved
if opp_x_hist:
    opp_x_range = max(opp_x_hist) - min(opp_x_hist)
    opp_y_vals = [0]  # We don't track Y for opponent
    print(f"\nVision opponent X range: {min(opp_x_hist)} to {max(opp_x_hist)} (spread: {opp_x_range})")
    print(f"Vision opponent seemed to stay: {'STATIONARY' if opp_x_range < 5 else 'MOVING'}")

ram_final = extraer_ram_segura(env)
print(f"\nFinal: W={score_w} RAM W={int(ram_final[18])} B={int(ram_final[19])} min_dist={min_dist}")
env.close()
