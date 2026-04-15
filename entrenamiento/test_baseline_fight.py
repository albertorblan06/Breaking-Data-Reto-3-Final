import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import importlib.util

# Test equipo_onnx VS equipo_random - log coordinates at scoring moments
ruta_inf = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))

a1 = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_onnx"))
a1.configurar()
a2 = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_random"))
a2.configurar()

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

score_w = 0
score_b = 0
close_encounters = []

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if agent_id == "first_0":
        score_w += reward
    else:
        score_b += reward
    
    if term or trunc or i > 4000:
        break
    
    ram = extraer_ram_segura(env)
    my_x, my_y = int(ram[32]), int(ram[34])
    opp_x, opp_y = int(ram[33]), int(ram[35])
    dist_x = abs(opp_x - my_x)
    dist_y = abs(opp_y - my_y)
    
    # Log when players are close
    if dist_x <= 20 and i % 10 == 0:
        close_encounters.append((i, my_x, my_y, opp_x, opp_y, dist_x, dist_y, score_w, score_b))
    
    if agent_id == "first_0":
        estado = {"imagen": obs, "ram": ram, "soy_blanco": True}
        action = a1.predict(estado)
    else:
        estado = {"imagen": obs, "ram": ram, "soy_blanco": False}
        action = a2.predict(estado)
    
    env.step(action)

print(f"Final: White(onnx)={score_w}, Black(random)={score_b}")
print(f"\nClose encounters (dist_x<=20):")
for ce in close_encounters[:40]:
    print(f"  Frame {ce[0]}: W=({ce[1]},{ce[2]}) B=({ce[3]},{ce[4]}) dist=({ce[5]},{ce[6]}) score=W:{ce[7]} B:{ce[8]}")
env.close()
