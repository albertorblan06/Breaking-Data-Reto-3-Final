import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import numpy as np

# Test random vs random - simple pure random
ruta_inf = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
a2 = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_random"))
a2.configurar()

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

score_a = 0
score_b = 0
score_changes = []
min_dist = 999

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if agent_id == "first_0":
        score_a += reward
        if reward != 0:
            ram = extraer_ram_segura(env)
            score_changes.append((i, "A", reward, int(ram[32]), int(ram[34]), int(ram[33]), int(ram[35]), abs(int(ram[33])-int(ram[32])), abs(int(ram[35])-int(ram[34]))))
    else:
        score_b += reward
        if reward != 0:
            ram = extraer_ram_segura(env)
            score_changes.append((i, "B", reward, int(ram[32]), int(ram[34]), int(ram[33]), int(ram[35]), abs(int(ram[33])-int(ram[32])), abs(int(ram[35])-int(ram[34]))))
    
    if term or trunc or i > 8000:
        break
    
    ram = extraer_ram_segura(env)
    dist_x = abs(int(ram[33]) - int(ram[32]))
    if dist_x < min_dist and i > 100:
        min_dist = dist_x
    
    if agent_id == "first_0":
        action = np.random.randint(0, 18)
    else:
        action = a2.predict({"imagen": obs, "ram": ram, "soy_blanco": False})
    
    env.step(action)

print(f"Random vs Random: A={score_a}, B={score_b}")
print(f"Minimum dist_x seen after frame 100: {min_dist}")
print(f"\nScoring events:")
for sc in score_changes:
    print(f"  Frame {sc[0]}: Player {sc[1]} scored {sc[2]}! X=({sc[3]},{sc[5]}) Y=({sc[4]},{sc[6]}) dist=({sc[7]},{sc[8]})")
env.close()
