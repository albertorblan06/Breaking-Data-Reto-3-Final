import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import numpy as np

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

random_agent = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_random"))
random_agent.configurar()

def test_camper(matches=5):
    wins, losses, ties = 0, 0, 0
    for i in range(matches):
        env = boxing_v2.env(obs_type="rgb_image")
        env.reset()
        scores = {'first_0': 0, 'second_0': 0}
        
        for agent_id in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            scores[agent_id] += reward
            if term or trunc: action = None
            else:
                ram = extraer_ram_segura(env)
                estado = {"imagen": obs, "ram": ram, "soy_blanco": (agent_id == "first_0")}
                if agent_id == "first_0":
                    my_x = int(ram[32]); my_y = int(ram[34])
                    opp_x = int(ram[33]); opp_y = int(ram[35])
                    dx = opp_x - my_x; dy = opp_y - my_y
                    adx = abs(dx); ady = abs(dy)
                    
                    # Safe Camper logic
                    if adx < 30 and ady < 5:
                        action = 1 if dx > 0 else 1 # FIRE
                    elif ady >= 5 and adx < 30:
                        # Align Y while backing off slightly
                        if dy > 0: action = 9 if dx > 0 else 8 # DOWNLEFT/DOWNRIGHT
                        else: action = 7 if dx > 0 else 6 # UPLEFT/UPRIGHT
                    else:
                        # Follow slowly
                        if ady > 3: action = 5 if dy > 0 else 2
                        else: action = 3 if dx > 0 else 4
                else:
                    action = random_agent.predict(estado)
            env.step(action)
        env.close()
        
        w = scores['first_0']; b = scores['second_0']
        if w > b: wins += 1
        elif b > w: losses += 1
        else: ties += 1
        print(f"Match {i+1}: Us {w} - {b} Random")
    print(f"Summary: {wins}W {losses}L {ties}T")

test_camper(10)

