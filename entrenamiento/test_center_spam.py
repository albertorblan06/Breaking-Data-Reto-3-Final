import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import numpy as np

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

random_agent = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_random"))
random_agent.configurar()

def test_center_spam(matches=5):
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
                    
                    if adx <= 28 and ady <= 15:
                        # In range of random punches
                        if dx > 0: action = 11 # RIGHTFIRE
                        else: action = 12 # LEFTFIRE
                    else:
                        # Move to center
                        if my_y > 95: action = 2
                        elif my_y < 80: action = 5
                        elif my_x > 80: action = 4
                        elif my_x < 60: action = 3
                        else: action = 0
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

test_center_spam(10)

