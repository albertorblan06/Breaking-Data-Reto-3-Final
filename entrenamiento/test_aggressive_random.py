import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import numpy as np

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))
random_agent = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_random"))
random_agent.configurar()

def test_aggro(matches=5):
    wins, losses, ties = 0, 0, 0
    for i in range(matches):
        env = boxing_v2.env(obs_type="rgb_image")
        env.reset()
        scores = {'first_0': 0, 'second_0': 0}
        f = 0
        for agent_id in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            scores[agent_id] += reward
            if term or trunc: action = None
            else:
                ram = extraer_ram_segura(env)
                estado = {"imagen": obs, "ram": ram, "soy_blanco": (agent_id == "first_0")}
                if agent_id == "first_0":
                    f += 1
                    my_x = int(ram[32]); my_y = int(ram[34])
                    opp_x = int(ram[33]); opp_y = int(ram[35])
                    dx = opp_x - my_x; dy = opp_y - my_y
                    adx = abs(dx); ady = abs(dy)
                    
                    if adx < 16:
                        action = 4 if dx > 0 else 3 # Back off
                    elif adx <= 28 and ady <= 5:
                        if f % 3 == 0:
                            # Diagonal punch logic
                            if dy > 2: action = 16 if dx > 0 else 17
                            elif dy < -2: action = 14 if dx > 0 else 15
                            else: action = 11 if dx > 0 else 12
                        else:
                            # Defensive bob/weave
                            action = 2 if f % 4 < 2 else 5
                    else:
                        if dy > 2 and dx > 0: action = 16 if f % 3 == 0 else 8
                        elif dy > 2 and dx < 0: action = 17 if f % 3 == 0 else 9
                        elif dy < -2 and dx > 0: action = 14 if f % 3 == 0 else 6
                        elif dy < -2 and dx < 0: action = 15 if f % 3 == 0 else 7
                        else:
                            action = 3 if dx > 0 else 4
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

test_aggro(10)

