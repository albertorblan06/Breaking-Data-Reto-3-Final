import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import numpy as np

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

vision_agent = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_vision"))
vision_agent.configurar()

env = boxing_v2.env(obs_type="rgb_image")
env.reset()
scores = {'first_0': 0, 'second_0': 0}
frame_count = 0

for agent_id in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    scores[agent_id] += reward
    
    if term or trunc:
        action = None
        if frame_count > 2000 and not (term or trunc):
            break
    else:
        ram = extraer_ram_segura(env)
        estado = {"imagen": obs, "ram": ram, "soy_blanco": (agent_id == "first_0")}
        
        if agent_id == "first_0":
            frame_count += 1
            my_x = int(ram[32]); my_y = int(ram[34])
            opp_x = int(ram[33]); opp_y = int(ram[35])
            dx = opp_x - my_x
            dy = opp_y - my_y
            adx = abs(dx); ady = abs(dy)
            
            # Extreme aggression: ALWAYS walk diagonally into them and punch
            # To corner them, we just hold DIR+FIRE towards their position
            action = 0
            if frame_count % 3 == 0:
                if dy > 0 and dx > 0: action = 16  # DOWNRIGHTFIRE
                elif dy > 0 and dx < 0: action = 17  # DOWNLEFTFIRE
                elif dy < 0 and dx > 0: action = 14  # UPRIGHTFIRE
                elif dy < 0 and dx < 0: action = 15  # UPLEFTFIRE
                elif dx > 0: action = 11  # RIGHTFIRE
                elif dx < 0: action = 12  # LEFTFIRE
                elif dy > 0: action = 13  # DOWNFIRE
                elif dy < 0: action = 10  # UPFIRE
                else: action = 1
            else:
                if dy > 0 and dx > 0: action = 8  # DOWNRIGHT
                elif dy > 0 and dx < 0: action = 9  # DOWNLEFT
                elif dy < 0 and dx > 0: action = 6  # UPRIGHT
                elif dy < 0 and dx < 0: action = 7  # UPLEFT
                elif dx > 0: action = 3  # RIGHT
                elif dx < 0: action = 4  # LEFT
                elif dy > 0: action = 5  # DOWN
                elif dy < 0: action = 2  # UP
                else: action = 0
                
            if reward > 0:
                print(f"Scored! f={frame_count} my_pos=({my_x},{my_y}) opp_pos=({opp_x},{opp_y}) dx={dx} dy={dy}")
        else:
            action = vision_agent.predict(estado)
    
    env.step(action)

env.close()
print(f"Final Score: {scores['first_0']} - {scores['second_0']}")

