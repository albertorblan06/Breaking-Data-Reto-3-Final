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
        if frame_count > 1000 and not (term or trunc):
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
            
            # Extreme bait: Just run to the far left corner (x=30, y=30)
            target_x = 30
            target_y = 30
            
            # If we are in the corner, wait. Log opponent position to see if they follow.
            if frame_count % 60 == 0:
                print(f"f={frame_count} Us:({my_x},{my_y}) Vision:({opp_x},{opp_y}) dx={dx}")
            
            if my_x > target_x: action = 4 # LEFT
            elif my_y > target_y: action = 2 # UP
            else: action = 0 # NOOP
                
        else:
            action = vision_agent.predict(estado)
    
    env.step(action)

env.close()
