import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import numpy as np

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

for baseline in ["equipo_vision", "equipo_onnx", "equipo_random"]:
    agent = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", baseline))
    agent.configurar()
    
    env = boxing_v2.env(obs_type="rgb_image")
    env.reset()
    
    opp_positions = []
    
    for agent_id in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        
        if term or trunc or len(opp_positions) > 60:
            action = None
            if len(opp_positions) > 60 and not (term or trunc):
                break
        else:
            ram = extraer_ram_segura(env)
            estado = {"imagen": obs, "ram": ram, "soy_blanco": (agent_id == "first_0")}
            
            if agent_id == "first_0":
                my_x = int(ram[32]); my_y = int(ram[34])
                opp_x = int(ram[33]); opp_y = int(ram[35])
                opp_positions.append((opp_x, opp_y))
                
                # Run to corner
                if my_x > 30: action = 4
                elif my_y > 30: action = 2
                else: action = 0
            else:
                action = agent.predict(estado)
        
        env.step(action)
    env.close()
    
    unique_pos = len(set(opp_positions))
    var_x = np.var([p[0] for p in opp_positions])
    var_y = np.var([p[1] for p in opp_positions])
    print(f"{baseline}: Unique pos: {unique_pos}, Var X: {var_x:.1f}, Var Y: {var_y:.1f}")

