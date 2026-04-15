import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import numpy as np

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

vision_agent = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_vision"))
vision_agent.configurar()

def test_strategy(target_dx, target_dy, punch_action, max_frames=1000):
    env = boxing_v2.env(obs_type="rgb_image")
    env.reset()
    
    scores = {'first_0': 0, 'second_0': 0}
    frame_count = 0
    
    for agent_id in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        scores[agent_id] += reward
        
        if term or trunc or frame_count > max_frames:
            action = None
            if frame_count > max_frames and not (term or trunc):
                # Force end
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
                
                # Move to target
                if adx > target_dx or ady > target_dy:
                    if ady > target_dy:
                        action = 5 if dy > 0 else 2
                    else:
                        action = 3 if dx > 0 else 4
                else:
                    # In position, execute punch action (with rhythm)
                    if frame_count % 3 == 0:
                        if punch_action == 'FIRE': action = 1
                        elif punch_action == 'DIRFIRE':
                            if dx > 0: action = 11
                            else: action = 12
                        else:
                            action = punch_action
                    else:
                        action = 0
            else:
                action = vision_agent.predict(estado)
        
        env.step(action)
    
    env.close()
    return scores['first_0'], scores['second_0']

print("Testing grid against Vision...")
for t_dx in [14, 16, 18, 20]:
    for t_dy in [0, 2, 5, 8]:
        for p_act in ['FIRE', 'DIRFIRE']:
            w, b = test_strategy(t_dx, t_dy, p_act, max_frames=800)
            if w > 0 or b > 0:
                print(f"dx={t_dx} dy={t_dy} act={p_act} -> Us: {w}, Vision: {b}")

