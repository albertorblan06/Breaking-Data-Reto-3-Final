import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura
import importlib.util
import numpy as np

# Load our agent 
ruta_ent = os.path.dirname(os.path.abspath(__file__))
archivo = os.path.join(ruta_ent, "submission_rf.py")
spec = importlib.util.spec_from_file_location("modulo_agente", archivo)
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)
a1 = modulo.AgenteInferencia()
a1.ruta_modelo = ruta_ent
a1.configurar()

ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))
import arena

# Test against random, tracking RAM score values
for opponent_name in ["equipo_random"]:
    print(f"\n=== Testing vs {opponent_name} ===")
    a2 = arena.cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", opponent_name))
    a2.configurar()
    
    env = boxing_v2.env(obs_type="rgb_image")
    env.reset()
    
    cumulative_p1 = 0
    cumulative_p2 = 0
    
    for i, agent_id in enumerate(env.agent_iter()):
        obs, reward, term, trunc, info = env.last()
        
        if agent_id == "first_0":
            cumulative_p1 += reward
        else:
            cumulative_p2 += reward
        
        if term or trunc:
            action = None
        else:
            ram = extraer_ram_segura(env)
            estado = {"imagen": obs, "ram": ram, "soy_blanco": True}
            
            # Every 500 frames, check RAM scores and cumulative reward
            if i % 500 == 0:
                print(f"  Frame {i}: RAM scores: W={ram[18]}, B={ram[19]} | cumul_reward: P1={cumulative_p1:.1f}, P2={cumulative_p2:.1f} | dist=({abs(int(ram[33])-int(ram[32]))},{abs(int(ram[35])-int(ram[34]))})")
            
            if agent_id == "first_0":
                action = a1.predict(estado)
            else:
                estado2 = {"imagen": obs, "ram": ram, "soy_blanco": False}
                action = a2.predict(estado2)
        
        env.step(action)
        
        if term or trunc:
            # Final state
            ram_final = extraer_ram_segura(env) if not hasattr(env, '_closed') else None
            if ram_final is not None:
                print(f"  FINAL: RAM scores: W={ram_final[18]}, B={ram_final[19]} | cumul_reward: P1={cumulative_p1:.1f}, P2={cumulative_p2:.1f}")
            break
    
    env.close()
