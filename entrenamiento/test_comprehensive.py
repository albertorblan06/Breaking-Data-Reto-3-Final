import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import importlib.util
import time

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

def run_matches(baseline_name, num_matches=10):
    wins = 0
    losses = 0
    ties = 0
    total_w = 0
    total_b = 0
    
    for i in range(num_matches):
        # Reload agent fresh each match
        spec = importlib.util.spec_from_file_location("mod_" + str(i), os.path.join(ruta_ent, "submission_rf.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        a1 = mod.AgenteInferencia()
        a1.ruta_modelo = ruta_ent
        a1.configurar()
        
        a2 = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", baseline_name))
        a2.configurar()
        
        env = boxing_v2.env(obs_type="rgb_image")
        env.reset()
        
        scores = {'first_0': 0, 'second_0': 0}
        max_lat = 0
        
        for agent_id in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            scores[agent_id] += reward
            
            if term or trunc:
                action = None
            else:
                ram = extraer_ram_segura(env)
                estado = {"imagen": obs, "ram": ram, "soy_blanco": (agent_id == "first_0")}
                t_start = time.perf_counter()
                if agent_id == "first_0":
                    action = a1.predict(estado)
                else:
                    action = a2.predict(estado)
                lat = (time.perf_counter() - t_start) * 1000
                max_lat = max(max_lat, lat)
            
            env.step(action)
        
        env.close()
        
        w_score = scores['first_0']
        b_score = scores['second_0']
        total_w += w_score
        total_b += b_score
        
        if w_score > b_score:
            result = "WIN"
            wins += 1
        elif b_score > w_score:
            result = "LOSS"
            losses += 1
        else:
            result = "TIE"
            ties += 1
        
        print(f"  Match {i+1}: {result} W={w_score:.0f} B={b_score:.0f}")
    
    print(f"\n=== {baseline_name} Summary ({num_matches} matches) ===")
    print(f"Wins: {wins}, Losses: {losses}, Ties: {ties}")
    print(f"Avg score: W={total_w/num_matches:.1f} B={total_b/num_matches:.1f}")
    print(f"Max latency: {max_lat:.2f}ms")
    return wins, losses, ties

print("=== Testing against equipo_random ===")
w, l, t = run_matches("equipo_random", 10)

print("\n=== Testing against equipo_onnx ===")
w2, l2, t2 = run_matches("equipo_onnx", 10)

print("\n=== Testing against equipo_vision ===")
w3, l3, t3 = run_matches("equipo_vision", 10)

print("\n=== OVERALL SUMMARY ===")
print(f"Random: {w}W {l}L {t}T")
print(f"ONNX: {w2}W {l2}L {t2}T")
print(f"Vision: {w3}W {l3}L {t3}T")
