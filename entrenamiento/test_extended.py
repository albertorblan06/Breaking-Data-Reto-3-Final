import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import importlib.util
import time

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

# Load our agent
archivo = os.path.join(ruta_ent, "submission_rf.py")
spec = importlib.util.spec_from_file_location("modulo_agente", archivo)
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)

def run_match(agent_a_name, agent_b_name):
    """Run a single match between two agents and return (a_score, b_score)"""
    # Reload agent each match for fresh state
    spec2 = importlib.util.spec_from_file_location("modulo_agente2", archivo)
    modulo2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(modulo2)
    
    a1 = modulo.AgenteInferencia()
    a1.ruta_modelo = ruta_ent
    a1.configurar()
    
    a2 = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", agent_b_name))
    a2.configurar()
    
    env = boxing_v2.env(obs_type="rgb_image")
    env.reset()
    
    scores = {'first_0': 0, 'second_0': 0}
    latency_ms = []
    
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
            latency_ms.append((time.perf_counter() - t_start) * 1000)
        
        env.step(action)
    
    env.close()
    max_latency = max(latency_ms) if latency_ms else 0
    avg_latency = sum(latency_ms) / len(latency_ms) if latency_ms else 0
    return scores['first_0'], scores['second_0'], max_latency, avg_latency

# Run 10 matches per baseline
for baseline in ["equipo_random"]:
    wins = 0
    losses = 0
    ties = 0
    total_w_score = 0
    total_b_score = 0
    
    for i in range(10):
        w_score, b_score, max_lat, avg_lat = run_match("Aquatic_Agents", baseline)
        if w_score > b_score:
            result = "WIN"
            wins += 1
        elif b_score > w_score:
            result = "LOSS"
            losses += 1
        else:
            result = "TIE"
            ties += 1
        
        total_w_score += w_score
        total_b_score += b_score
        print(f"Match {i+1}: {result} W={w_score:.0f} B={b_score:.0f} max_lat={max_lat:.2f}ms avg_lat={avg_lat:.2f}ms")
    
    print(f"\n=== {baseline} Summary ===")
    print(f"Wins: {wins}/{10}, Losses: {losses}/{10}, Ties: {ties}/{10}")
    print(f"Avg score: W={total_w_score/10:.1f} B={total_b_score/10:.1f}")

