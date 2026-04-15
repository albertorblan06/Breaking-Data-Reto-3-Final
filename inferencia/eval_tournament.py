import sys, os, importlib.util
from arena import torneo

def run_large_tournament(equipo_1, equipo_2, matches=10):
    print(f"🥊 {equipo_1} vs {equipo_2} ({matches} matches) 🥊")
    stats = {"e1": 0, "e2": 0, "ties": 0}
    import arena
    a1 = arena.cargar_agente_desde_carpeta(f"modelos/{equipo_1}")
    a2 = arena.cargar_agente_desde_carpeta(f"modelos/{equipo_2}")
    from pettingzoo.atari import boxing_v2
    env = boxing_v2.env(obs_type="rgb_image")
    ias = {'first_0': a1, 'second_0': a2}
    for _ in range(matches):
        env.reset()
        recompensas = {'first_0': 0, 'second_0': 0}
        for agent_id in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            recompensas[agent_id] += reward
            if term or trunc: action = None
            else:
                ram = arena.extraer_ram_segura(env)
                try: action = ias[agent_id].predict({"imagen": obs, "ram": ram, "soy_blanco": (agent_id == 'first_0')})
                except: action = arena.ACCION_PENALIZACION
            env.step(action)
        if recompensas['first_0'] > recompensas['second_0']: stats["e1"] += 1
        elif recompensas['second_0'] > recompensas['first_0']: stats["e2"] += 1
        else: stats["ties"] += 1
    print(f"Stats: {stats}")

if __name__ == "__main__":
    run_large_tournament("Aquatic_Agents", "equipo_random", matches=10)
    run_large_tournament("Aquatic_Agents", "equipo_onnx", matches=5)
    run_large_tournament("Aquatic_Agents", "equipo_vision", matches=5)
