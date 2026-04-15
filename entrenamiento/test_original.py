import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

# Original aquatic agents
agent = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "Aquatic_Agents"))
agent.configurar()

# Random baseline
random_agent = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_random"))
random_agent.configurar()

wins, losses, ties = 0, 0, 0
for i in range(10):
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
            if agent_id == "first_0": action = agent.predict(estado)
            else: action = random_agent.predict(estado)
        env.step(action)
    env.close()
    
    w = scores['first_0']; b = scores['second_0']
    if w > b: wins += 1
    elif b > w: losses += 1
    else: ties += 1
    print(f"Match {i+1}: Us {w} - {b} Random")
print(f"Summary: {wins}W {losses}L {ties}T")

