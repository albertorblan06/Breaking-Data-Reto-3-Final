import sys
import os

ruta_entrenamiento = os.path.dirname(os.path.abspath(__file__))
ruta_inferencia = os.path.abspath(os.path.join(ruta_entrenamiento, "..", "inferencia"))
sys.path.append(ruta_inferencia)

from pettingzoo.atari import boxing_v2
import arena

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

a1 = arena.cargar_agente_desde_carpeta(
    os.path.join(ruta_inferencia, "modelos", "equipo_random")
)
a1.configurar()

a2 = arena.cargar_agente_desde_carpeta(
    os.path.join(ruta_inferencia, "modelos", "equipo_onnx")
)
a2.configurar()

score = 0
for agent_id in env.agent_iter():
    obs, reward, term, trunc, info = env.last()

    if agent_id == "first_0":
        score += reward

    if term or trunc:
        action = None
    else:
        ram = arena.extraer_ram_segura(env)
        if agent_id == "first_0":
            action = a1.predict({"imagen": obs, "ram": ram, "soy_blanco": True})
        else:
            action = a2.predict({"imagen": obs, "ram": ram, "soy_blanco": False})

    env.step(action)

print("Final Score Random vs ONNX:", score)
