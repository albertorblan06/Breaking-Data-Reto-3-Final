import sys
import os
import numpy as np

ruta_entrenamiento = os.path.dirname(os.path.abspath(__file__))
ruta_inferencia = os.path.abspath(os.path.join(ruta_entrenamiento, "..", "inferencia"))
sys.path.append(ruta_inferencia)

import arena
import importlib.util


def cargar_mi_agente():
    archivo_submission = os.path.join(
        ruta_entrenamiento, "submission_aggressive_rhythm.py"
    )
    spec = importlib.util.spec_from_file_location("modulo_agente", archivo_submission)
    modulo_agente = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo_agente)
    agente = modulo_agente.AgenteInferencia()
    agente.ruta_modelo = ruta_entrenamiento
    agente.configurar()
    return agente


from pettingzoo.atari import boxing_v2

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

a1 = cargar_mi_agente()
a2 = arena.cargar_agente_desde_carpeta(
    os.path.join(ruta_inferencia, "modelos", "equipo_onnx")
)
a2.configurar()

print("Running match and tracking distances...")

steps = 0
for agent_id in env.agent_iter():
    obs, reward, term, trunc, info = env.last()

    if term or trunc:
        action = None
    else:
        ram = arena.extraer_ram_segura(env)

        mi_x, mi_y = ram[32], ram[34]
        su_x, su_y = ram[33], ram[35]

        if agent_id == "first_0":
            dist_x = abs(int(su_x) - int(mi_x))
            dist_y = abs(int(su_y) - int(mi_y))
            if steps % 50 == 0:
                print(
                    f"Step {steps}: P1({mi_x},{mi_y}) P2({su_x},{su_y}) Dist=({dist_x},{dist_y})"
                )

            estado = {"imagen": obs, "ram": ram, "soy_blanco": True}
            action = a1.predict(estado)
            steps += 1
        else:
            estado = {"imagen": obs, "ram": ram, "soy_blanco": False}
            action = a2.predict(estado)

    env.step(action)
    if steps > 300:
        break
