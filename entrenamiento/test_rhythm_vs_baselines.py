import sys
import os
import time
import numpy as np

# Use absolute paths to guarantee we can run this from anywhere
ruta_entrenamiento = os.path.dirname(os.path.abspath(__file__))
ruta_inferencia = os.path.abspath(os.path.join(ruta_entrenamiento, "..", "inferencia"))
sys.path.append(ruta_inferencia)

from arena import torneo
import importlib.util


def cargar_mi_agente():
    archivo_submission = os.path.join(ruta_entrenamiento, "submission_rf.py")
    spec = importlib.util.spec_from_file_location("modulo_agente", archivo_submission)
    modulo_agente = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo_agente)
    agente = modulo_agente.AgenteInferencia()
    agente.ruta_modelo = ruta_entrenamiento
    agente.configurar()
    return agente


def test_vs_baseline(baseline_name, matches=3):
    import arena

    a1 = cargar_mi_agente()
    a2 = arena.cargar_agente_desde_carpeta(
        os.path.join(ruta_inferencia, "modelos", baseline_name)
    )
    a2.configurar()

    from pettingzoo.atari import boxing_v2

    env = boxing_v2.env(obs_type="rgb_image")

    print(f"🥊 Rhythm vs {baseline_name} ({matches} matches) 🥊")
    stats = {"wins": 0, "losses": 0, "ties": 0}

    for _ in range(matches):
        env.reset()
        recompensas = {"first_0": 0, "second_0": 0}

        for agent_id in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            recompensas[agent_id] += reward

            if term or trunc:
                action = None
            else:
                ram = arena.extraer_ram_segura(env)
                estado = {
                    "imagen": obs,
                    "ram": ram,
                    "soy_blanco": (agent_id == "first_0"),
                }
                if agent_id == "first_0":
                    action = a1.predict(estado)
                else:
                    action = a2.predict(estado)
            env.step(action)

        if recompensas["first_0"] > recompensas["second_0"]:
            stats["wins"] += 1
        elif recompensas["second_0"] > recompensas["first_0"]:
            stats["losses"] += 1
        else:
            stats["ties"] += 1

        print(
            f"Match: Rhythm {recompensas['first_0']} - {recompensas['second_0']} {baseline_name}"
        )

    print(f"Resultados: {stats}\n")


if __name__ == "__main__":
    test_vs_baseline("equipo_random", matches=5)
    test_vs_baseline("equipo_onnx", matches=5)
    test_vs_baseline("equipo_vision", matches=5)
