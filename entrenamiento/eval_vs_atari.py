import sys
import os
import time
import numpy as np
import gymnasium as gym
import ale_py

# Add inferencia to the path so we can import the agent class and the interface
ruta_inferencia = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "inferencia")
)
sys.path.append(ruta_inferencia)

# We need to load the module from modelos/Aquatic_Agents/submission.py dynamically
import importlib.util


def cargar_agente_desde_carpeta(ruta_carpeta):
    archivo_submission = os.path.join(ruta_carpeta, "submission.py")
    if not os.path.exists(archivo_submission):
        raise FileNotFoundError(f"No se encontró submission.py en {ruta_carpeta}")

    spec = importlib.util.spec_from_file_location("modulo_agente", archivo_submission)
    modulo_agente = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo_agente)

    if not hasattr(modulo_agente, "AgenteInferencia"):
        raise AttributeError(
            f"El archivo {archivo_submission} no define la clase 'AgenteInferencia'"
        )

    return modulo_agente.AgenteInferencia()


def extraer_ram_segura(env):
    unwrapped = env.unwrapped
    try:
        return unwrapped.ale.getRAM()
    except AttributeError:
        return np.zeros(128, dtype=np.uint8)


def evaluate_against_atari(matches=10):
    gym.register_envs(ale_py)
    env = gym.make("ALE/Boxing-v5", obs_type="rgb")

    ruta_modelo = os.path.join(ruta_inferencia, "modelos", "Aquatic_Agents")
    agente = cargar_agente_desde_carpeta(ruta_modelo)

    # Need to call configurar because the arena does it before running matches
    agente.configurar()

    print(f"🥊 Iniciando torneo ({matches} matches): Aquatic_Agents vs IA de ATARI 🥊")

    stats = {
        "wins": 0,
        "total_score": 0,
    }
    ties = 0

    for match in range(matches):
        obs, info = env.reset()
        score_match = 0
        done = False

        while not done:
            ram = extraer_ram_segura(env)
            estado_completo = {
                "imagen": obs,
                "ram": ram,
                "soy_blanco": True,  # In ALE/Boxing-v5, the agent is always P1 (White)
            }

            action = agente.predict(estado_completo)

            obs, reward, term, trunc, info = env.step(action)
            score_match += reward
            done = term or trunc

        stats["total_score"] += score_match

        if score_match > 0:
            stats["wins"] += 1
            result = "WIN P1"
        elif score_match < 0:
            result = "WIN ATARI"
        else:
            ties += 1
            result = "TIE"

        print(f"Match {match + 1}/{matches} | Score: {score_match} | {result}")

    print("\n📊 --- RESULTADOS FINALES ---")
    print(f"Matches: {matches} | Ties: {ties}")
    print(f"Aquatic_Agents (White/P1):")
    print(f"  Wins: {stats['wins']}")
    print(f"  Avg Score (Agent vs Atari): {stats['total_score'] / matches:.2f}")


if __name__ == "__main__":
    evaluate_against_atari(matches=100)
