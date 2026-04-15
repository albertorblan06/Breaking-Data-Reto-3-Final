import sys
import os
import time
import importlib.util
import numpy as np
import gymnasium as gym
import ale_py

ruta_inferencia = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "inferencia")
)
sys.path.append(ruta_inferencia)


def cargar_agente(ruta_submission):
    spec = importlib.util.spec_from_file_location("modulo_agente", ruta_submission)
    modulo_agente = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo_agente)
    agente = modulo_agente.AgenteInferencia()
    # Mock 'ruta_modelo' so the submission script looks in the same directory for its .onnx
    agente.ruta_modelo = os.path.dirname(ruta_submission)
    return agente


def evaluate_model(submission_script, matches=10):
    env = gym.make("ALE/Boxing-v5", obs_type="rgb")

    agente = cargar_agente(submission_script)
    agente.configurar()

    if hasattr(agente, "ruta_modelo_onnx"):
        agente.ruta_modelo_onnx = os.path.join(
            os.path.dirname(submission_script),
            os.path.basename(agente.ruta_modelo_onnx),
        )

    print(
        f"\n🥊 Evaluando {os.path.basename(submission_script)} ({matches} matches) 🥊"
    )
    stats = {"wins": 0, "losses": 0, "ties": 0, "total_score": 0}

    for match in range(matches):
        obs, _ = env.reset()
        score_match = 0
        done = False

        while not done:
            ram = env.unwrapped.ale.getRAM()
            estado = {"imagen": obs, "ram": ram, "soy_blanco": True}
            action = agente.predict(estado)
            obs, reward, term, trunc, _ = env.step(action)
            score_match += reward
            done = term or trunc

        stats["total_score"] += score_match
        if score_match > 0:
            stats["wins"] += 1
        elif score_match < 0:
            stats["losses"] += 1
        else:
            stats["ties"] += 1

        print(f"Match {match + 1}/{matches}: Score {score_match}")

    print(f"\nResultados para {os.path.basename(submission_script)}:")
    print(
        f"  Wins: {stats['wins']} | Losses: {stats['losses']} | Ties: {stats['ties']}"
    )
    print(f"  Avg Score Diff: {stats['total_score'] / matches:.2f}")


if __name__ == "__main__":
    gym.register_envs(ale_py)
    # We use __file__ dir to ensure absolute paths are correct no matter where we call it from
    dir_path = os.path.dirname(os.path.abspath(__file__))
    evaluate_model(os.path.join(dir_path, "submission_rhythm.py"), matches=5)
