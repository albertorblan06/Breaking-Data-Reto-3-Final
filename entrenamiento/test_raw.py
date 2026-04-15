import os, sys, gymnasium as gym, ale_py
from evaluate_candidates import cargar_agente
import numpy as np
gym.register_envs(ale_py)
dir_path = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(dir_path, "submission_rf.py")

env = gym.make("ALE/Boxing-v5", obs_type="rgb")
agente = cargar_agente(script)
agente.configurar()
agente.ruta_modelo_onnx = os.path.join(os.path.dirname(script), "modelo.onnx")

old_predict = agente.predict
def new_p(estado):
    ram = estado["ram"].copy()
    if not estado["soy_blanco"]:
        ram[32], ram[33] = ram[33], ram[32]
        ram[34], ram[35] = ram[35], ram[34]
        ram[18], ram[19] = ram[19], ram[18]
        ram[107], ram[109] = ram[109], ram[107]
        ram[111], ram[113] = ram[113], ram[111]
        ram[101], ram[105] = ram[105], ram[101]
        ram[103], ram[105] = ram[105], ram[103]
    outputs = agente.ort_session.run(None, {agente.input_name: np.array([ram], dtype=np.float32)})
    return int(outputs[0][0])
agente.predict = new_p

matches = 10
wins = 0
for match in range(matches):
    obs, _ = env.reset()
    score = 0
    done = False
    while not done:
        ram = env.unwrapped.ale.getRAM()
        action = agente.predict({"imagen": obs, "ram": ram, "soy_blanco": True})
        obs, reward, term, trunc, _ = env.step(action)
        score += reward
        done = term or trunc
    if score > 0: wins += 1
    print(f"Match score: {score}")
print(f"Raw action wins: {wins}/{matches}")
