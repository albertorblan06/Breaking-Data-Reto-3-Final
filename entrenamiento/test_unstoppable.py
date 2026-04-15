import os, sys, gymnasium as gym, ale_py
from evaluate_candidates import cargar_agente
import numpy as np
gym.register_envs(ale_py)
dir_path = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(dir_path, "submission_unstoppable.py")

env = gym.make("ALE/Boxing-v5", obs_type="rgb")
agente = cargar_agente(script)
agente.configurar()
agente.ruta_modelo_onnx = os.path.join(os.path.dirname(script), "modelo.onnx")

obs, _ = env.reset()
done = False
steps = 0
while not done and steps < 20:
    ram = env.unwrapped.ale.getRAM()
    
    # Run through ONNX directly
    outputs = agente.ort_session.run(None, {agente.input_name: np.array([ram], dtype=np.uint8)})
    raw_action = int(np.argmax(outputs[0], axis=1)[0])
    
    action = agente.predict({"imagen": obs, "ram": ram, "soy_blanco": True})
    obs, reward, term, trunc, _ = env.step(action)
    print(f"Step {steps}, raw_action {raw_action}, predict returns {action}")
    steps += 1

