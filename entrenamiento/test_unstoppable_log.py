import os, sys, gymnasium as gym, ale_py, numpy as np
from evaluate_candidates import cargar_agente
gym.register_envs(ale_py)
script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_unstoppable.py")

env = gym.make("ALE/Boxing-v5", obs_type="rgb")
agente = cargar_agente(script)
agente.configurar()
agente.ruta_modelo_onnx = os.path.join(os.path.dirname(script), "modelo.onnx")

obs, _ = env.reset()
done = False
for i in range(20):
    ram = env.unwrapped.ale.getRAM()
    action = agente.predict({"imagen": obs, "ram": ram, "soy_blanco": True})
    
    mi_x, mi_y = ram[32], ram[34]
    su_x, su_y = ram[33], ram[35]
    
    obs, reward, term, trunc, _ = env.step(action)
    print(f"Frame {i:02d}: mi_x={mi_x:03d} mi_y={mi_y:03d} | su_x={su_x:03d} su_y={su_y:03d} | Action={action}")
