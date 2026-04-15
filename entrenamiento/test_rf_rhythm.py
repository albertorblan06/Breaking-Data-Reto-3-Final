import os, sys, gymnasium as gym, ale_py
from evaluate_candidates import cargar_agente
import numpy as np
gym.register_envs(ale_py)
dir_path = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(dir_path, "submission_rf.py")

for rhythm in [2, 3, 4, 5, 6, 7]:
    env = gym.make("ALE/Boxing-v5", obs_type="rgb")
    agente = cargar_agente(script)
    agente.configurar()
    agente.ruta_modelo_onnx = os.path.join(os.path.dirname(script), "rf_behavioral_clone.onnx")
    
    old_predict = agente.predict
    def p(estado):
        try:
            agente.jab_timer += 1
            ram = estado["ram"].copy()
            soy_blanco = estado["soy_blanco"]
            if not soy_blanco:
                ram[32], ram[33] = ram[33], ram[32]
                ram[34], ram[35] = ram[35], ram[34]
                ram[18], ram[19] = ram[19], ram[18]
                ram[107], ram[109] = ram[109], ram[107]
                ram[111], ram[113] = ram[113], ram[111]
                ram[101], ram[105] = ram[105], ram[101]
                ram[103], ram[105] = ram[105], ram[103]
            outputs = agente.ort_session.run(None, {agente.input_name: np.array([ram], dtype=np.float32)})
            action = int(outputs[0][0])
            if action == 1:
                if agente.jab_timer >= rhythm:
                    agente.jab_timer = 0
                else:
                    action = 0
            return action
        except: return 0
    agente.predict = p
    
    matches = 10
    wins = 0
    ties = 0
    score_diff = 0
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
        elif score == 0: ties += 1
        score_diff += score
    print(f"RF Rhythm {rhythm} Wins: {wins}/{matches}, Ties: {ties}, Avg Score: {score_diff/matches}")
