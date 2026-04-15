import sys, os, time, importlib.util
import numpy as np, gymnasium as gym, ale_py
from evaluate_candidates import cargar_agente

def test_rhythm(script, rhythm_threshold, matches=10):
    env = gym.make("ALE/Boxing-v5", obs_type="rgb")
    agente = cargar_agente(script)
    agente.configurar()
    agente.ruta_modelo_onnx = os.path.join(os.path.dirname(script), "modelo.onnx")
    
    agente.rhythm_threshold = rhythm_threshold
    
    old_predict = agente.predict
    def new_predict(estado):
        try:
            ram = estado["ram"].copy()
            soy_blanco = estado["soy_blanco"]
            agente.jab_timer += 1
            if not soy_blanco:
                ram[32], ram[33] = ram[33], ram[32]
                ram[34], ram[35] = ram[35], ram[34]
                ram[18], ram[19] = ram[19], ram[18]
                ram[107], ram[109] = ram[109], ram[107]
                ram[111], ram[113] = ram[113], ram[111]
                ram[101], ram[105] = ram[105], ram[101]
                ram[103], ram[105] = ram[105], ram[103]
            obs = np.array([ram], dtype=np.uint8)
            outputs = agente.ort_session.run(None, {agente.input_name: obs})
            action_logits = outputs[0]
            action = int(np.argmax(action_logits, axis=1)[0])
            if action == 1:
                if agente.jab_timer >= agente.rhythm_threshold:
                    agente.jab_timer = 0
                else:
                    action = 0
            return action
        except: return 0

    agente.predict = new_predict

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
        if score_match > 0: stats["wins"] += 1
        elif score_match < 0: stats["losses"] += 1
        else: stats["ties"] += 1
    
    print(f"Rhythm {rhythm_threshold} -> Wins: {stats['wins']}, Ties: {stats['ties']}, Losses: {stats['losses']}, Avg: {stats['total_score'] / matches:.2f}")

if __name__ == "__main__":
    gym.register_envs(ale_py)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(dir_path, "submission_aggressive.py")
    test_rhythm(script, rhythm_threshold=5, matches=5)
    test_rhythm(script, rhythm_threshold=6, matches=5)
    test_rhythm(script, rhythm_threshold=7, matches=5)
