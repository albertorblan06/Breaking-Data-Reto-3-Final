import sys
import os
import gymnasium as gym
import numpy as np

ruta_entrenamiento = os.path.dirname(os.path.abspath(__file__))
ruta_inferencia = os.path.abspath(os.path.join(ruta_entrenamiento, "..", "inferencia"))
sys.path.append(ruta_inferencia)

from pettingzoo.atari import boxing_v2
import arena

# Let's find exactly what dist_x connects punches
best_score = -999

for test_x in range(15, 30):
    for test_y in range(-2, 3):
        env = boxing_v2.env(obs_type="rgb_image")
        env.reset()

        a2 = arena.cargar_agente_desde_carpeta(
            os.path.join(ruta_inferencia, "modelos", "equipo_onnx")
        )
        a2.configurar()

        steps = 0
        score = 0
        jab_timer = 0

        for agent_id in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if agent_id == "first_0":
                score += reward

            if term or trunc or steps > 200:
                if agent_id == "second_0" and term or trunc:
                    pass
                if steps > 200:
                    env.close()
                    break
                action = None
                env.step(action)
                continue

            ram = arena.extraer_ram_segura(env)
            mi_x, mi_y = ram[32], ram[34]
            su_x, su_y = ram[33], ram[35]

            if agent_id == "first_0":
                target_y = int(su_y) + test_y
                dist_y = abs(target_y - int(mi_y))
                dist_x = abs(int(su_x) - int(mi_x))

                action = 0  # NOOP
                if int(mi_y) < target_y:
                    action = 5  # DOWN
                elif int(mi_y) > target_y:
                    action = 2  # UP
                elif dist_x > test_x:
                    action = 3 if int(su_x) > int(mi_x) else 4
                elif dist_x < test_x:
                    action = 4 if int(su_x) > int(mi_x) else 3
                else:
                    action = 1  # ALWAYS PUNCH

                env.step(action)
                steps += 1
            else:
                action = a2.predict({"imagen": obs, "ram": ram, "soy_blanco": False})
                env.step(action)

        if score > best_score:
            best_score = score
        if score > 0:
            print(f"BINGO! Test X={test_x}, Y={test_y}: Score = {score}")
