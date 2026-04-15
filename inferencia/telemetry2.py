import os
import sys

# Agregamos la ruta base de inferencia al path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from arena import extraer_ram_segura
from pettingzoo.atari import boxing_v2
import time

env = boxing_v2.env(render_mode=None, obs_type="rgb_image")
env.reset()

print("Enviando FIRE para iniciar la partida...")
for i in range(20):
    for agent_id in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        env.step(1)

print("Esperando animacion de inicio...")
for i in range(150):
    for agent_id in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        env.step(0)

ram_inicial = extraer_ram_segura(env)
print(
    f"Inicio: P1 X:{ram_inicial[32]}, Y:{ram_inicial[34]} | P2 X:{ram_inicial[33]}, Y:{ram_inicial[35]}"
)

print("Moviendo agente 1 hacia la derecha, agente 2 se queda quieto...")
for i in range(30):
    for agent_id in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        action = 3 if agent_id == "first_0" else 0
        env.step(action)

ram = extraer_ram_segura(env)
print(f"Derecha: P1 X:{ram[32]}, Y:{ram[34]} | P2 X:{ram[33]}, Y:{ram[35]}")

print("Moviendo agente 1 hacia abajo...")
for i in range(30):
    for agent_id in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        action = 5 if agent_id == "first_0" else 0
        env.step(action)

ram = extraer_ram_segura(env)
print(f"Abajo: P1 X:{ram[32]}, Y:{ram[34]} | P2 X:{ram[33]}, Y:{ram[35]}")

env.close()
