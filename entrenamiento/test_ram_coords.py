import sys
sys.path.append("../inferencia")
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura
env = boxing_v2.env(obs_type="rgb_image")
env.reset()
for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        break
    ram = extraer_ram_segura(env)
    if agent_id == "first_0":
        action = 5  # DOWN
        print(f"Frame {i}: Y1={ram[34]}, Y2={ram[35]}")
    else:
        action = 0
    env.step(action)
    if i > 20:
        break
