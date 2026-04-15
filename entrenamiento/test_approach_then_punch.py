import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura

# Test: Opponent walks TOWARD us, we punch. Can we score?
env = boxing_v2.env(obs_type="rgb_image")
env.reset()

score = 0
frame = 0

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if agent_id == "first_0":
        score += reward
    
    if term or trunc or i > 8000:
        break
    
    ram = extraer_ram_segura(env)
    my_x, my_y = int(ram[32]), int(ram[34])
    opp_x, opp_y = int(ram[33]), int(ram[35])
    dist_x = abs(opp_x - my_x)
    dist_y = abs(opp_y - my_y)
    
    if agent_id == "first_0":
        # We stand still and ONLY punch when close
        if dist_x <= 20 and dist_y <= 5:
            action = 1  # Pure FIRE
        elif dist_y >= 3:
            if opp_y > my_y: action = 5  # DOWN
            else: action = 2  # UP
        elif dist_x > 20:
            if opp_x > my_x: action = 3  # RIGHT
            else: action = 4  # LEFT
        else:
            action = 1  # FIRE when close
        frame += 1
    else:
        # Opponent walks STRAIGHT TOWARD US
        if dist_y > 3:
            if my_y > opp_y: action = 5  # DOWN (toward us)
            else: action = 2  # UP
        elif my_x < opp_x:
            action = 4  # LEFT (toward us)
        else:
            action = 3  # RIGHT (toward us)
    
    if i % 500 == 0:
        print(f"Frame {i}: me=({my_x},{my_y}) opp=({opp_x},{opp_y}) dist=({dist_x},{dist_y}) score={score}")
    
    env.step(action)

print(f"\nFinal score: {score}")
env.close()
