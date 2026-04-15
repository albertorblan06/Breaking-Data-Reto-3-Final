import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura

# Test: Both players approach each other and BOTH spam FIRE
env = boxing_v2.env(obs_type="rgb_image")
env.reset()

score_w = 0
score_b = 0

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if agent_id == "first_0":
        score_w += reward
    else:
        score_b += reward
    
    if term or trunc or i > 8000:
        break
    
    ram = extraer_ram_segura(env)
    my_x = int(ram[32])
    my_y = int(ram[34])
    opp_x = int(ram[33])
    opp_y = int(ram[35])
    dist_x = abs(opp_x - my_x)
    dist_y = abs(opp_y - my_y)
    
    if agent_id == "first_0":
        # Approach then FIRE
        if dist_y > 3:
            if opp_y > my_y: action = 8  # DOWNRIGHT
            elif opp_y < my_y: action = 6  # UPRIGHT
            else: action = 3  # RIGHT
        elif dist_x > 14:
            action = 3  # RIGHT
        else:
            action = 1  # FIRE
    else:
        # Opponent also approaches and also FIREs
        if dist_y > 3:
            if my_y > opp_y: action = 7  # UPLEFT  (move toward player 1)
            elif my_y < opp_y: action = 9  # DOWNLEFT
            else: action = 4  # LEFT
        elif dist_x > 14:
            action = 4  # LEFT
        else:
            action = 1  # FIRE
    
    if i % 500 == 0:
        print(f"Frame {i}: W=({int(ram[32])},{int(ram[34])}) B=({int(ram[33])},{int(ram[35])}) dist=({dist_x},{dist_y}) ram18={int(ram[18])} ram19={int(ram[19])} reward_w={score_w} reward_b={score_b}")
    
    env.step(action)

print(f"\nFinal: W={score_w} B={score_b}, RAM: W={int(ram[18])} B={int(ram[19])}")
env.close()
