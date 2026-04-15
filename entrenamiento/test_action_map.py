import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura

action_names = {
    0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
    6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT",
    10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE",
    14: "UPRIGHTFIRE", 15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE"
}

# Test actions and also check: can we EVER get closer than dist_x=14?
env = boxing_v2.env(obs_type="rgb_image")
env.reset()
min_dist_x_seen = 999
min_dist_x_noop = 999

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if term or trunc or i > 20000:
        break
    
    ram = extraer_ram_segura(env)
    my_x, my_y = int(ram[32]), int(ram[34])
    opp_x, opp_y = int(ram[33]), int(ram[35])
    dist_x = abs(opp_x - my_x)
    dist_y = abs(opp_y - my_y)
    
    if agent_id == "first_0":
        # Approach opponent and get close, then try different actions
        # Phase 1: Approach (first 500 steps)
        if i < 500:
            if dist_y > 2:
                if opp_y > my_y: action = 8  # DOWNRIGHT
                else: action = 6  # UPRIGHT
            else:
                if opp_x > my_x: action = 3  # RIGHT
                else: action = 4  # LEFT
        # Phase 2: When in range, alternate between FIRE and RIGHT
        else:
            # Try to close the X distance by walking RIGHT into opponent
            if i % 3 == 0:
                action = 1  # FIRE
            else:
                if opp_x > my_x: action = 3  # RIGHT  
                elif opp_x < my_x: action = 4  # LEFT
                else: action = 1  # FIRE
    else:
        # Opponent walks toward us (LEFT)
        action = 4  # LEFT
    
    if dist_x < min_dist_x_seen:
        min_dist_x_seen = dist_x
    
    if i % 1000 == 0 and i > 0:
        print(f"Frame {i}: me=({my_x},{my_y}) opp=({opp_x},{opp_y}) dist=({dist_x},{dist_y}) min_dist_x={min_dist_x_seen}")

    env.step(action)

ram = extraer_ram_segura(env)
print(f"\nFinal: W_Score={int(ram[18])} B_Score={int(ram[19])} Min dist_x seen: {min_dist_x_seen}")
env.close()
