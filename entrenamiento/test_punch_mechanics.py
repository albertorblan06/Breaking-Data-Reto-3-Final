import sys
sys.path.append("../inferencia")
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura

# Test: Approach opponent, get to close range, then spam FIRE with different strategies
# Goal: Find what works to actually score
env = boxing_v2.env(obs_type="rgb_image")
env.reset()

score_us = 0
last_dist = 999
min_dist_seen = 999
hit_frames = []
close_frames = []

action_name = {
    0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
    6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT",
    10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE",
    14: "UPRIGHTFIRE", 15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE"
}

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if agent_id == "first_0":
        score_us += reward
    if term or trunc or i > 4000:
        break
    
    ram = extraer_ram_segura(env)
    my_x = int(ram[32])
    my_y = int(ram[34])
    opp_x = int(ram[33])
    opp_y = int(ram[35])
    dist_x = abs(opp_x - my_x)
    dist_y = abs(opp_y - my_y)
    
    if min_dist_seen > dist_x:
        min_dist_seen = dist_x
    
    if agent_id == "first_0":
        # Phase 1: Approach (first 500 steps)
        if i < 500:
            if dist_y > 3:
                if opp_y > my_y: action = 8  # DOWNRIGHT
                else: action = 6  # UPRIGHT
            elif dist_x > 14:
                if opp_x > my_x: action = 3  # RIGHT
                else: action = 4  # LEFT
            else:
                action = 11  # RIGHTFIRE toward opponent
            
            # Log when close
            if dist_x <= 16 and dist_y <= 4:
                close_frames.append((i, my_x, my_y, opp_x, opp_y, dist_x, dist_y, action))
        
        # Phase 2: Spam FIRE when close (500-2000)
        elif i < 2000:
            if dist_y > 2:
                if opp_y > my_y: action = 8  # DOWNRIGHT
                elif opp_y < my_y: action = 6  # UPRIGHT
                else: action = 11  # RIGHTFIRE
            else:
                # Alternate: move toward and fire
                if i % 2 == 0:
                    # Just fire
                    action = 1  # plain FIRE
                else:
                    # Move toward opponent and fire
                    if opp_x > my_x: action = 11  # RIGHTFIRE
                    else: action = 12  # LEFTFIRE
            
            if reward > 0:
                hit_frames.append((i, my_x, my_y, opp_x, opp_y, dist_x, dist_y, score_us))
        
        # Phase 3: Try different strategies
        else:
            # Just move right toward opponent and spam fire
            if opp_x > my_x:
                action = 11  # RIGHTFIRE  
            else:
                action = 12  # LEFTFIRE
            if reward > 0:
                hit_frames.append((i, my_x, my_y, opp_x, opp_y, dist_x, dist_y, score_us))
    else:
        action = 0  # Opponent does NOTHING
    
    env.step(action)

print(f"Final score: {score_us}")
print(f"Minimum dist_x seen: {min_dist_seen}")
print(f"Hit frames: {hit_frames}")
if close_frames:
    print(f"\nClose-range frames (first 20):")
    for f in close_frames[:20]:
        print(f"  Frame {f[0]}: me=({f[1]},{f[2]}) opp=({f[3]},{f[4]}) dist=({f[5]},{f[6]}) action={action_name[f[7]]}")
env.close()
