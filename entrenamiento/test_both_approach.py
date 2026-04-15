import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura

# Both players walk TOWARD each other, then FIRE when very close
env = boxing_v2.env(obs_type="rgb_image")
env.reset()

prev_opp_x = 109
prev_opp_y = 87
score_w = 0
min_dist = 999
hit_frames = []

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if agent_id == "first_0":
        score_w += reward
        if reward > 0:
            hit_frames.append((i, reward))
    if term or trunc or i > 10000:
        break
    
    ram = extraer_ram_segura(env)
    my_x, my_y = int(ram[32]), int(ram[34])
    opp_x, opp_y = int(ram[33]), int(ram[35])
    dist_x = abs(opp_x - my_x)
    dist_y = abs(opp_y - my_y)
    
    if dist_x < min_dist:
        min_dist = dist_x
    
    if agent_id == "first_0":
        # Both approach, then PUNCH when opponent is moving toward us AND dist drops
        opp_approaching = (prev_opp_x > my_x and opp_x < prev_opp_x) or (prev_opp_x < my_x and opp_x > prev_opp_x)
        
        if dist_x <= 16 and dist_y <= 4:
            # Close enough! FIRE!
            action = 1  # FIRE
        elif dist_y >= 3:
            if opp_y > my_y:
                action = 8  # DOWNRIGHT
            else:
                action = 6  # UPRIGHT
        elif opp_x > my_x:
            action = 3  # RIGHT
        else:
            action = 4  # LEFT
        
        prev_opp_x = opp_x
        prev_opp_y = opp_y
    else:
        # Opponent walks STRAIGHT toward us at full speed
        if my_y > opp_y + 2:
            action = 8  # DOWNRIGHT (move toward us in Y and X)
        elif my_y < opp_y - 2:
            action = 6  # UPRIGHT
        elif my_x < opp_x:
            action = 4  # LEFT (move toward us in X)
        else:
            action = 3  # RIGHT
    
    env.step(action)

print(f"Final score: W={score_w:.0f}, min_dist={min_dist}")
print(f"Hit frames (white scored): {hit_frames}")

# Check final RAM
env2 = boxing_v2.env(obs_type="rgb_image")
env2.reset()
# Try a DIFFERENT approach: approach, then STOP and let opponent walk into our fist
score_w2 = 0
min_dist2 = 999
prev_opp_x2 = 109
prev_opp_y2 = 87

for i, agent_id in enumerate(env2.agent_iter()):
    obs, reward, term, trunc, info = env2.last()
    if agent_id == "first_0":
        score_w2 += reward
    if term or trunc or i > 10000:
        break
    
    ram = extraer_ram_segura(env2)
    my_x, my_y = int(ram[32]), int(ram[34])
    opp_x, opp_y = int(ram[33]), int(ram[35])
    dist_x = abs(opp_x - my_x)
    dist_y = abs(opp_y - my_y)
    
    if dist_x < min_dist2:
        min_dist2 = dist_x
    
    if agent_id == "first_0":
        # Approach then STOP and spam FIRE
        if dist_y > 3:
            if opp_y > my_y:
                action = 8  # DOWNRIGHT
            else:
                action = 6  # UPRIGHT
        elif dist_x > 14:
            if opp_x > my_x:
                action = 3  # RIGHT
            else:
                action = 4  # LEFT
        else:
            # We're close - alternate between FIRE and NOOP
            action = 1  # FIRE only
    else:
        # Opponent: approach and also PUNCH
        if dist_y > 3:
            if my_y > opp_y:
                action = 8  # DOWNRIGHT
            else:
                action = 6  # UPRIGHT
        elif abs(my_x - opp_x) > 14:
            if my_x < opp_x:
                action = 4  # LEFT  
            else:
                action = 3  # RIGHT
        else:
            action = 1  # FIRE
    
    env2.step(action)

print(f"\nTest 2 (approach+spam FIRE vs opponent approach+FIRE): Score W={score_w2:.0f}, min_dist={min_dist2}")
env2.close()

# Test 3: Check if scoring happens when we play AS second_0 (black player)
env3 = boxing_v2.env(obs_type="rgb_image")
env3.reset()
score_b3 = 0
min_dist3 = 999

for i, agent_id in enumerate(env3.agent_iter()):
    obs, reward, term, trunc, info = env3.last()
    if agent_id == "second_0":
        score_b3 += reward
    if term or trunc or i > 10000:
        break
    
    ram = extraer_ram_segura(env3)
    white_x, white_y = int(ram[32]), int(ram[34])
    black_x, black_y = int(ram[33]), int(ram[35])
    
    if agent_id == "first_0":
        # White player (opponent) walks toward black player
        dist_x = abs(white_x - black_x)
        dist_y = abs(white_y - black_y)
        if dist_y > 3:
            if black_y > white_y:
                action = 8  # DOWNRIGHT
            else:
                action = 6  # UPRIGHT
        elif dist_x > 14:
            if black_x > white_x:
                action = 3  # RIGHT
            else:
                action = 4  # LEFT
        else:
            action = 1  # FIRE
    else:
        # Black player (US) uses aggressive heuristic
        dist_x = abs(white_x - black_x)
        dist_y = abs(white_y - black_y)
        if dist_x < min_dist3:
            min_dist3 = dist_x
        
        if dist_y > 3:
            if white_y < black_y:
                action = 2  # UP (toward white)
            else:
                action = 5  # DOWN
        elif dist_x > 14:
            if white_x > black_x:
                action = 3  # RIGHT (nope, we're on the right, need left)
                action = 4  # LEFT (toward white)
            else:
                action = 3  # RIGHT (toward white)
        else:
            action = 1  # FIRE when close
    
    env3.step(action)

print(f"\nTest 3 (playing as BLACK/FIRE): Score B={score_b3:.0f}, min_dist={min_dist3}")
env3.close()
