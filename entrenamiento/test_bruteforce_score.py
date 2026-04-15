import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura

# For each action, approach then repeatedly try that action and check scores
action_names = {
    0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
    6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT",
    10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE",
    14: "UPRIGHTFIRE", 15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE"
}

for punch_action in [1, 11, 12, 13, 14, 15, 16, 17, 10]:
    env = boxing_v2.env(obs_type="rgb_image")
    env.reset()
    score_w = 0
    score_b = 0
    approached = False
    min_dist = 999
    
    for i, agent_id in enumerate(env.agent_iter()):
        obs, reward, term, trunc, info = env.last()
        if agent_id == "first_0":
            score_w += reward
        else:
            score_b += reward
        
        if term or trunc or i > 6000:
            break
        
        ram = extraer_ram_segura(env)
        my_x, my_y = int(ram[32]), int(ram[34])
        opp_x, opp_y = int(ram[33]), int(ram[35])
        dist_x = abs(opp_x - my_x)
        dist_y = abs(opp_y - my_y)
        
        if dist_x < min_dist:
            min_dist = dist_x
        
        if agent_id == "first_0":
            # Phase 1: Approach opponent (first 500 steps)
            if i < 500:
                if dist_y > 3:
                    if opp_y > my_y: action = 8  # DOWNRIGHT
                    elif opp_y < my_y: action = 6  # UPRIGHT
                    else: action = 3  # RIGHT
                elif dist_x > 14:
                    if opp_x > my_x: action = 3  # RIGHT
                    else: action = 4  # LEFT
                else:
                    approached = True
                    action = punch_action
            else:
                # Phase 2: Alternate between aligning and punching
                if i % 2 == 0:
                    action = punch_action
                else:
                    # Stay aligned
                    if dist_y > 2:
                        if opp_y > my_y: action = 5  # DOWN
                        else: action = 2  # UP
                    elif dist_x > 18:
                        if opp_x > my_x: action = 3  # RIGHT
                        else: action = 4  # LEFT
                    else:
                        action = punch_action
        else:
            # Opponent: walk toward us NONSTOP
            if my_y > opp_y:
                action = 9  # DOWNLEFT (toward us)
            elif my_y < opp_y:
                action = 7  # UPLEFT (toward us)
            else:
                action = 4  # LEFT (toward us)
        
        env.step(action)
    
    ram = extraer_ram_segura(env)
    env.close()
    print(f"Punch action {punch_action:2d} ({action_names[punch_action]:15s}): W_score={score_w:.0f} B_score={score_b:.0f} RAM_W={int(ram[18])} RAM_B={int(ram[19])} min_dist={min_dist}")

