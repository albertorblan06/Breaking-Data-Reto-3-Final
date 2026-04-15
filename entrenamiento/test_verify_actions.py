import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura

# Check the action mapping by testing each action for 20 steps and seeing position changes
action_names = {
    0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
    6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT",
    10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE",
    14: "UPRIGHTFIRE", 15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE"
}

# Also check what the actual action space looks like
env = boxing_v2.env(obs_type="rgb_image")
print(f"Action space: {env.action_space}")
print(f"Possible agents: {env.possible_agents}")

# Get the ALE minor action set
try:
    print(f"\nALE action set: {env.unwrapped.ale.getMinimalActionSet()}")
    print(f"ALE legal actions: {env.unwrapped.ale.getLegalActionSet()}")
except Exception as e:
    print(f"Could not get ALE actions: {e}")

env.reset()

# Test: run a few steps with known actions and track position/score changes
for action_id in [0, 1, 2, 3, 4, 5, 11]:
    env_test = boxing_v2.env(obs_type="rgb_image")
    env_test.reset()
    
    start_x, start_y = None, None
    end_x, end_y = None, None
    start_score_w = 0
    start_score_b = 0
    
    step_count = 0
    for i, agent_id in enumerate(env_test.agent_iter()):
        obs, reward, term, trunc, info = env_test.last()
        if term or trunc:
            break
        
        ram = extraer_ram_segura(env_test)
        
        if agent_id == "first_0":
            if start_x is None:
                start_x = int(ram[32])
                start_y = int(ram[34])
                start_score_w = int(ram[18])
            end_x = int(ram[32])
            end_y = int(ram[34])
            step_count += 1
            action = action_id
        else:
            action = 0  # NOOP
        
        if step_count > 20:
            break
        
        env_test.step(action)
    
    ram_final = extraer_ram_segura(env_test)
    dx = end_x - start_x
    dy = end_y - start_y
    print(f"Action {action_id:2d} ({action_names[action_id]:15s}): DX={dx:+3d}, DY={dy:+3d} Score W={int(ram_final[18])} B={int(ram_final[19])}")
    env_test.close()

# Now let's try to SCORE: approach + FIRE with both players
print("\n--- SCORING TEST: Both players approach + FIRE ---")
env2 = boxing_v2.env(obs_type="rgb_image")
env2.reset()
score_w = 0
score_b = 0

for i, agent_id in enumerate(env2.agent_iter()):
    obs, reward, term, trunc, info = env2.last()
    if agent_id == "first_0":
        score_w += reward
    else:
        score_b += reward
    
    if term or trunc or i > 4000:
        break
    
    ram = extraer_ram_segura(env2)
    my_x, my_y = int(ram[32]), int(ram[34])
    opp_x, opp_y = int(ram[33]), int(ram[35])
    
    if agent_id == "first_0":
        # Approach then FIRE
        if abs(opp_y - my_y) > 2:
            if opp_y > my_y: action = 8
            else: action = 6
        elif opp_x > my_x: action = 11  # RIGHTFIRE
        elif opp_x < my_x: action = 12  # LEFTFIRE
        else: action = 1  # FIRE
    else:
        # Opponent also approaches and FIREs
        if abs(my_y - opp_y) > 2:
            if my_y < opp_y: action = 9  # DOWNLEFT (approach + fire)
            elif my_y > opp_y: action = 7  # UPLEFT (approach + fire)
            else: action = 12  # LEFTFIRE
        elif my_x < opp_x: action = 12  # LEFTFIRE
        elif my_x > opp_x: action = 11  # RIGHTFIRE
        else: action = 1  # FIRE
    
    env2.step(action)

ram_final = extraer_ram_segura(env2)
print(f"Score: W={score_w} B={score_b} RAM: W={int(ram_final[18])} B={int(ram_final[19])}")
env2.close()
