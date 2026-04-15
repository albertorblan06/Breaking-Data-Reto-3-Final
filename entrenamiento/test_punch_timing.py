import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura

# Test: Approach, then press FIRE on EVERY possible frame and check reward
env = boxing_v2.env(obs_type="rgb_image")
env.reset()

prev_score_w = 0
prev_score_b = 0
punch_frame = 0
approaching = True
last_fire_frame = -100

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if term or trunc or i > 12000:
        break
    
    ram = extraer_ram_segura(env)
    my_x, my_y = int(ram[32]), int(ram[34])
    opp_x, opp_y = int(ram[33]), int(ram[35])
    dist_x = abs(opp_x - my_x)
    dist_y = abs(opp_y - my_y)
    score_w = int(ram[18])
    score_b = int(ram[19])
    
    if agent_id == "first_0":
        # Approach, then try FIRE once every N frames
        if approaching:
            if dist_y > 3:
                if opp_y > my_y: action = 8
                else: action = 6
            elif dist_x > 14:
                if opp_x > my_x: action = 3
                else: action = 4
            else:
                approaching = False
                action = 1  # FIRE first time
                last_fire_frame = i
        else:
            # Press FIRE exactly once every 4 frames
            if (i - last_fire_frame) >= 4:
                action = 1  # FIRE
                last_fire_frame = i
            else:
                # Don't move, just wait
                action = 0  # NOOP
    else:
        # Opponent: walk toward us
        if dist_y > 3:
            if my_y > opp_y: action = 8
            else: action = 6
        elif my_x > opp_x:
            action = 4  # LEFT (toward us)
        else:
            action = 3  # RIGHT
    
    # Check for score changes
    if score_w != prev_score_w or score_b != prev_score_b:
        print(f"  FRAME {i}: SCORE CHANGE! W={score_w} B={score_b} (prev W={prev_score_w} B={prev_score_b}) dist=({dist_x},{dist_y}) my=({my_x},{my_y}) opp=({opp_x},{opp_y}) action={'FIRE' if action == 1 else 'other'}")
        prev_score_w = score_w
        prev_score_b = score_b
    
    env.step(action)

# Now test with opponent doing random actions (more realistic)
env2 = boxing_v2.env(obs_type="rgb_image")
env2.reset()
import numpy as np
prev_score_w = 0
prev_score_b = 0
approaching2 = True
last_fire2 = -100

for i, agent_id in enumerate(env2.agent_iter()):
    obs, reward, term, trunc, info = env2.last()
    if term or trunc or i > 12000:
        break
    
    ram = extraer_ram_segura(env2)
    my_x, my_y = int(ram[32]), int(ram[34])
    opp_x, opp_y = int(ram[33]), int(ram[35])
    dist_x = abs(opp_x - my_x)
    dist_y = abs(opp_y - my_y)
    score_w = int(ram[18])
    score_b = int(ram[19])
    
    if agent_id == "first_0":
        if approaching2:
            if dist_y > 3:
                if opp_y > my_y: action = 8
                else: action = 6
            elif dist_x > 14:
                if opp_x > my_x: action = 3
                else: action = 4
            else:
                approaching2 = False
                action = 1
                last_fire2 = i
        else:
            # FIRE every 4 frames
            if (i - last_fire2) >= 4:
                action = 1
                last_fire2 = i
            else:
                action = 0
    else:
        action = np.random.randint(0, 18)
    
    if score_w != prev_score_w or score_b != prev_score_b:
        print(f"  RANDOM OPP FRAME {i}: SCORE CHANGE! W={score_w} B={score_b} (prev W={prev_score_w} B={prev_score_b}) dist=({dist_x},{dist_y})")
        prev_score_w = score_w
        prev_score_b = score_b
    
    env2.step(action)

print(f"\nTest 1 (approach + opponent): Final RAM W={score_w} B={score_b}")
ram2 = extraer_ram_segura(env2)
print(f"Test 2 (random opponent): Final RAM W={int(ram2[18])} B={int(ram2[19])}")
env.close()
env2.close()
