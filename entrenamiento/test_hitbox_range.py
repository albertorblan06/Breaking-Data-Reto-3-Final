import sys
sys.path.append("../inferencia")
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura
import time

# Test: walk directly into the opponent and keep pressing FIRE (action 1)
# to find the exact hitbox range where we score points
env = boxing_v2.env(obs_type="rgb_image")
env.reset()

score_us = 0
prev_score = 0
approach_phase = True
punch_count = 0
best_dist = 999

for i, agent_id in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if agent_id == "first_0":
        score_us += reward
    
    if term or trunc or i > 6000:
        break
    
    ram = extraer_ram_segura(env)
    my_x = int(ram[32])
    my_y = int(ram[34])
    opp_x = int(ram[33])
    opp_y = int(ram[35])
    dist_x = abs(opp_x - my_x)
    dist_y = abs(opp_y - my_y)
    
    if agent_id == "first_0":
        # First approach the opponent, then alternate PUNCH and move toward
        if approach_phase:
            if dist_y > 2:
                # Close Y distance
                if opp_y > my_y:
                    action = 5  # DOWN
                else:
                    action = 2  # UP
            elif dist_x > 10:
                # Close X distance
                if opp_x > my_x:
                    action = 3  # RIGHT
                else:
                    action = 4  # LEFT
            else:
                approach_phase = False
                action = 1  # FIRE
        else:
            # Alternate: every 3rd frame punch, rest move toward
            if punch_count % 3 == 0:
                action = 1  # FIRE
            else:
                # Move toward opponent
                if opp_x > my_x + 1:
                    action = 3  # RIGHT
                elif opp_x < my_x - 1:
                    action = 4  # LEFT
                elif opp_y > my_y + 1:
                    action = 5  # DOWN
                elif opp_y < my_y - 1:
                    action = 2  # UP
                else:
                    action = 1  # FIRE - we're on top
            punch_count += 1
        
        if dist_x < best_dist:
            best_dist = dist_x
        
        # Check if we scored
        if reward > 0:
            print(f"  HIT! Frame {i}: dist_x={dist_x}, dist_y={dist_y}, my=({my_x},{my_y}), opp=({opp_x},{opp_y}), reward={reward}")
    else:
        # Opponent does NOOP
        action = 0
    
    env.step(action)

print(f"\nFinal score: {score_us}, Closest dist_x: {best_dist}")
env.close()
