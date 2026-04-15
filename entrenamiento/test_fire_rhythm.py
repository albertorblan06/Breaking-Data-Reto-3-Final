import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura

# Test different FIRE rhythms against a NOOP opponent
env = boxing_v2.env(obs_type="rgb_image")
env.reset()

for rhythm_name, rhythm_fn in [
    ("spam_FIRE", lambda f: True),
    ("every_2", lambda f: f % 2 == 0),
    ("every_3", lambda f: f % 3 == 0),
    ("every_4", lambda f: f % 4 == 0),
    ("every_6", lambda f: f % 6 == 0),
    ("every_8", lambda f: f % 8 == 0),
    ("punch_2_rest_4", lambda f: (f % 6) < 2),
    ("punch_3_rest_3", lambda f: (f % 6) < 3),
    ("punch_1_rest_2", lambda f: (f % 3) < 1),
    ("punch_1_rest_1", lambda f: (f % 2) < 1),
]:
    env = boxing_v2.env(obs_type="rgb_image")
    env.reset()
    frame = 0
    score = 0
    for agent_id in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if agent_id == "first_0":
            score += reward
        
        if term or trunc:
            break
        
        ram = extraer_ram_segura(env)
        my_x, my_y = int(ram[32]), int(ram[34])
        opp_x, opp_y = int(ram[33]), int(ram[35])
        dist_x = abs(opp_x - my_x)
        dist_y = abs(opp_y - my_y)
        
        if agent_id == "first_0":
            # Approach then use specified rhythm
            dx = opp_x - my_x
            dy = opp_y - my_y
            
            if dist_y > 3:
                if dy > 0: action = 5
                else: action = 2
            elif dist_x > 14:
                if dx > 0: action = 3
                else: action = 4
            else:
                # In range! Use the rhythm
                if rhythm_fn(frame):
                    action = 1  # FIRE
                else:
                    # Stay aligned
                    if dist_y >= 2:
                        if dy > 0: action = 5
                        elif dy < 0: action = 2
                        else: action = 0
                    else:
                        action = 0  # NOOP
            frame += 1
        else:
            action = 0  # Opponent NOOP
        
        env.step(action)
    
    env.close()
    print(f"{rhythm_name}: score = {score}")

