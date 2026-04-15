"""Diagnostic: Log frame-by-frame positions against Vision to understand why we can't score."""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import importlib.util
import numpy as np

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

# Load our agent
spec = importlib.util.spec_from_file_location("mod_ours", os.path.join(ruta_ent, "submission_rf.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
our_agent = mod.AgenteInferencia()
our_agent.ruta_modelo = ruta_ent
our_agent.configurar()

# Load vision agent
vision_agent = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_vision"))
vision_agent.configurar()

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

# Track data
frames = []
scores = {'first_0': 0, 'second_0': 0}
prev_scores = {'first_0': 0, 'second_0': 0}

for agent_id in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    scores[agent_id] += reward
    
    if term or trunc:
        action = None
    else:
        ram = extraer_ram_segura(env)
        estado = {"imagen": obs, "ram": ram, "soy_blanco": (agent_id == "first_0")}
        
        if agent_id == "first_0":
            # Our agent
            action = our_agent.predict(estado)
            
            # Log data
            my_x = int(ram[32])
            my_y = int(ram[34])
            opp_x = int(ram[33])
            opp_y = int(ram[35])
            dx = opp_x - my_x
            dy = opp_y - my_y
            dist_x = abs(dx)
            dist_y = abs(dy)
            dist = (dist_x**2 + dist_y**2)**0.5
            
            # Check for score event
            score_event = ""
            cur_our = scores['first_0']
            cur_opp = scores['second_0']
            if cur_our > prev_scores['first_0']:
                score_event = "WE SCORED!"
            if cur_opp > prev_scores['second_0']:
                score_event = "OPP SCORED!"
            prev_scores['first_0'] = cur_our
            prev_scores['second_0'] = cur_opp
            
            frame_data = {
                'frame': our_agent.frame_count,
                'my_x': my_x, 'my_y': my_y,
                'opp_x': opp_x, 'opp_y': opp_y,
                'dx': dx, 'dy': dy,
                'dist_x': dist_x, 'dist_y': dist_y,
                'action': action,
                'score_event': score_event,
                'in_punch_range': (14 <= dist_x <= 30 and dist_y < 15),
                'in_sweet_spot': (14 <= dist_x <= 30 and dist_y < 5),
            }
            frames.append(frame_data)
        else:
            action = vision_agent.predict(estado)
    
    env.step(action)

env.close()

# Analyze
print(f"Final score: Us={scores['first_0']:.0f} Vision={scores['second_0']:.0f}")
print(f"Total frames: {len(frames)}")

# Time in punch range
punch_range_frames = sum(1 for f in frames if f['in_punch_range'])
sweet_spot_frames = sum(1 for f in frames if f['in_sweet_spot'])
print(f"Frames in punch range (dx 14-30, dy<15): {punch_range_frames} ({100*punch_range_frames/max(len(frames),1):.1f}%)")
print(f"Frames in sweet spot (dx 14-30, dy<5): {sweet_spot_frames} ({100*sweet_spot_frames/max(len(frames),1):.1f}%)")

# Score events
score_events = [f for f in frames if f['score_event']]
print(f"\nScore events: {len(score_events)}")
for f in score_events:
    print(f"  Frame {f['frame']}: {f['score_event']} dx={f['dx']} dy={f['dy']} dist_x={f['dist_x']} dist_y={f['dist_y']}")

# Distance distribution
dist_buckets = {'<14': 0, '14-20': 0, '20-30': 0, '30-50': 0, '>50': 0}
for f in frames:
    dx = f['dist_x']
    if dx < 14: dist_buckets['<14'] += 1
    elif dx < 20: dist_buckets['14-20'] += 1
    elif dx < 30: dist_buckets['20-30'] += 1
    elif dx < 50: dist_buckets['30-50'] += 1
    else: dist_buckets['>50'] += 1

print(f"\nX distance distribution:")
for bucket, count in dist_buckets.items():
    print(f"  {bucket}: {count} frames ({100*count/max(len(frames),1):.1f}%)")

# Y distance when X is in range
y_in_range = [f['dist_y'] for f in frames if 14 <= f['dist_x'] <= 30]
if y_in_range:
    print(f"\nY distance when X in range (14-30):")
    print(f"  mean={np.mean(y_in_range):.1f}, median={np.median(y_in_range):.1f}, max={max(y_in_range)}")
    y_buckets = {'0-3': 0, '3-5': 0, '5-10': 0, '10-20': 0, '>20': 0}
    for y in y_in_range:
        if y < 3: y_buckets['0-3'] += 1
        elif y < 5: y_buckets['3-5'] += 1
        elif y < 10: y_buckets['5-10'] += 1
        elif y < 20: y_buckets['10-20'] += 1
        else: y_buckets['>20'] += 1
    for bucket, count in y_buckets.items():
        print(f"  dy {bucket}: {count} ({100*count/len(y_in_range):.1f}%)")

# Action distribution in punch range
punch_range_actions = [f['action'] for f in frames if f['in_punch_range']]
if punch_range_actions:
    action_names = {0:'NOOP', 1:'FIRE', 2:'UP', 3:'RIGHT', 4:'LEFT', 5:'DOWN',
                    6:'UPRIGHT', 7:'UPLEFT', 8:'DOWNRIGHT', 9:'DOWNLEFT',
                    10:'UPFIRE', 11:'RIGHTFIRE', 12:'LEFTFIRE', 13:'DOWNFIRE',
                    14:'UPRIGHTFIRE', 15:'UPLEFTFIRE', 16:'DOWNRIGHTFIRE', 17:'DOWNLEFTFIRE'}
    action_counts = {}
    for a in punch_range_actions:
        name = action_names.get(a, str(a))
        action_counts[name] = action_counts.get(name, 0) + 1
    print(f"\nActions in punch range:")
    for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} ({100*count/len(punch_range_actions):.1f}%)")

# Print frame-by-frame for last 50 frames in sweet spot
print(f"\nLast 30 sweet-spot frames:")
sweet_frames = [f for f in frames if f['in_sweet_spot']]
for f in sweet_frames[-30:]:
    print(f"  f={f['frame']:4d} dx={f['dx']:3d} dy={f['dy']:3d} act={f['action']:2d} {f['score_event']}")
