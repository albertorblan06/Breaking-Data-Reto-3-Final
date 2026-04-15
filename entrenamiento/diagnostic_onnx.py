"""Diagnostic: Log frame-by-frame positions against ONNX to understand v7 behavior."""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inferencia"))
from pettingzoo.atari import boxing_v2
from arena import extraer_ram_segura, cargar_agente_desde_carpeta
import importlib.util
import numpy as np

ruta_ent = os.path.dirname(os.path.abspath(__file__))
ruta_inf = os.path.abspath(os.path.join(ruta_ent, "..", "inferencia"))

spec = importlib.util.spec_from_file_location("mod_ours", os.path.join(ruta_ent, "submission_rf.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
our_agent = mod.AgenteInferencia()
our_agent.ruta_modelo = ruta_ent
our_agent.configurar()

onnx_agent = cargar_agente_desde_carpeta(os.path.join(ruta_inf, "modelos", "equipo_onnx"))
onnx_agent.configurar()

env = boxing_v2.env(obs_type="rgb_image")
env.reset()

frames = []
scores = {'first_0': 0, 'second_0': 0}

for agent_id in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    scores[agent_id] += reward
    if term or trunc:
        action = None
    else:
        ram = extraer_ram_segura(env)
        estado = {"imagen": obs, "ram": ram, "soy_blanco": (agent_id == "first_0")}
        if agent_id == "first_0":
            action = our_agent.predict(estado)
            my_x = int(ram[32])
            my_y = int(ram[34])
            opp_x = int(ram[33])
            opp_y = int(ram[35])
            dx = opp_x - my_x
            dy = opp_y - my_y
            dist_x = abs(dx)
            dist_y = abs(dy)
            frames.append({
                'frame': our_agent.frame_count,
                'my_x': my_x, 'my_y': my_y,
                'opp_x': opp_x, 'opp_y': opp_y,
                'dx': dx, 'dy': dy,
                'dist_x': dist_x, 'dist_y': dist_y,
                'action': action,
                'in_sweet_spot': (18 <= dist_x <= 28 and dist_y < 5),
            })
        else:
            action = onnx_agent.predict(estado)
    env.step(action)
env.close()

print(f"Final score: Us={scores['first_0']:.0f} ONNX={scores['second_0']:.0f}")
print(f"Total frames: {len(frames)}")

# Score change tracking
score_events = [i for i in range(1, len(frames)) if True]  # just print stats

# Position stats
dist_x_vals = [f['dist_x'] for f in frames]
dist_y_vals = [f['dist_y'] for f in frames]
print(f"dist_x: mean={np.mean(dist_x_vals):.1f} median={np.median(dist_y_vals):.1f} min={min(dist_x_vals)} max={max(dist_x_vals)}")
print(f"dist_y: mean={np.mean(dist_y_vals):.1f} median={np.median(dist_y_vals):.1f} min={min(dist_y_vals)} max={max(dist_y_vals)}")

# X distance distribution
dist_buckets = {'<14': 0, '14-17': 0, '18-28': 0, '29-35': 0, '>35': 0}
for f in frames:
    dx = f['dist_x']
    if dx < 14: dist_buckets['<14'] += 1
    elif dx < 18: dist_buckets['14-17'] += 1
    elif dx <= 28: dist_buckets['18-28'] += 1
    elif dx <= 35: dist_buckets['29-35'] += 1
    else: dist_buckets['>35'] += 1
print(f"\nX distance distribution:")
for bucket, count in dist_buckets.items():
    print(f"  {bucket}: {count} frames ({100*count/max(len(frames),1):.1f}%)")

sweet_spot_frames = sum(1 for f in frames if f['in_sweet_spot'])
print(f"\nFrames in sweet spot (dx 18-28, dy<5): {sweet_spot_frames} ({100*sweet_spot_frames/max(len(frames),1):.1f}%)")

# Action distribution
action_names = {0:'NOOP', 1:'FIRE', 2:'UP', 3:'RIGHT', 4:'LEFT', 5:'DOWN',
                6:'UPRIGHT', 7:'UPLEFT', 8:'DOWNRIGHT', 9:'DOWNLEFT',
                10:'UPFIRE', 11:'RIGHTFIRE', 12:'LEFTFIRE', 13:'DOWNFIRE',
                14:'UPRIGHTFIRE', 15:'UPLEFTFIRE', 16:'DOWNRIGHTFIRE', 17:'DOWNLEFTFIRE'}
action_counts = {}
for f in frames:
    name = action_names.get(f['action'], str(f['action']))
    action_counts[name] = action_counts.get(name, 0) + 1
print(f"\nAction distribution:")
for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
    print(f"  {name}: {count} ({100*count/len(frames):.1f}%)")

# Print last 30 sweet-spot frames
sweet_frames = [f for f in frames if f['in_sweet_spot']]
print(f"\nLast 30 sweet-spot frames:")
for f in sweet_frames[-30:]:
    print(f"  f={f['frame']:4d} dx={f['dx']:3d} dy={f['dy']:3d} act={f['action']:2d}")
