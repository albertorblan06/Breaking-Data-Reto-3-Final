import os
import numpy as np
from interfaz import AgenteBase


class AgenteInferencia(AgenteBase):
    """
    Hunt v7: Bait-and-punch strategy with aggressive chase.
    
    Key insights:
    1. Punches only connect when dist_x < 14 (brief collision gap)
    2. Against wall-hugging opponents, we need to bait them forward
    3. Use timed punch rhythm (every 3 frames) for maximum hit rate
    4. Move away briefly to create opponent movement, then rush back
    """
    def __init__(self):
        self.frame_count = 0
        self.prev_opp_x = -1
        self.prev_opp_y = -1
        self.bait_timer = 0
        self.bait_phase = False  # True when we're moving away to bait
        super().__init__(nombre_equipo="Aquatic_Agents")

    def configurar(self):
        print(f"[{self.nombre_equipo}] Hunt v7 - bait and punch!")

    def predict(self, estado):
        ram = estado["ram"].copy()
        soy_blanco = estado["soy_blanco"]
        self.frame_count += 1

        # Extract positions with mirroring
        if soy_blanco:
            my_x = int(ram[32])
            my_y = int(ram[34])
            opp_x = int(ram[33])
            opp_y = int(ram[35])
        else:
            my_x = int(ram[33])
            my_y = int(ram[35])
            opp_x = int(ram[32])
            opp_y = int(ram[34])

        dx = opp_x - my_x
        dy = opp_y - my_y
        dist_x = abs(dx)
        dist_y = abs(dy)

        # Track opponent movement
        opp_moving_toward = False
        if self.prev_opp_x >= 0:
            if dx > 0 and opp_x < self.prev_opp_x:  # Opponent moving left (toward us if we're left)
                opp_moving_toward = True
            elif dx < 0 and opp_x > self.prev_opp_x:  # Opponent moving right (toward us if we're right)
                opp_moving_toward = True
        self.prev_opp_x = opp_x
        self.prev_opp_y = opp_y

        # Punch every 3rd frame
        is_punch = (self.frame_count % 3 == 0)

        # BAIT PHASE: If opponent isn't moving toward us, move away to create bait
        # Cycle: 30 frames bait (move away) + 60 frames chase (move toward + punch)
        cycle_length = 90
        chase_start = 30
        
        if not opp_moving_toward and dist_x <= 20 and self.bait_timer < chase_start:
            self.bait_timer += 1
            # Move AWAY to create a gap and bait opponent forward
            if is_punch:
                # Punch even while backing away
                if dx > 0:
                    return 12  # LEFTFIRE (move away + punch)
                else:
                    return 11  # RIGHTFIRE (move away + punch)
            else:
                if dx > 0:
                    return 4  # LEFT (move away)
                else:
                    return 3  # RIGHT (move away)
        else:
            self.bait_timer = max(0, self.bait_timer - 1)

        # CHASE PHASE: Aggressive pursuit + timed punching
        # Priority 1: Y alignment
        if dist_y >= 5:
            if is_punch:
                if dy > 0 and dx > 0: return 16  # DOWNRIGHTFIRE
                elif dy > 0 and dx < 0: return 17  # DOWNLEFTFIRE
                elif dy > 0: return 13               # DOWNFIRE
                elif dy < 0 and dx > 0: return 14   # UPRIGHTFIRE
                elif dy < 0 and dx < 0: return 15   # UPLEFTFIRE
                else: return 10                        # UPFIRE
            else:
                if dy > 0 and dx > 0: return 8       # DOWNRIGHT
                elif dy > 0 and dx < 0: return 9     # DOWNLEFT
                elif dy > 0: return 5                   # DOWN
                elif dy < 0 and dx > 0: return 6     # UPRIGHT
                elif dy < 0 and dx < 0: return 7     # UPLEFT
                else: return 2                          # UP

        # Priority 2: Far approach (dist_x > 30) with Y aligned
        if dist_x > 30:
            if is_punch:
                if dx > 0: return 11       # RIGHTFIRE
                else: return 12             # LEFTFIRE
            else:
                if dx > 0: return 3         # RIGHT
                else: return 4               # LEFT

        # Priority 3: In punch range - timed punches with position maintenance
        if dist_y < 5 and 14 <= dist_x <= 30:
            if is_punch:
                # Directional fire toward opponent
                if dx > 0: return 11           # RIGHTFIRE
                elif dx < 0: return 12          # LEFTFIRE
                else: return 1                   # FIRE
            
            # Non-punch: maintain position
            if dist_y >= 3:
                if dy > 0: return 5           # DOWN
                else: return 2                   # UP
            
            if dist_x > 28:
                if dx > 0: return 3            # RIGHT
                else: return 4                  # LEFT
            elif dist_x < 18:
                if dx > 0: return 4            # LEFT (back off)
                else: return 3                  # RIGHT (back off)
            else:
                return 0                         # NOOP (hold position)

        # Priority 4: Too close or fallback
        if dist_x < 14:
            if dx > 0: return 4                 # LEFT (back off)
            else: return 3                       # RIGHT (back off)
        
        # Fallback: chase opponent
        if dx > 0: return 3
        elif dx < 0: return 4
        elif dy > 0: return 5
        elif dy < 0: return 2
        else: return 0
