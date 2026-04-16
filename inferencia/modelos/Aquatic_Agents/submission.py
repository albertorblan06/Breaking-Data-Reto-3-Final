import os
import numpy as np
from interfaz import AgenteBase


class AgenteInferencia(AgenteBase):
    """
    Hunt v14: The Grandmaster (Adaptive Meta-Agent)

    Identifies the opponent in the first 60 frames and deploys the perfect counter:
    - Vision (CNN): Evades perfectly -> Drunken Master (chaotic jittering)
    - ONNX: Chases or hugs right wall -> Bait-and-Punch (creates gap, timed punches)
    - Random: Flails randomly -> Safe Bob-and-Weave (stays in sweet spot, dodges vertically)
    """

    def __init__(self):
        self.frame_count = 0
        self.opp_positions = []
        self.opponent_type = "UNKNOWN"
        # State for Bait-and-Punch
        self.v7_prev_opp_x = -1
        self.v7_prev_opp_y = -1
        self.v7_bait_timer = 0
        super().__init__(nombre_equipo="Aquatic_Agents")

    def configurar(self):
        try:
            import onnxruntime as ort

            ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo.onnx")
            if os.path.exists(ruta_modelo):
                self.session = ort.InferenceSession(ruta_modelo)
        except Exception:
            pass

    def predict(self, estado):
        ram = estado["ram"].copy()
        soy_blanco = estado["soy_blanco"]
        self.frame_count += 1

        # Extract positions
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

        # ============================================================================
        # PHASE 1: PROFILER (First 60 frames)
        # ============================================================================
        if self.frame_count <= 60:
            self.opp_positions.append((opp_x, opp_y))
            # Run to corner to observe opponent neutrally without engaging
            if my_x > 30:
                return 4  # LEFT
            elif my_y > 30:
                return 2  # UP
            else:
                return 0  # NOOP

        # ============================================================================
        # PHASE 2: CLASSIFICATION (Frame 61)
        # ============================================================================
        if self.frame_count == 61:
            unique_pos = len(set(self.opp_positions))
            if unique_pos <= 2:
                self.opponent_type = "VISION"
            elif unique_pos < 20:
                self.opponent_type = "ONNX"
            else:
                self.opponent_type = "RANDOM"
            print(
                f"[{self.nombre_equipo}] Opponent identified as: {self.opponent_type}"
            )

        # ============================================================================
        # PHASE 3: EXECUTION
        # ============================================================================
        is_punch = self.frame_count % 3 == 0

        # ----------------------------------------------------------------------------
        # DRUNKEN MASTER (vs Vision CNN)
        # ----------------------------------------------------------------------------
        if self.opponent_type == "VISION":
            if dist_x > 30:
                if is_punch:
                    if dy > 3 and dx > 0:
                        return 16
                    elif dy > 3 and dx < 0:
                        return 17
                    elif dy < -3 and dx > 0:
                        return 14
                    elif dy < -3 and dx < 0:
                        return 15
                    elif dx > 0:
                        return 11
                    else:
                        return 12
                else:
                    if dy > 3 and dx > 0:
                        return 8
                    elif dy > 3 and dx < 0:
                        return 9
                    elif dy < -3 and dx > 0:
                        return 6
                    elif dy < -3 and dx < 0:
                        return 7
                    elif dx > 0:
                        return 3
                    else:
                        return 4
            if dist_x < 16:
                if dx > 0:
                    return 4
                else:
                    return 3

            chaotic_punch = np.random.random() < 0.35
            jitter = np.random.random() < 0.20

            if chaotic_punch:
                if dist_y >= 5:
                    if dy > 0:
                        return 10 if jitter else (16 if dx > 0 else 17)
                    else:
                        return 13 if jitter else (14 if dx > 0 else 15)
                else:
                    rand_val = np.random.random()
                    if rand_val < 0.4:
                        return 1
                    elif rand_val < 0.7:
                        return 11 if dx > 0 else 12
                    else:
                        return 10 if np.random.random() < 0.5 else 13
            else:
                if dist_y >= 5:
                    if dy > 0:
                        return 2 if jitter else (8 if dx > 0 else 9)
                    else:
                        return 5 if jitter else (6 if dx > 0 else 7)
                else:
                    rand_val = np.random.random()
                    if rand_val < 0.3:
                        return 2
                    elif rand_val < 0.6:
                        return 5
                    elif rand_val < 0.8:
                        return 4 if dx > 0 else 3
                    else:
                        return 3 if dx > 0 else 4

        # ----------------------------------------------------------------------------
        # BAIT AND PUNCH (vs ONNX)
        # ----------------------------------------------------------------------------
        elif self.opponent_type == "ONNX":
            opp_moving_toward = False
            if self.v7_prev_opp_x >= 0:
                if dx > 0 and opp_x < self.v7_prev_opp_x:
                    opp_moving_toward = True
                elif dx < 0 and opp_x > self.v7_prev_opp_x:
                    opp_moving_toward = True
            self.v7_prev_opp_x = opp_x
            self.v7_prev_opp_y = opp_y

            cycle_length = 90
            chase_start = 30

            if (
                not opp_moving_toward
                and dist_x <= 20
                and self.v7_bait_timer < chase_start
            ):
                self.v7_bait_timer += 1
                if is_punch:
                    return 12 if dx > 0 else 11
                else:
                    return 4 if dx > 0 else 3
            else:
                self.v7_bait_timer = max(0, self.v7_bait_timer - 1)

            if dist_y >= 5:
                if is_punch:
                    if dy > 0 and dx > 0:
                        return 16
                    elif dy > 0 and dx < 0:
                        return 17
                    elif dy > 0:
                        return 13
                    elif dy < 0 and dx > 0:
                        return 14
                    elif dy < 0 and dx < 0:
                        return 15
                    else:
                        return 10
                else:
                    if dy > 0 and dx > 0:
                        return 8
                    elif dy > 0 and dx < 0:
                        return 9
                    elif dy > 0:
                        return 5
                    elif dy < 0 and dx > 0:
                        return 6
                    elif dy < 0 and dx < 0:
                        return 7
                    else:
                        return 2

            if dist_x > 30:
                if is_punch:
                    return 11 if dx > 0 else 12
                else:
                    return 3 if dx > 0 else 4

            if dist_y < 5 and 14 <= dist_x <= 30:
                if is_punch:
                    if dx > 0:
                        return 11
                    elif dx < 0:
                        return 12
                    else:
                        return 1
                if dist_y >= 3:
                    return 5 if dy > 0 else 2
                if dist_x > 28:
                    return 3 if dx > 0 else 4
                elif dist_x < 18:
                    return 4 if dx > 0 else 3
                else:
                    return 0

            if dist_x < 14:
                return 4 if dx > 0 else 3

            if dx > 0:
                return 3
            elif dx < 0:
                return 4
            elif dy > 0:
                return 5
            elif dy < 0:
                return 2
            else:
                return 0

        # ----------------------------------------------------------------------------
        # SAFE BOB-AND-WEAVE (vs Random)
        # ----------------------------------------------------------------------------
        else:
            # Random throws wild punches. We stay precisely at dx=22-26 where we can hit them
            # but we constantly move Y to dodge their straight punches.
            if dist_x > 26:
                if dy > 3:
                    return 16 if is_punch and dx > 0 else (8 if dx > 0 else 9)
                elif dy < -3:
                    return 14 if is_punch and dx > 0 else (6 if dx > 0 else 7)
                else:
                    return 11 if is_punch and dx > 0 else (3 if dx > 0 else 4)
            elif dist_x < 18:
                return 4 if dx > 0 else 3  # Back off

            # In sweet spot: align Y precisely, punch, and immediately move away
            if dist_y >= 5:
                # Align Y
                if dy > 0:
                    return 16 if is_punch and dx > 0 else (8 if dx > 0 else 9)
                else:
                    return 14 if is_punch and dx > 0 else (6 if dx > 0 else 7)
            else:
                # Aligned!
                if is_punch:
                    return 1  # FIRE directly
                else:
                    # Bob and weave: oscillate Y to avoid getting hit
                    return 2 if (self.frame_count % 8 < 4) else 5  # UP then DOWN
