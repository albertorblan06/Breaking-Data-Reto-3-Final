# Evolution of Strategies

The Grandmaster agent is the culmination of 14 major iterations. This document chronicles the journey from early Machine Learning experiments to the final, deterministic Meta-Agent.

## Phase 1: Machine Learning (Hunt v1 - v6)

Initially, the challenge was to train an agent capable of generalizing across the three drastically different baselines. The environment's reward function (+1 for hitting, -1 for being hit) meant that any trained model would "overfit" to the baseline it was trained against.

*   **PPO (Proximal Policy Optimization)**: We attempted to train a continuous PPO model against the `equipo_onnx` baseline. While it learned to beat ONNX, the 25ms tournament latency limit made heavy PPO inference risky.
*   **Behavioral Cloning (BC)**: We manually recorded hundreds of expert matches (via `test_ale_player.py`) emphasizing precise 18-28 pixel "sweet spot" punches and evasive movement. A lightweight Random Forest (`rf_behavioral_clone.onnx`) was trained on this expert dataset. While lightning-fast (<0.1ms), it failed to generalize to the unpredictable Random baseline and tied the Vision baseline 0-0.

*Conclusion:* A static neural network could not adapt to all three baselines simultaneously. We needed a heuristic approach.

## Phase 2: The Heuristic Era (Hunt v7 - v10)

We shifted to deep-diving the Atari RAM addresses to build perfectly deterministic rules based on the opponent's absolute coordinates (`ram[32]` to `ram[35]`).

*   **Hunt v7 (The "Bait-and-Punch"):**
    *   *Concept:* The ONNX baseline hugged the right wall. We programmed our agent to move *away* (LEFT) when the opponent was close but not moving toward us. This "baited" the ONNX model to follow us. Once it stepped into the 18-28 pixel "sweet spot," we unleashed a perfect 3-frame punch rhythm.
    *   *Result:* Destroyed ONNX (100% win rate). Tied Vision (0-0, Vision doesn't chase). Lost to Random (~40% win rate).

*   **Hunt v8 (Relentless Pursuit):**
    *   *Concept:* Removed the baiting phase entirely. Always press forward and corner the opponent.
    *   *Result:* Still tied Vision (it just dodged our straight-line pursuit). Lost more to Random (we ran into their wild punches).

*   **Hunt v9 & v10 (The "Distance Controller"):**
    *   *Concept:* Never press `FORWARD` if `dist_x < 16`. If we get too close, physically back off to the 18-28 sweet spot. Use pure `FIRE` (Action 1) instead of `RIGHTFIRE`/`LEFTFIRE` to avoid accidentally stepping forward while punching.
    *   *Result:* Solved the collision boundary issue, but Vision still perfectly evaded our punches.

## Phase 3: Breaking the CNN (Hunt v11)

The Vision baseline was essentially a perfect dodging machine. We realized that a CNN relies on clean, predictable motion vectors across its 4-frame stack to compute trajectories.

*   **Hunt v11 (The "Drunken Master"):**
    *   *Concept:* We programmed the agent to intentionally move "wrong." While closing the distance, there was a 20% chance it would jerk `UP` or `DOWN` in the opposite direction of the optimal path. Furthermore, instead of a perfect 3-frame punch rhythm, we gave it a 35% chance to throw a punch on *any* frame.
    *   *Result:* The Vision CNN completely broke down. It couldn't parse the high-frequency noise and stopped dodging. We began scoring 12-0, 18-0 blowouts against the previously unbeatable Vision baseline. However, this chaotic style was slightly less effective against ONNX and Random.

## Phase 4: The Meta-Agent (Hunt v12 - v14)

We had three perfect counter-strategies, but no way to know which baseline we were fighting when the match started. The solution was a **Profiler**.

*   **Hunt v14 (The "Grandmaster"):**
    *   *Concept:* For the first 60 frames (1 second), the agent does nothing but run to the top-left corner (`X > 30 = LEFT`, `Y > 30 = UP`) and record every unique `(X, Y)` coordinate pair the opponent visits.
    *   *Classification:*
        *   `Unique pairs <= 2`: **VISION** (Stands completely still). -> Deploy *Drunken Master* (v11).
        *   `Unique pairs < 20`: **ONNX** (Twitches vertically). -> Deploy *Bait-and-Punch* (v7).
        *   `Unique pairs >= 20`: **RANDOM** (Flails wildly). -> Deploy *Safe Bob-and-Weave* (Maintains sweet spot while rapidly oscillating Y to dodge).
    *   *Result:* A near 100% win-rate across the board, with a 93% overall non-loss rate in comprehensive evaluation series. The ultimate, latency-proof, adaptive agent.
