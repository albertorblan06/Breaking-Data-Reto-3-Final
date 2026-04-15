# Discoveries & Game Mechanics

This document outlines the core discoveries about the PettingZoo Atari Boxing environment that formed the foundation of our agent's strategy.

## RAM Addresses

The environment state is exposed as 128 bytes of RAM. By reverse-engineering the ALE (Arcade Learning Environment) memory map and testing via scripts like `test_ram_coords.py` and `test_score_tracking.py`, we identified the critical addresses:

*   `ram[32]`: Player 1 X-coordinate
*   `ram[34]`: Player 1 Y-coordinate
*   `ram[33]`: Player 2 X-coordinate
*   `ram[35]`: Player 2 Y-coordinate
*   `ram[18]`: Player 1 Score
*   `ram[19]`: Player 2 Score

*Note: These addresses are absolute. If we are playing as the black boxer (`second_0`), we must internally mirror our reads (e.g., treat `ram[33]` as our X and `ram[32]` as the opponent's X).*

## Collision and Hitboxes

A critical breakthrough was understanding the game's physical constraints via `test_distances.py`:

1.  **The Collision Boundary (`dist_x < 14`)**:
    The boxers occupy physical space. They cannot get closer than approximately 14 pixels horizontally. If you press `LEFT` or `RIGHT` into the opponent, you will hit this invisible wall.
    *Crucial finding:* Punches **cannot** connect when players are stuck at `dist_x = 14`. If you just hold `FORWARD + FIRE`, you will infinitely bump into the opponent and score 0 points.

2.  **The Sweet Spot (`14 < dist_x <= 30`, `dist_y < 5`)**:
    Punches connect at surprisingly large distances. The optimal scoring range is when the horizontal distance (`dist_x`) is between 18 and 28 pixels, and the vertical distance (`dist_y`) is tightly aligned (less than 5 pixels).

## Punch Timing (The Rhythm)

Through scripts like `test_punch_timing.py` and `test_pulse_fire.py`, we discovered that continuous `FIRE` spamming (Action 1) does *not* result in continuous punches. The Atari animation locks the player for several frames after a punch is thrown.

*   **Continuous FIRE:** The game registers the button hold, but the boxer only punches occasionally, often missing the tiny window when the opponent is vulnerable.
*   **The 3-Frame Rhythm:** The optimal punch frequency is exactly once every 3 frames (`frame_count % 3 == 0`). This perfectly matches the animation recovery time, maximizing the number of active hitboxes thrown per second while allowing movement on the "off" frames.

## Movement Mechanics

*   **Directional Fires:** The action space includes combined actions (e.g., `UPRIGHTFIRE` = 14). These are incredibly useful for closing distance diagonally while maintaining offensive pressure.
*   **Static Punches:** `FIRE` (Action 1) throws a punch without moving horizontally. `RIGHTFIRE` (11) or `LEFTFIRE` (12) moves the player slightly. We found that using pure `FIRE` when already in the "sweet spot" prevents the player from accidentally shifting into the `dist_x < 14` collision boundary during a combo.
