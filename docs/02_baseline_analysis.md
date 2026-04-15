# Baseline Analysis

The PettingZoo Atari Boxing environment was populated with three unique baselines. Defeating all three required deep profiling of their behaviors.

## 1. The Random Baseline (`equipo_random`)
*   **Behavior:** A uniform random agent that selects actions 0-17 with equal probability.
*   **Threat Profile:** Flails wildly around the screen.
*   **Vulnerability:** It is completely unpredictable but generally moves randomly toward the center of the ring or bounces randomly off walls.
*   **Why It's Dangerous:** Because it randomly throws all types of directional punches, it *will* occasionally land lucky hits against deterministic strategies that simply run straight at it or attempt to perfectly camp the center of the ring.

## 2. The ONNX Baseline (`equipo_onnx`)
*   **Architecture:** A Multi-Layer Perceptron (MLP) trained on the raw `128` byte RAM vector. It takes a `float32` input of shape `(1, 128)` and returns an `int64` action index.
*   **Behavior:** It is highly deterministic. It strongly prefers to sit on the far right side of the screen (around `X=109`) and twitches vertically to match the opponent's Y-coordinate.
*   **Threat Profile:** Wall-camping. If you approach it head-on, it will attempt to punch as soon as you enter range.
*   **Vulnerability:** If you move *away* from it (baiting), it cannot handle the distance and will mindlessly walk forward to close the gap. When it does, its guard is completely open to a perfectly timed 3-frame punch rhythm.

## 3. The Vision Baseline (`equipo_vision`)
*   **Architecture:** A Convolutional Neural Network (CNN) that uses 84x84 grayscale "frame stacking" (memory of the last 4 frames) to infer movement velocity.
*   **Behavior:** The most complex and frustrating baseline. It plays an extreme evasive and defensive strategy. It will literally stand perfectly still in its corner if it is not approached.
*   **Threat Profile:** The "0-0 Tie Machine." Our early heuristics simply could not score on it. If you approach it in a straight line or diagonally, its CNN perfectly reads your velocity vector and steps out of the way on the exact frame your punch connects.
*   **Vulnerability:** Because it is a CNN trained on specific trajectory patterns, it cannot handle high-frequency "noise." If you approach it while rapidly jittering (randomly oscillating `UP` and `DOWN` every few frames) and throwing punches off-rhythm (e.g., 35% chance to punch on any frame instead of a perfect 3-frame loop), the CNN's predictive capability breaks down, and it fails to dodge. We called this the "Drunken Master" exploit.
