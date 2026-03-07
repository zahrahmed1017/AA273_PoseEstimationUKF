# Stanford AA273 Winter 2026 Final Project: 
## Robust Measurement Handling of Neural-Network Based UKF for Spacecraft Pose Estimation

### Overview

This project investigates several possible methods for improving the robustness of integrating a CNN-based pose estimation pipeline with a Multiplicative Unscented Kalman Filter (MUKF).
The baseline implementation of the CNN-based MEKF is described in [1]. For this project, the baseline implementation is modified in 2 ways. First, it is simplified by removing the adaptive process noise formulation. Second, it is extended to include orbit control such that it can be evaluated on more complex, controlled rendezvous trajectories.

Then, two modifications are explored for improving the robustness of the measurement pipeline. 
1. Covariance Guided ROI Selection
2. Variable Measurement Vector

