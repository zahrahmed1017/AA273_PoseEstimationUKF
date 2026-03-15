# Stanford AA273 Winter 2026 Final Project: 
## Robust Measurement Handling of Neural-Network Based UKF for Spacecraft Pose Estimation

### Overview

This project investigates several possible methods for improving the robustness of integrating a CNN-based pose estimation pipeline with a Multiplicative Unscented Kalman Filter (MUKF).
The baseline implementation of the CNN-based MEKF is described in [1]. For this project, the baseline implementation is simplified from the original formulation by removing the adaptive process noise formulation and using simple two-body dynamics STM for the dynamics model.

Then, two modifications are explored for improving the robustness of the measurement pipeline. 
1. Covariance Guided ROI Selection
2. Variable Measurement Vector

