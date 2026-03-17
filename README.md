# Stanford AA273 Winter 2026 Final Project: 
## Robust Measurement Handling of Neural-Network Based UKF for Spacecraft Pose Estimation

### Overview

This project investigates several possible methods for improving the robustness of integrating a CNN-based pose estimation pipeline with a Multiplicative Unscented Kalman Filter (MUKF).
The baseline implementation of the CNN-based MEKF is described in [1]. For this project, the baseline implementation is simplified from the original formulation by removing the adaptive process noise formulation and using simple two-body dynamics STM for the dynamics model.

### Running Code:

The primary script is the ``run_filter.py`` script in the ``ukf`` filter. However, this repository does have some external dependecies that it requires to run. Specifically, the pytorch implementation of SPNv2 was not re-implemented for this work (although I did create a new inference pipeline/interface which is available in ``spn/model.py``). So you will need to have SPNv2 installed and then would likely have to change around some of the imports in model.py. I will work on making this a cleaner interface for use by others in the Space Rendezvous Lab. 


To Do:
* Clean the interface for SPNv2 so it's easier to use by others
* Set up a specific poetry environment and generate the .toml file. 
