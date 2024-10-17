# Deep Ensemble for Signal Separation in Cherenkov Telescope Observations

This repository contains the Python code for a deep ensemble method developed for signal-background separation in Cherenkov telescope observations. The method leverages multiple deep learning models to improve robustness and accuracy in separating astrophysical signals from background noise, a critical task in high-energy astrophysics.

The codebase includes:

- The deep ensemble architecture (NN_archis.py).
- The training/evaluation pipeline (sgn_sep_NN.py)
- Visualization/evaluation tools for analyzing model performance (qltests.py).
- Mock observations, including a simple toy case and a [simulated dark matter annihilation signal by A. Montanari](https://theses.hal.science/tel-04091290) and (preprocessed) observational data from the [HESS public release](https://www.mpi-hd.mpg.de/HESS/pages/dl3-dr1/) for testing (in obs_data)

Runs are organized iteratively in a for loop for ease of use on PC; they can and should be run in parallel to speed up training when using servers.

For a detailed explanation of the methodology and results, please refer to the associated [research article](https://arxiv.org/abs/2407.01329).

Feel free to reach out at marion.ullmo@gmail.com if you have any questions or suggestions!
