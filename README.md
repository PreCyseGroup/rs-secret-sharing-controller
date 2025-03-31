# Robot Trajectory Tracking with Secret Sharing

## Overview
This project implements a robot trajectory tracking system using two different techniques: Reed-Solomon (RS) coding and Shamir's Secret Sharing. The system simulates a robot tracking a reference trajectory and where the controller is executed on a set of N clouds. The proposed schemes try to handle missing and incorrect shares in control inputs computed by the clouds.

## Features
- **Custom Quantization**: Implements a quantization method to handle floating-point values.
- **Error Correction**: Utilizes Reed-Solomon coding for robust control input reconstruction.
- **Simulation**: Simulates the robot's trajectory and computes position errors over time.
- **Visualization**: Generates plots for trajectory comparison, tracking errors, and share status.

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- RSSecretSharing Class (Developed by using https://github.com/mortendahl/privateml.)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```bash
   pip install numpy matplotlib
   ```

3. Ensure the RSSecretSharing library is available in your Python environment.

## Usage
1. Prepare the reference trajectory files (`xr.txt`, `yr.txt`, `thetar.txt`) in the `./Reference Trajectories/` directory.
2. Run the simulation:
   ```bash
   python robot-code.py
   ```

3. The simulation will generate plots showing:
   - Robot trajectory vs. reference trajectory
   - Position errors over time
   - Share status table indicating errors and NaN values

## Output
The following files will be generated in the working directory:
- `trajectory_tracking_comparison.eps`: A plot comparing the robot's trajectory with the reference trajectory.
- `tracking_errors.eps`: A plot showing the position errors over time.
- `share_status_table.eps`: A table visualizing the status of shares during the simulation.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
