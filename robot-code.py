import numpy as np
import matplotlib.pyplot as plt
from RSSecretSharing import RSSecretSharing
import math
# Global Quantization Parameters
QU_BASE = 2
QU_GAMMA = 4
QU_DELTA = 5
#Global Parameters / Sharing 
PRIME = 2053
NUM_SHARES = 5
T_POLY_DEGREE = 1 
MAX_MISSING = 1
MAX_MANIPULATED = 1
SEED = 3
rscode = RSSecretSharing(PRIME, 1, NUM_SHARES, T_POLY_DEGREE, MAX_MISSING, MAX_MANIPULATED, SEED)

def custom_quantize(x):
    if x < -QU_BASE**QU_GAMMA or x > QU_BASE**QU_GAMMA - QU_BASE**(-QU_DELTA):
        raise ValueError("Input out of range")
    scaled = round(x * (QU_BASE ** QU_DELTA))
    return scaled % PRIME

def decode_quantized_value(z_mod_Q):
    half_Q = PRIME // 2
    unscaled = z_mod_Q / (QU_BASE ** QU_DELTA)
    if z_mod_Q > half_Q:
        unscaled -= PRIME / (QU_BASE ** QU_DELTA)
    max_value = QU_BASE**QU_GAMMA - QU_BASE**(-QU_DELTA)
    if unscaled < -QU_BASE**QU_GAMMA:
        unscaled += 2 * QU_BASE**QU_GAMMA
    elif unscaled > max_value:
        unscaled -= 2 * QU_BASE**QU_GAMMA
    return unscaled
    
# Simulation parameters
sampling_time = 0.15                        # Sampling time [s]
lookahead_distance = 0.1                    # Lookahead distance [m]
total_simulation_time = 355 * sampling_time # Total simulation time [s]
time_vector = np.arange(0, total_simulation_time, sampling_time)  # Time vector
num_time_steps = len(time_vector)

# Robot parameters
wheel_radius = 0.0205      # Wheel radius [m]
wheel_base = 0.053         # Wheelbase width [m]
max_wheel_speed = 10       # Max wheel angular speed [rad/s]

# Controller gains
proportional_gain = 4.0
integral_gain = 0.2

# Read reference trajectory from files
reference_x = np.loadtxt('./Reference Trajectories/xr.txt', delimiter=",")
reference_y = np.loadtxt('./Reference Trajectories/yr.txt', delimiter=",")
reference_theta = np.loadtxt('./Reference Trajectories/thetar.txt', delimiter=",")

# Robot initial conditions
current_x = 0.0
current_y = 0.0
current_x_rscode = 0.0
current_y_rscode = 0.0
current_theta = np.pi

# Initialize variables for storing simulation data
robot_trajectory_x = []
robot_trajectory_y = []
robot_trajectory_theta = []
position_error_x = []
position_error_y = []
absolute_position_error = []
control_input_x = []
control_input_y = []
control_input_x_rscode = []
control_input_y_rscode = []
u_x_shares = [0] * NUM_SHARES
u_y_shares = [0] * NUM_SHARES
control_input_x_rscode = []
control_input_y_rscode = []

# Controller states (integral components)
integral_state_x = np.zeros(num_time_steps + 1)
integral_state_y = np.zeros(num_time_steps + 1)
integral_state_x_rscode = np.zeros(num_time_steps + 1)
integral_state_y_rscode = np.zeros(num_time_steps + 1)
integral_state_x_qu = custom_quantize(integral_state_x_rscode[0])
integral_state_y_qu = custom_quantize(integral_state_y_rscode[0])
integral_state_x_shares = rscode.shamir_share(integral_state_x_qu)
integral_state_y_shares = rscode.shamir_share(integral_state_y_qu)
integral_state_x_shares = rscode.shares_noissy_channel(integral_state_x_shares, SEED)
integral_state_y_shares = rscode.shares_noissy_channel(integral_state_y_shares, SEED)


for k in range(num_time_steps):
    # Store current position
    robot_trajectory_x.append(current_x)
    robot_trajectory_y.append(current_y)
    robot_trajectory_theta.append(current_theta)

    # Reference signals at time step k
    ref_x = reference_x[k]
    ref_y = reference_y[k]
    ref_theta = reference_theta[k]

    # Compute position errors
    error_x = ref_x - current_x
    error_y = ref_y - current_y
    position_error_x.append(error_x)
    position_error_y.append(error_y)
    error_magnitude = np.sqrt(error_x**2 + error_y**2)
    absolute_position_error.append(error_magnitude)

    # Compute shifted reference and current positions
    shifted_ref_x = ref_x + lookahead_distance * np.cos(ref_theta)
    shifted_ref_y = ref_y + lookahead_distance * np.sin(ref_theta)
    shifted_current_x = current_x + lookahead_distance * np.cos(current_theta)
    shifted_current_y = current_y + lookahead_distance * np.sin(current_theta)

    # Compute errors in shifted positions
    error_shifted_x = shifted_ref_x - shifted_current_x
    error_shifted_y = shifted_ref_y - shifted_current_y
    error_shifted_x_qu = custom_quantize(error_shifted_x)
    error_shifted_y_qu = custom_quantize(error_shifted_y)
    error_shifted_x_shares = rscode.shamir_share(error_shifted_x_qu)
    error_shifted_y_shares = rscode.shamir_share(error_shifted_y_qu)
    error_shifted_x_shares = rscode.shares_noissy_channel(error_shifted_x_shares, SEED)
    error_shifted_y_shares = rscode.shares_noissy_channel(error_shifted_y_shares, SEED)
    # Compute control inputs using PI controller
    u_x = integral_gain * integral_state_x[k] + proportional_gain * error_shifted_x
    u_y = integral_gain * integral_state_y[k] + proportional_gain * error_shifted_y
    control_input_x.append(u_x)
    control_input_y.append(u_y)

    # Compute the Control Inputs Using the RSCode
    for share in range(NUM_SHARES): 
        u_x_shares[share] = custom_quantize(integral_gain) * integral_state_x_shares[share] + custom_quantize(proportional_gain) * error_shifted_x_shares[share]
        u_y_shares[share] = custom_quantize(integral_gain) * integral_state_y_shares[share] + custom_quantize(proportional_gain) * error_shifted_y_shares[share]
    u_x_shares = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in u_x_shares]
    u_y_shares = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in u_y_shares]
    # Reconstruct the Control Inputs
    u_x_rscode, error_indices = rscode.shamir_robust_reconstruct(u_x_shares)
    u_y_rscode, error_indices = rscode.shamir_robust_reconstruct(u_y_shares)
    u_x_rscode = decode_quantized_value(decode_quantized_value(u_x_rscode))
    u_y_rscode = decode_quantized_value(decode_quantized_value(u_y_rscode))
    control_input_x_rscode.append(u_x_rscode)
    control_input_y_rscode.append(u_y_rscode)
    # Update integral states
    integral_state_x[k + 1] = integral_state_x[k] + sampling_time * error_shifted_x
    integral_state_y[k + 1] = integral_state_y[k] + sampling_time * error_shifted_y
    integral_state_x_qu = custom_quantize(integral_state_x_rscode[k + 1])
    integral_state_y_qu = custom_quantize(integral_state_y_rscode[k + 1])
    integral_state_x_shares = rscode.shamir_share(integral_state_x_qu)
    integral_state_y_shares = rscode.shamir_share(integral_state_y_qu)  
    integral_state_x_shares = rscode.shares_noissy_channel(integral_state_x_shares, SEED)
    integral_state_y_shares = rscode.shares_noissy_channel(integral_state_y_shares, SEED)

    # Compute linear and angular velocities
    transformation_inverse = np.array([
        [np.cos(current_theta), np.sin(current_theta)],
        [-np.sin(current_theta) / lookahead_distance, np.cos(current_theta) / lookahead_distance]
    ])
    control_vector = np.array([u_x, u_y])
    control_vector_rscode = np.array([u_x_rscode, u_y_rscode])
    velocities = transformation_inverse @ control_vector
    velocities_rscode = transformation_inverse @ control_vector_rscode
    linear_velocity = velocities[0]
    angular_velocity = velocities[1]
    linear_velocity_rscode = velocities_rscode[0]
    angular_velocity_rscode = velocities_rscode[1]

    # Saturate velocities to robot's physical limits
    max_linear_velocity = wheel_radius * max_wheel_speed
    max_angular_velocity = 2 * max_wheel_speed * wheel_radius / wheel_base
    linear_velocity = np.clip(linear_velocity, -max_linear_velocity, max_linear_velocity)
    angular_velocity = np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity)
    linear_velocity_rscode = np.clip(linear_velocity_rscode, -max_linear_velocity, max_linear_velocity)
    angular_velocity_rscode = np.clip(angular_velocity_rscode, -max_angular_velocity, max_angular_velocity)

    # Update robot's state
    # current_x += sampling_time * linear_velocity * np.cos(current_theta)
    # current_y += sampling_time * linear_velocity * np.sin(current_theta)
    # current_theta += sampling_time * angular_velocity

    current_x += sampling_time * linear_velocity_rscode * np.cos(current_theta)
    current_y += sampling_time * linear_velocity_rscode * np.sin(current_theta)
    current_theta += sampling_time * angular_velocity_rscode

    print(angular_velocity, angular_velocity_rscode)

    # Wrap current_theta to [-π, π]
    current_theta = (current_theta + np.pi) % (2 * np.pi) - np.pi

# Convert lists to numpy arrays
robot_trajectory_x = np.array(robot_trajectory_x)
robot_trajectory_y = np.array(robot_trajectory_y)
robot_trajectory_theta = np.array(robot_trajectory_theta)
position_error_x = np.array(position_error_x)
position_error_y = np.array(position_error_y)
absolute_position_error = np.array(absolute_position_error)
control_input_x = np.array(control_input_x)
control_input_y = np.array(control_input_y)

# Plotting the robot trajectory vs. reference trajectory
plt.figure()
plt.plot(reference_x, reference_y, 'r--', label='Reference Trajectory')
plt.plot(robot_trajectory_x, robot_trajectory_y, 'b-', label='Robot Trajectory')
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.title('Robot Trajectory Tracking')
plt.show()

# Plotting position errors over time
plt.figure()
plt.plot(time_vector, position_error_x, label='X Error')
plt.plot(time_vector, position_error_y, label='Y Error')
plt.xlabel('Time [s]')
plt.ylabel('Position Error [m]')
plt.legend()
plt.grid(True)
plt.title('Tracking Errors')
plt.show()