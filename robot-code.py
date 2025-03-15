import numpy as np
import matplotlib.pyplot as plt
from RSSecretSharing import RSSecretSharing
import math
import random

# Global Quantization Parameters
QU_BASE = 2
QU_GAMMA = 100
QU_DELTA = 10
#Global Parameters / Sharing 
PRIME = 18446744073709551557
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

# Initialize variables for storing simulation data
robot_trajectory_x = []
robot_trajectory_y = []
robot_trajectory_theta = []
robot_trajectory_x_shamir = []  # New trajectory for shamir_decode
robot_trajectory_y_shamir = []  # New trajectory for shamir_decode
robot_trajectory_theta_shamir = []  # New trajectory for shamir_decode
position_error_x = []
position_error_x_shamir = []  # New position errors for shamir_decode
position_error_y_shamir = []  # New position errors for shamir_decode
absolute_position_error_shamir = []  # New absolute position errors for shamir_decode
position_error_y = []
absolute_position_error = []
control_input_x = []
control_input_y = []
control_input_x_rscode = []
control_input_y_rscode = []
control_input_x_shamir = []  # New control inputs for shamir_decode
control_input_y_shamir = []  # New control inputs for shamir_decode

# Controller states (integral components)
rscode_error_index = np.zeros(num_time_steps + 1)
rscode_nan_index = np.zeros(num_time_steps + 1)
integral_state_x_rscode = np.zeros(num_time_steps + 1)
integral_state_y_rscode = np.zeros(num_time_steps + 1)
integral_state_x_shamir = np.zeros(num_time_steps + 1)  # New integral states for shamir_decode
integral_state_y_shamir = np.zeros(num_time_steps + 1)  # New integral states for shamir_decode
integral_state_x_qu = custom_quantize(integral_state_x_rscode[0])
integral_state_y_qu = custom_quantize(integral_state_y_rscode[0])
integral_state_x_shares = rscode.shamir_share(integral_state_x_qu)
integral_state_y_shares = rscode.shamir_share(integral_state_y_qu)
# Robot initial conditions for both versions
current_x = 0.0
current_y = 0.0
current_theta = np.pi
# -------------------- RS Code Simulation Loop --------------------
for k in range(num_time_steps):
    # Store current position for RS Code
    robot_trajectory_x.append(current_x)
    robot_trajectory_y.append(current_y)
    robot_trajectory_theta.append(current_theta)
    # Reference signals at time step k
    ref_x = reference_x[k]
    ref_y = reference_y[k]
    ref_theta = reference_theta[k]
    # Compute position errors for RS Code
    error_x = ref_x - current_x
    error_y = ref_y - current_y
    position_error_x.append(error_x)
    position_error_y.append(error_y)
    error_magnitude = np.sqrt(error_x**2 + error_y**2)
    absolute_position_error.append(error_magnitude)
    # Compute shifted reference and current positions for RS Code
    shifted_ref_x = ref_x + lookahead_distance * np.cos(ref_theta)
    shifted_ref_y = ref_y + lookahead_distance * np.sin(ref_theta)
    shifted_current_x = current_x + lookahead_distance * np.cos(current_theta)
    shifted_current_y = current_y + lookahead_distance * np.sin(current_theta)
    # Compute errors in shifted positions for RS Code
    error_shifted_x = shifted_ref_x - shifted_current_x
    error_shifted_y = shifted_ref_y - shifted_current_y
    error_shifted_x_qu = custom_quantize(error_shifted_x)
    error_shifted_y_qu = custom_quantize(error_shifted_y)
    error_shifted_x_shares = rscode.shamir_share(error_shifted_x_qu)
    error_shifted_y_shares = rscode.shamir_share(error_shifted_y_qu)
    error_shifted_x_shares = rscode.shares_noissy_channel(error_shifted_x_shares, SEED)
    error_shifted_y_shares = rscode.shares_noissy_channel(error_shifted_y_shares, SEED)
    #
    u_x_shares = [0] * NUM_SHARES
    u_y_shares = [0] * NUM_SHARES
    integral_state_x_shares_new = [0] * NUM_SHARES
    integral_state_y_shares_new = [0] * NUM_SHARES
    proportional_gain_qu = custom_quantize(proportional_gain)
    integral_gain_qu = custom_quantize(integral_gain)
    sampling_time_qu = custom_quantize(sampling_time)
    # Compute the Control Inputs Using RS Code
    for share in range(NUM_SHARES):
        integral_state_x_shares_new[share] = (integral_state_x_shares[share] + sampling_time_qu * error_shifted_x_shares[share]) % PRIME
        integral_state_y_shares_new[share] = (integral_state_y_shares[share] + sampling_time_qu * error_shifted_y_shares[share]) % PRIME
        u_x_shares[share] = (integral_gain_qu * integral_state_x_shares[share] + proportional_gain_qu * error_shifted_x_shares[share]) % PRIME
        u_y_shares[share] = (integral_gain_qu * integral_state_y_shares[share] + proportional_gain_qu * error_shifted_y_shares[share]) % PRIME
    # Handle NaN values
    u_x_shares = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in u_x_shares]
    u_y_shares = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in u_y_shares]
    integral_state_x_shares_new = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in integral_state_x_shares_new]
    integral_state_y_shares_new = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in integral_state_y_shares_new]
    # Reconstruct the Control Inputs using RS Code
    u_x_shares = [int(x) if x is not None else x for x in u_x_shares]
    u_x_rscode, rscode_error_index[k], rscode_nan_index[k] = rscode.shamir_robust_reconstruct(u_x_shares)
    u_y_shares = [int(x) if x is not None else x for x in u_y_shares]
    u_y_rscode, _, _ = rscode.shamir_robust_reconstruct(u_y_shares)
    u_x_rscode = decode_quantized_value(decode_quantized_value(u_x_rscode))
    u_y_rscode = decode_quantized_value(decode_quantized_value(u_y_rscode))
    control_input_x_rscode.append(u_x_rscode)
    control_input_y_rscode.append(u_y_rscode)
    #
    SEED = random.randint(0, 1000000)
    # Update integral states for RS Code
    integral_state_x_re, _, _ = rscode.shamir_robust_reconstruct(integral_state_x_shares_new)
    integral_state_y_re, _, _ = rscode.shamir_robust_reconstruct(integral_state_y_shares_new)
    integral_state_x_rscode[k + 1] = decode_quantized_value(decode_quantized_value(integral_state_x_re))
    integral_state_y_rscode[k + 1] = decode_quantized_value(decode_quantized_value(integral_state_y_re))
    integral_state_x_qu = custom_quantize(integral_state_x_rscode[k + 1])
    integral_state_y_qu = custom_quantize(integral_state_y_rscode[k + 1])
    integral_state_x_shares = rscode.shamir_share(integral_state_x_qu)
    integral_state_y_shares = rscode.shamir_share(integral_state_y_qu)
    integral_state_x_shares = rscode.shares_noissy_channel(integral_state_x_shares, SEED)
    integral_state_y_shares = rscode.shares_noissy_channel(integral_state_y_shares, SEED)
    # Compute linear and angular velocities for RS Code
    transformation_inverse = np.array([
        [np.cos(current_theta), np.sin(current_theta)],
        [-np.sin(current_theta) / lookahead_distance, np.cos(current_theta) / lookahead_distance]
    ])
    control_vector_rscode = np.array([u_x_rscode, u_y_rscode])
    velocities_rscode = transformation_inverse @ control_vector_rscode
    linear_velocity_rscode = velocities_rscode[0]
    angular_velocity_rscode = velocities_rscode[1]
    # Saturate velocities to robot's physical limits for RS Code
    max_linear_velocity = wheel_radius * max_wheel_speed
    max_angular_velocity = 2 * max_wheel_speed * wheel_radius / wheel_base
    linear_velocity_rscode = np.clip(linear_velocity_rscode, -max_linear_velocity, max_linear_velocity)
    angular_velocity_rscode = np.clip(angular_velocity_rscode, -max_angular_velocity, max_angular_velocity)
    # Update robot position for RS Code
    current_x += sampling_time * linear_velocity_rscode * np.cos(current_theta)
    current_y += sampling_time * linear_velocity_rscode * np.sin(current_theta)
    current_theta += sampling_time * angular_velocity_rscode
    # Wrap current_theta to [-π, π] for RS Code
    current_theta = (current_theta + np.pi) % (2 * np.pi) - np.pi
# -------------------- Shamir Simulation Loop --------------------
current_x_shamir = 0.0
current_y_shamir = 0.0
current_theta_shamir = np.pi
for k in range(num_time_steps):
    # Store current position for Shamir
    robot_trajectory_x_shamir.append(current_x_shamir)
    robot_trajectory_y_shamir.append(current_y_shamir)
    robot_trajectory_theta_shamir.append(current_theta_shamir)
    # Reference signals at time step k
    ref_x = reference_x[k]
    ref_y = reference_y[k]
    ref_theta = reference_theta[k]
    # Compute position errors for Shamir
    error_x_shamir = ref_x - current_x_shamir
    error_y_shamir = ref_y - current_y_shamir
    position_error_x_shamir.append(error_x_shamir)
    position_error_y_shamir.append(error_y_shamir)
    error_magnitude_shamir = np.sqrt(error_x_shamir**2 + error_y_shamir**2)
    absolute_position_error_shamir.append(error_magnitude_shamir)
    # Compute shifted reference and current positions for Shamir
    shifted_ref_x = ref_x + lookahead_distance * np.cos(ref_theta)
    shifted_ref_y = ref_y + lookahead_distance * np.sin(ref_theta)
    shifted_current_x_shamir = current_x_shamir + lookahead_distance * np.cos(current_theta_shamir)
    shifted_current_y_shamir = current_y_shamir + lookahead_distance * np.sin(current_theta_shamir)
    # Compute errors in shifted positions for Shamir
    error_shifted_x_shamir = shifted_ref_x - shifted_current_x_shamir
    error_shifted_y_shamir = shifted_ref_y - shifted_current_y_shamir
    error_shifted_x_qu_shamir = custom_quantize(error_shifted_x_shamir)
    error_shifted_y_qu_shamir = custom_quantize(error_shifted_y_shamir)
    error_shifted_x_shares_shamir = rscode.shamir_share(error_shifted_x_qu_shamir)
    error_shifted_y_shares_shamir = rscode.shamir_share(error_shifted_y_qu_shamir)
    error_shifted_x_shares_shamir = rscode.shares_noissy_channel(error_shifted_x_shares_shamir, SEED)
    error_shifted_y_shares_shamir = rscode.shares_noissy_channel(error_shifted_y_shares_shamir, SEED)
    #
    u_x_shares_shamir = [0] * NUM_SHARES
    u_y_shares_shamir = [0] * NUM_SHARES
    integral_state_x_shares_new_shamir = [0] * NUM_SHARES
    integral_state_y_shares_new_shamir = [0] * NUM_SHARES
    proportional_gain_qu_shamir = custom_quantize(proportional_gain)
    integral_gain_qu_shamir = custom_quantize(integral_gain)
    sampling_time_qu_shamir = custom_quantize(sampling_time)
    # Compute the Control Inputs Using Shamir
    for share in range(NUM_SHARES):
        integral_state_x_shares_new_shamir[share] = (integral_state_x_shares[share] + sampling_time_qu_shamir * error_shifted_x_shares_shamir[share]) % PRIME
        integral_state_y_shares_new_shamir[share] = (integral_state_y_shares[share] + sampling_time_qu_shamir * error_shifted_y_shares_shamir[share]) % PRIME
        u_x_shares_shamir[share] = (integral_gain_qu_shamir * integral_state_x_shares[share] + proportional_gain_qu_shamir * error_shifted_x_shares_shamir[share]) % PRIME
        u_y_shares_shamir[share] = (integral_gain_qu_shamir * integral_state_y_shares[share] + proportional_gain_qu_shamir * error_shifted_y_shares_shamir[share]) % PRIME
    # Handle NaN values
    u_x_shares_shamir = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in u_x_shares_shamir]
    u_y_shares_shamir = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in u_y_shares_shamir]
    integral_state_x_shares_new_shamir = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in integral_state_x_shares_new_shamir]
    integral_state_y_shares_new_shamir = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in integral_state_y_shares_new_shamir]
    # Reconstruct the Control Inputs using Shamir
    u_x_shares_shamir = [int(x) if x is not None else x for x in u_x_shares_shamir]
    u_x_shamir = rscode.shamir_decode(u_x_shares_shamir)
    u_y_shares_shamir = [int(x) if x is not None else x for x in u_y_shares_shamir]
    u_y_shamir = rscode.shamir_decode(u_y_shares_shamir)
    u_x_shamir = decode_quantized_value(decode_quantized_value(u_x_shamir))
    u_y_shamir = decode_quantized_value(decode_quantized_value(u_y_shamir))
    control_input_x_shamir.append(u_x_shamir)
    control_input_y_shamir.append(u_y_shamir)
    #
    SEED = random.randint(0, 1000000)
    # Update integral states for Shamir
    integral_state_x_re_shamir = rscode.shamir_decode(integral_state_x_shares_new_shamir)
    integral_state_y_re_shamir = rscode.shamir_decode(integral_state_y_shares_new_shamir)
    integral_state_x_shamir[k + 1] = decode_quantized_value(decode_quantized_value(integral_state_x_re_shamir))
    integral_state_y_shamir[k + 1] = decode_quantized_value(decode_quantized_value(integral_state_y_re_shamir))
    integral_state_x_qu = custom_quantize(integral_state_x_shamir[k + 1])
    integral_state_y_qu = custom_quantize(integral_state_y_shamir[k + 1])
    integral_state_x_shares = rscode.shamir_share(integral_state_x_qu)
    integral_state_y_shares = rscode.shamir_share(integral_state_y_qu)
    integral_state_x_shares = rscode.shares_noissy_channel(integral_state_x_shares, SEED)
    integral_state_y_shares = rscode.shares_noissy_channel(integral_state_y_shares, SEED)
    # Compute linear and angular velocities for Shamir
    transformation_inverse_shamir = np.array([
        [np.cos(current_theta_shamir), np.sin(current_theta_shamir)],
        [-np.sin(current_theta_shamir) / lookahead_distance, np.cos(current_theta_shamir) / lookahead_distance]
    ])
    control_vector_shamir = np.array([u_x_shamir, u_y_shamir])
    velocities_shamir = transformation_inverse_shamir @ control_vector_shamir
    linear_velocity_shamir = velocities_shamir[0]
    angular_velocity_shamir = velocities_shamir[1]
    # Saturate velocities to robot's physical limits for Shamir
    max_linear_velocity = wheel_radius * max_wheel_speed
    max_angular_velocity = 2 * max_wheel_speed * wheel_radius / wheel_base
    linear_velocity_shamir = np.clip(linear_velocity_shamir, -max_linear_velocity, max_linear_velocity)
    angular_velocity_shamir = np.clip(angular_velocity_shamir, -max_angular_velocity, max_angular_velocity)
    # Update robot position for Shamir
    current_x_shamir += sampling_time * linear_velocity_shamir * np.cos(current_theta_shamir)
    current_y_shamir += sampling_time * linear_velocity_shamir * np.sin(current_theta_shamir)
    current_theta_shamir += sampling_time * angular_velocity_shamir
    # Wrap current_theta to [-π, π] for Shamir
    current_theta_shamir = (current_theta_shamir + np.pi) % (2 * np.pi) - np.pi

# Convert lists to numpy arrays
robot_trajectory_x = np.array(robot_trajectory_x)
robot_trajectory_y = np.array(robot_trajectory_y)
robot_trajectory_theta = np.array(robot_trajectory_theta)
robot_trajectory_x_shamir = np.array(robot_trajectory_x_shamir)
robot_trajectory_y_shamir = np.array(robot_trajectory_y_shamir)
robot_trajectory_theta_shamir = np.array(robot_trajectory_theta_shamir)
position_error_x = np.array(position_error_x)
position_error_y = np.array(position_error_y)
absolute_position_error = np.array(absolute_position_error)
position_error_x_shamir = np.array(position_error_x_shamir)
position_error_y_shamir = np.array(position_error_y_shamir)
absolute_position_error_shamir = np.array(absolute_position_error_shamir)
# Control inputs
control_input_x_rscode = np.array(control_input_x_rscode)
control_input_y_rscode = np.array(control_input_y_rscode)
control_input_x_shamir = np.array(control_input_x_shamir)
control_input_y_shamir = np.array(control_input_y_shamir)
# -------------------- Plotting --------------------
# Plotting the robot trajectories vs. reference trajectory
plt.figure(figsize=(12, 8))
plt.plot(reference_x, reference_y, 'r--', label='Reference Trajectory', linewidth=4.5)
plt.plot(robot_trajectory_x, robot_trajectory_y, 'b-', label='Robot Trajectory (Reed-Solomon)')
plt.plot(robot_trajectory_x_shamir, robot_trajectory_y_shamir, 'g-', label='Robot Trajectory (Shamir)')
plt.plot(0, 0, 'k*', markersize=15, label='Start Point')  # Add black star at starting point
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.legend(loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.title('Robot Trajectory Tracking Comparison')
plt.savefig('trajectory_tracking_comparison.eps', format='eps', bbox_inches='tight')
plt.show()
#
# Plotting position errors over time
plt.figure(figsize=(12, 8))
ax1 = plt.gca()
ax2 = ax1.twinx()
# Plot Robust errors on left axis
line1 = ax1.plot(time_vector, position_error_x, 'b-', label='X Error (Reed-Solomon)')
line2 = ax1.plot(time_vector, position_error_y, 'b--', label='Y Error (Reed-Solomon)')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Position Error [m] (Reed-Solomon)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
# Plot Shamir errors on right axis
line3 = ax2.plot(time_vector, position_error_x_shamir, 'g-', label='X Error (Shamir)')
line4 = ax2.plot(time_vector, position_error_y_shamir, 'g--', label='Y Error (Shamir)')
ax2.set_ylabel('Position Error [m] (Shamir)', color='g')
ax2.tick_params(axis='y', labelcolor='g')
# Combine lines for legend
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')
plt.grid(True)
plt.title('Tracking Errors')
plt.savefig('tracking_errors.eps', format='eps', bbox_inches='tight')
plt.show()
#
# Number of time steps to display in the table
TABLE_DISPLAY_STEPS = 20
# Create a table visualization showing error and NaN indices
display_steps = min(TABLE_DISPLAY_STEPS, num_time_steps)  # Limit to TABLE_DISPLAY_STEPS or less
# Calculate figure size to maintain square cells
# Adjust the figure size based on the ratio of rows to columns
cell_aspect_ratio = NUM_SHARES / display_steps
if cell_aspect_ratio > 1:
    # More rows than columns, make figure wider
    fig_width = 10
    fig_height = 10 / cell_aspect_ratio
else:
    # More columns than rows, make figure taller
    fig_height = 10
    fig_width = 10 * cell_aspect_ratio
# Ensure minimum dimensions
fig_width = max(fig_width, 6)
fig_height = max(fig_height, 6)
plt.figure(figsize=(fig_width, fig_height))
# Create a matrix of all green cells initially
cell_colors = np.full((NUM_SHARES, display_steps), 'green')
# Mark cells with errors as red and cells with NaN as black
for t in range(display_steps):
    if rscode_error_index[t] >= 0 and rscode_error_index[t] < NUM_SHARES:
        cell_colors[int(rscode_error_index[t]), t] = 'red'
    if rscode_nan_index[t] >= 0 and rscode_nan_index[t] < NUM_SHARES:
        cell_colors[int(rscode_nan_index[t]), t] = 'black'
# Create a table with colored cells
table = plt.table(
    cellColours=cell_colors,
    cellLoc='center',
    loc='center',
    rowLabels=[f'Share {i}' for i in range(NUM_SHARES)],
    colLabels=[f'{i}' for i in range(display_steps)],
)
# Adjust table appearance
table.auto_set_font_size(False)
table.set_fontsize(9)
# Scale to maintain square cells
table.scale(1, 1)
# Remove axes
plt.axis('off')
plt.title(f'Share Status for First {display_steps} Time Steps (Green: OK, Red: Error, Black: NaN)')
# Add a legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='OK'),
    Patch(facecolor='red', label='Error'),
    Patch(facecolor='black', label='NaN')
]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05))
plt.tight_layout()
plt.savefig('share_status_table.eps', format='eps', dpi=1200)
plt.show()