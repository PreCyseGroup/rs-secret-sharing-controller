import numpy as np
import matplotlib.pyplot as plt
from RSSecretSharing import RSSecretSharing
import math
import scipy.io  # Ensure this import is at the top of your file
SC_SWIT_TIME = 15
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
NON_RESPONSIVE_CLOUD= [2] 
MALICIOUS_CLOUD = [0]
rscode = RSSecretSharing(PRIME, 1, NUM_SHARES, T_POLY_DEGREE, MAX_MISSING, MAX_MANIPULATED, SEED)
##
def custom_quantize(x):
    if x < -QU_BASE**QU_GAMMA or x > QU_BASE**QU_GAMMA - QU_BASE**(-QU_DELTA):
        raise ValueError("Input out of range")
    scaled = round(x * (QU_BASE ** QU_DELTA))
    return scaled % PRIME
##
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
##
# Simulation parameters
sampling_time = 0.15                        # Sampling time [s]
lookahead_distance = 0.1                    # Lookahead distance [m]
total_simulation_time = 355 * sampling_time # Total simulation time [s]
time_vector = np.arange(0, total_simulation_time, sampling_time)  # Time vector
num_time_steps = len(time_vector)
# Robot parameters
wheel_radius = 0.0205      # Wheel radius [m]
wheel_base = 0.053         # Wheelbase width [m]
max_wheel_speed = 10     # Max Angular Velocity
conversion_matrix_robot = np.array([[wheel_radius / 2, wheel_radius / 2], 
                                     [wheel_radius / wheel_base, -wheel_radius / wheel_base]])
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
robot_trajectory_x_shamir = []  
robot_trajectory_y_shamir = []  
robot_trajectory_theta_shamir = []  
position_error_x = []
position_error_x_shamir = []  
position_error_y_shamir = []  
absolute_position_error_shamir = []  
position_error_y = []
absolute_position_error = []
control_input_x = []
control_input_y = []
control_input_x_rscode = []
control_input_y_rscode = []
control_input_x_shamir = []  
control_input_y_shamir = []  
# Controller states (integral components)
rscode_error_index = np.zeros(num_time_steps + 1)
rscode_nan_index = np.zeros(num_time_steps + 1)
integral_state_x_rscode = np.zeros(num_time_steps + 1)
integral_state_y_rscode = np.zeros(num_time_steps + 1)
integral_state_x_shamir = np.zeros(num_time_steps + 1) 
integral_state_y_shamir = np.zeros(num_time_steps + 1)  
integral_state_x_qu = custom_quantize(integral_state_x_rscode[0])
integral_state_y_qu = custom_quantize(integral_state_y_rscode[0])
integral_state_x_shares = rscode.shamir_share(integral_state_x_qu)
integral_state_y_shares = rscode.shamir_share(integral_state_y_qu)
# Robot initial conditions for both versions
current_x = 0.0
current_y = 0.0
current_theta = np.pi
# Initialize lists to store velocities and wheel speeds
linear_velocities_rscode = []
angular_velocities_rscode = []
linear_velocities_shamir = []
angular_velocities_shamir = []
right_wheel_speeds_rscode = []
left_wheel_speeds_rscode = []
right_wheel_speeds_shamir = []
left_wheel_speeds_shamir = []
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
    ##
    ### Scenarios Conditions / Begin
    if k * sampling_time < SC_SWIT_TIME: 
        print(f"Scenario 1 - Step {k + 1}")
    elif (k * sampling_time) >= SC_SWIT_TIME and k * sampling_time < (2*SC_SWIT_TIME):
        print(f"Scenario 2 - Step {k + 1}")
        error_shifted_x_shares = rscode.simulate_missing_data(error_shifted_x_shares, NON_RESPONSIVE_CLOUD)
        error_shifted_x_shares = error_shifted_x_shares[0]
        error_shifted_y_shares = rscode.simulate_missing_data(error_shifted_y_shares, NON_RESPONSIVE_CLOUD)
        error_shifted_y_shares = error_shifted_y_shares[0]
    else:
        print(f"Scenario 3 - Step {k + 1}")
        error_shifted_x_shares = rscode.simulate_missing_data(error_shifted_x_shares, NON_RESPONSIVE_CLOUD)
        error_shifted_x_shares = error_shifted_x_shares[0]
        error_shifted_x_shares = rscode.simulate_manipulated_data(error_shifted_x_shares, MALICIOUS_CLOUD)
        error_shifted_x_shares = error_shifted_x_shares[0]
        error_shifted_y_shares = rscode.simulate_missing_data(error_shifted_y_shares, NON_RESPONSIVE_CLOUD)
        error_shifted_y_shares = error_shifted_y_shares[0]
        error_shifted_y_shares = rscode.simulate_manipulated_data(error_shifted_y_shares, MALICIOUS_CLOUD)
        error_shifted_y_shares = error_shifted_y_shares[0]
    ### Scenarios Conditions / End
    ##
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
    # Update integral states for RS Code
    integral_state_x_re, _, _ = rscode.shamir_robust_reconstruct(integral_state_x_shares_new)
    integral_state_y_re, _, _ = rscode.shamir_robust_reconstruct(integral_state_y_shares_new)
    integral_state_x_rscode[k + 1] = decode_quantized_value(decode_quantized_value(integral_state_x_re))
    integral_state_y_rscode[k + 1] = decode_quantized_value(decode_quantized_value(integral_state_y_re))
    integral_state_x_qu = custom_quantize(integral_state_x_rscode[k + 1])
    integral_state_y_qu = custom_quantize(integral_state_y_rscode[k + 1])
    integral_state_x_shares = rscode.shamir_share(integral_state_x_qu)
    integral_state_y_shares = rscode.shamir_share(integral_state_y_qu)
    ##
    ### Scenarios Conditions / Begin
    if k * sampling_time < SC_SWIT_TIME:  
        print(f"Scenario 1 - Step {k + 1}")
    elif (k * sampling_time) >= SC_SWIT_TIME and k * sampling_time < (2*SC_SWIT_TIME):
        print(f"Scenario 2 - Step {k + 1}")
        integral_state_x_shares = rscode.simulate_missing_data(integral_state_x_shares, NON_RESPONSIVE_CLOUD)
        integral_state_x_shares = integral_state_x_shares[0]
        integral_state_y_shares = rscode.simulate_missing_data(integral_state_y_shares, NON_RESPONSIVE_CLOUD)
        integral_state_y_shares = integral_state_y_shares[0]
    else:
        print(f"Scenario 3 - Step {k + 1}")
        integral_state_x_shares = rscode.simulate_missing_data(integral_state_x_shares, NON_RESPONSIVE_CLOUD)
        integral_state_x_shares = integral_state_x_shares[0]
        print(integral_state_x_shares)
        integral_state_x_shares = rscode.simulate_manipulated_data(integral_state_x_shares, MALICIOUS_CLOUD)
        integral_state_x_shares = integral_state_x_shares[0]
        print(integral_state_x_shares)
        integral_state_y_shares = rscode.simulate_missing_data(integral_state_y_shares, NON_RESPONSIVE_CLOUD)
        integral_state_y_shares = integral_state_y_shares[0]
        integral_state_y_shares = rscode.simulate_manipulated_data(integral_state_y_shares, MALICIOUS_CLOUD)
        integral_state_y_shares = integral_state_y_shares[0]
    ### Scenarios Conditions / End
    ##
    # Compute linear and angular velocities for RS Code
    transformation_inverse = np.array([
        [np.cos(current_theta), np.sin(current_theta)],
        [-np.sin(current_theta) / lookahead_distance, np.cos(current_theta) / lookahead_distance]
    ])
    control_vector_rscode = np.array([u_x_rscode, u_y_rscode])
    wheel_velocities_rscode = np.linalg.inv(conversion_matrix_robot) @ transformation_inverse @ control_vector_rscode
    omega_r_rscode = wheel_velocities_rscode[0]
    omega_l_rscode = wheel_velocities_rscode[1]
    omega_r_rscode = np.clip(omega_r_rscode, -max_wheel_speed, max_wheel_speed)
    omega_l_rscode = np.clip(omega_l_rscode, -max_wheel_speed, max_wheel_speed)
    left_wheel_speeds_rscode.append(omega_l_rscode)
    right_wheel_speeds_rscode.append(omega_r_rscode)
    velocities_rscode = conversion_matrix_robot @ np.array([[omega_r_rscode], [omega_l_rscode]])
    linear_velocity_rscode = velocities_rscode[0]
    angular_velocity_rscode = velocities_rscode[1]
    linear_velocities_rscode.append(linear_velocity_rscode)
    angular_velocities_rscode.append(angular_velocity_rscode)
    current_x += sampling_time * linear_velocity_rscode * np.cos(current_theta)
    print(current_x, current_y, current_theta)
    current_y += sampling_time * linear_velocity_rscode * np.sin(current_theta)
    current_theta += sampling_time * angular_velocity_rscode
    current_theta = (current_theta + np.pi) % (2 * np.pi) - np.pi
    current_x = current_x[0]
    current_y = current_y[0]
    current_theta = current_theta[0]
    print(current_x, current_y, current_theta)
# -------------------- Shamir Simulation Loop --------------------
current_x_shamir = 0.0
current_y_shamir = 0.0
current_theta_shamir = np.pi
integral_state_x_shares = rscode.shamir_share(0)
integral_state_y_shares = rscode.shamir_share(0)
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
    ##
    # ### Scenarios Conditions / Begin
    if k * sampling_time < SC_SWIT_TIME: 
        print(f"Scenario 1 - Step {k + 1}")
    elif (k * sampling_time) >= SC_SWIT_TIME and k * sampling_time < (2*SC_SWIT_TIME):
        print(f"Scenario 2 - Step {k + 1}")
        error_shifted_x_shares_shamir = rscode.simulate_missing_data(error_shifted_x_shares_shamir, NON_RESPONSIVE_CLOUD)
        error_shifted_x_shares_shamir = error_shifted_x_shares_shamir[0]
        error_shifted_y_shares_shamir = rscode.simulate_missing_data(error_shifted_y_shares_shamir, NON_RESPONSIVE_CLOUD)
        error_shifted_y_shares_shamir = error_shifted_y_shares_shamir[0]
    else:
        print(f"Scenario 3 - Step {k + 1}")
        error_shifted_x_shares_shamir = rscode.simulate_missing_data(error_shifted_x_shares_shamir, NON_RESPONSIVE_CLOUD)
        error_shifted_x_shares_shamir = error_shifted_x_shares_shamir[0]
        error_shifted_x_shares_shamir = rscode.simulate_manipulated_data(error_shifted_x_shares_shamir, MALICIOUS_CLOUD)
        error_shifted_x_shares_shamir = error_shifted_x_shares_shamir[0]
        error_shifted_y_shares_shamir = rscode.simulate_missing_data(error_shifted_y_shares_shamir, NON_RESPONSIVE_CLOUD)
        error_shifted_y_shares_shamir = error_shifted_y_shares_shamir[0]
        error_shifted_y_shares_shamir = rscode.simulate_manipulated_data(error_shifted_y_shares_shamir, MALICIOUS_CLOUD)
        error_shifted_y_shares_shamir = error_shifted_y_shares_shamir[0]
    ### Scenarios Conditions / End 
    ##
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
    # Update integral states for Shamir
    integral_state_x_re_shamir = rscode.shamir_decode(integral_state_x_shares_new_shamir)
    integral_state_y_re_shamir = rscode.shamir_decode(integral_state_y_shares_new_shamir)
    integral_state_x_shamir[k + 1] = decode_quantized_value(decode_quantized_value(integral_state_x_re_shamir))
    integral_state_y_shamir[k + 1] = decode_quantized_value(decode_quantized_value(integral_state_y_re_shamir))
    integral_state_x_qu = custom_quantize(integral_state_x_shamir[k + 1])
    integral_state_y_qu = custom_quantize(integral_state_y_shamir[k + 1])
    integral_state_x_shares = rscode.shamir_share(integral_state_x_qu)
    integral_state_y_shares = rscode.shamir_share(integral_state_y_qu)
    ###
    ### Scenarios Conditions / Begin
    if k * sampling_time < SC_SWIT_TIME: 
        print(f"Scenario 1 - Step {k + 1}")
    elif (k * sampling_time) >= SC_SWIT_TIME and k * sampling_time < (2*SC_SWIT_TIME):
        print(f"Scenario 2 - Step {k + 1}")
        integral_state_x_shares = rscode.simulate_missing_data(integral_state_x_shares, NON_RESPONSIVE_CLOUD)
        integral_state_x_shares = integral_state_x_shares[0]
        integral_state_y_shares = rscode.simulate_missing_data(integral_state_y_shares, NON_RESPONSIVE_CLOUD)
        integral_state_y_shares = integral_state_y_shares[0]
    else:
        print(f"Scenario 3 - Step {k + 1}")
        integral_state_x_shares = rscode.simulate_missing_data(integral_state_x_shares, NON_RESPONSIVE_CLOUD)
        integral_state_x_shares = integral_state_x_shares[0]
        print(integral_state_x_shares)
        integral_state_x_shares = rscode.simulate_manipulated_data(integral_state_x_shares, MALICIOUS_CLOUD)
        integral_state_x_shares = integral_state_x_shares[0]
        print(integral_state_x_shares)
        integral_state_y_shares = rscode.simulate_missing_data(integral_state_y_shares, NON_RESPONSIVE_CLOUD)
        integral_state_y_shares = integral_state_y_shares[0]
        integral_state_y_shares = rscode.simulate_manipulated_data(integral_state_y_shares, MALICIOUS_CLOUD)
        integral_state_y_shares = integral_state_y_shares[0]
    ### Scenarios Conditions / End
    ###
    # Compute linear and angular velocities for Shamir
    transformation_inverse_shamir = np.array([
        [np.cos(current_theta_shamir), np.sin(current_theta_shamir)],
        [-np.sin(current_theta_shamir) / lookahead_distance, np.cos(current_theta_shamir) / lookahead_distance]
    ])
    control_vector_shamir = np.array([u_x_shamir, u_y_shamir])
    wheel_velocities_shamir = np.linalg.inv(conversion_matrix_robot) @ transformation_inverse_shamir @ control_vector_shamir
    omega_r_shamir = wheel_velocities_shamir[0]
    omega_l_shamir = wheel_velocities_shamir[1]
    omega_r_shamir = np.clip(omega_r_shamir, -max_wheel_speed, max_wheel_speed)
    omega_l_shamir = np.clip(omega_l_shamir, -max_wheel_speed, max_wheel_speed)
    left_wheel_speeds_shamir.append(omega_l_shamir)
    right_wheel_speeds_shamir.append(omega_r_shamir)
    velocities_shamir = conversion_matrix_robot @ np.array([[omega_r_shamir], [omega_l_shamir]])
    linear_velocity_shamir = velocities_shamir[0]
    angular_velocity_shamir = velocities_shamir[1]
    linear_velocities_shamir.append(linear_velocity_shamir)
    angular_velocities_shamir.append(angular_velocity_shamir)
    current_x_shamir += sampling_time * linear_velocity_shamir * np.cos(current_theta_shamir)
    current_y_shamir += sampling_time * linear_velocity_shamir * np.sin(current_theta_shamir)
    current_theta_shamir += sampling_time * angular_velocity_shamir
    current_theta_shamir = (current_theta_shamir + np.pi) % (2 * np.pi) - np.pi
    current_x_shamir = current_x_shamir[0]
    current_y_shamir = current_y_shamir[0]
    current_theta_shamir = current_theta_shamir[0]
##
##
# Convert lists to numpy arrays
##
##
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
control_input_x_rscode = np.array(control_input_x_rscode)
control_input_y_rscode = np.array(control_input_y_rscode)
control_input_x_shamir = np.array(control_input_x_shamir)
control_input_y_shamir = np.array(control_input_y_shamir)
# ---------------------------------------------------------------------------------------------------- Plotting --------------------------------------------------------------------------------
##
##
## -------------------- Plotting the robot trajectories vs. reference trajectory --------------------
##
##
plt.figure(figsize=(12, 6))
# Subplot for RS Code trajectory
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(reference_x, reference_y, 'r--', label='Reference Trajectory', linewidth=6.5)
plt.plot(robot_trajectory_x, robot_trajectory_y, 'b-', label='Robot Trajectory (Reed-Solomon)', linewidth=2.5)
plt.plot(0, 0, 'k*', markersize=20, label='Start Point')  # Starting point
plt.plot(robot_trajectory_x[-1], robot_trajectory_y[-1], 'go', markersize=10, label='Stop Point')  # Ending point
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.legend(loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.title('RS Code Trajectory Tracking Comparison')
# Subplot for Shamir trajectory
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(reference_x, reference_y, 'r--', label='Reference Trajectory', linewidth=6.5)
plt.plot(robot_trajectory_x_shamir, robot_trajectory_y_shamir, 'b-', label='Robot Trajectory (Shamir)', linewidth=2.5)
plt.plot(0, 0, 'k*', markersize=20, label='Start Point')  # Starting point
plt.plot(robot_trajectory_x_shamir[-1], robot_trajectory_y_shamir[-1], 'go', markersize=10, label='Stop Point')  # Ending point
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.legend(loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.title('Shamir Trajectory Tracking Comparison')
plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('Figures/trajectory_tracking_comparison_subplots.eps', format='eps', bbox_inches='tight')
plt.show()
##
##
## -------------------- Plot Tracking Error Signal --------------------
##
##
plt.figure(figsize=(12, 8))
line1 = plt.plot(time_vector, position_error_x, 'b-', label='X Error (Reed-Solomon)')
line2 = plt.plot(time_vector, position_error_y, 'b--', label='Y Error (Reed-Solomon)')
line3 = plt.plot(time_vector, position_error_x_shamir, 'g-', label='X Error (Shamir)')
line4 = plt.plot(time_vector, position_error_y_shamir, 'g--', label='Y Error (Shamir)')
plt.xlabel('Time [s]')
plt.ylabel('Position Error [m]')
plt.grid(True)
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
plt.legend(lines, labels, loc='upper right')
plt.title('Tracking Errors')
plt.savefig('Figures/tracking_errors.eps', format='eps', bbox_inches='tight')
plt.show()
##
##
## -------------------- Plot Timeline of the Events and Detection --------------------
##
##
# TABLE_DISPLAY_STEPS = 450
# display_steps = min(TABLE_DISPLAY_STEPS, num_time_steps)  
# cell_aspect_ratio = NUM_SHARES / display_steps
# if cell_aspect_ratio > 1:
#     fig_width = 10
#     fig_height = 10 / cell_aspect_ratio
# else:
#     fig_height = 10
#     fig_width = 10 * cell_aspect_ratio
# fig_width = max(fig_width, 6)
# fig_height = max(fig_height, 6)
# plt.figure(figsize=(fig_width, fig_height))
# cell_colors = np.full((NUM_SHARES, display_steps), 'green')
# for t in range(display_steps):
#     if rscode_error_index[t] >= 0 and rscode_error_index[t] < NUM_SHARES:
#         cell_colors[int(rscode_error_index[t]), t] = 'red'
#     if rscode_nan_index[t] >= 0 and rscode_nan_index[t] < NUM_SHARES:
#         cell_colors[int(rscode_nan_index[t]), t] = 'black'
# table = plt.table(
#     cellColours=cell_colors,
#     cellLoc='center',
#     loc='center',
#     rowLabels=[f'Share {i}' for i in range(NUM_SHARES)],
#     colLabels=[f'{i}' for i in range(display_steps)],
# )
# table.auto_set_font_size(False)
# table.set_fontsize(9)
# table.scale(1, 1)
# plt.axis('off')
# plt.title(f'Share Status for First {display_steps} Time Steps (Green: OK, Red: Error, Black: NaN)')
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='green', label='OK'),
#     Patch(facecolor='red', label='Error'),
#     Patch(facecolor='black', label='NaN')
# ]
# plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05))
# plt.tight_layout()
# plt.savefig('share_status_table.eps', format='eps', dpi=1200)
# plt.show()
##
##
# -------------------- Plotting Velocities --------------------
##
##
fig, axs = plt.subplots(2, 1, figsize=(12, 12))  # Create 2 subplots in a single column
# Plot linear velocities
axs[0].plot(time_vector, linear_velocities_rscode, 'b-', label='Linear Velocity (Reed-Solomon)')
axs[0].plot(time_vector, linear_velocities_shamir, 'g-', label='Linear Velocity (Shamir)')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Linear Velocity [m/s]')
axs[0].set_title('Linear Velocity Comparison')
axs[0].legend(loc='upper right')
axs[0].grid(True)
# Plot angular velocities
axs[1].plot(time_vector, angular_velocities_rscode, 'b--', label='Angular Velocity (Reed-Solomon)')
axs[1].plot(time_vector, angular_velocities_shamir, 'g--', label='Angular Velocity (Shamir)')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Angular Velocity [rad/s]')
axs[1].set_title('Angular Velocity Comparison')
axs[1].legend(loc='upper right')
axs[1].grid(True)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('Figures/velocity_comparison.eps', format='eps', bbox_inches='tight')
plt.show()
##
##
# -------------------- Plotting Wheel Speeds --------------------
##
##
fig, axs = plt.subplots(2, 1, figsize=(12, 12))  # Create 2 subplots in a single column
# Plot right wheel speeds
axs[0].plot(time_vector, right_wheel_speeds_rscode, 'b-', label='Right Wheel Speed (Reed-Solomon)')
axs[0].plot(time_vector, right_wheel_speeds_shamir, 'g-', label='Right Wheel Speed (Shamir)')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Right Wheel Speed [rad/s]')
axs[0].set_title('Right Wheel Speed Comparison')
axs[0].legend(loc='upper right')
axs[0].grid(True)
# Plot left wheel speeds
axs[1].plot(time_vector, left_wheel_speeds_rscode, 'b-', label='Left Wheel Speed (Reed-Solomon)')
axs[1].plot(time_vector, left_wheel_speeds_shamir, 'g-', label='Left Wheel Speed (Shamir)')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Left Wheel Speed [rad/s]')
axs[1].set_title('Left Wheel Speed Comparison')
axs[1].legend(loc='upper right')
axs[1].grid(True)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('Figures/wheel_speed_comparison.eps', format='eps', bbox_inches='tight')
plt.show()
##
##
## -------------------- Store Data --------------------
##
##
right_wheel_speeds_rscode = np.array(right_wheel_speeds_rscode)
left_wheel_speeds_rscode = np.array(left_wheel_speeds_rscode)
right_wheel_speeds_shamir = np.array(right_wheel_speeds_shamir)
left_wheel_speeds_shamir = np.array(left_wheel_speeds_shamir)
data_to_save = {
    'robot_trajectory_x_rscode': robot_trajectory_x,
    'robot_trajectory_y_rscode': robot_trajectory_y,
    'robot_trajectory_theta_rscode': robot_trajectory_theta,
    'robot_trajectory_x_shamir': robot_trajectory_x_shamir,
    'robot_trajectory_y_shamir': robot_trajectory_y_shamir,
    'robot_trajectory_theta_shamir': robot_trajectory_theta_shamir,
    'position_error_x': position_error_x,
    'position_error_y': position_error_y,
    'absolute_position_error': absolute_position_error,
    'position_error_x_shamir': position_error_x_shamir,
    'position_error_y_shamir': position_error_y_shamir,
    'absolute_position_error_shamir': absolute_position_error_shamir,
    'control_input_x_rscode': control_input_x_rscode,
    'control_input_y_rscode': control_input_y_rscode,
    'control_input_x_shamir': control_input_x_shamir,
    'control_input_y_shamir': control_input_y_shamir,
    'linear_velocities_rscode': linear_velocities_rscode,
    'angular_velocities_rscode': angular_velocities_rscode,
    'linear_velocities_shamir': linear_velocities_shamir,
    'angular_velocities_shamir': angular_velocities_shamir,
    'right_wheel_speeds_rscode': right_wheel_speeds_rscode,
    'left_wheel_speeds_rscode': left_wheel_speeds_rscode,
    'right_wheel_speeds_shamir': right_wheel_speeds_shamir,
    'left_wheel_speeds_shamir': left_wheel_speeds_shamir,
    'reference_x': reference_x, 
    'reference_y': reference_y,  
    'reference_theta': reference_theta 
}
scipy.io.savemat('Figures/robot_simulation_data.mat', data_to_save)