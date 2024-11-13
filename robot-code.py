import numpy as np
import matplotlib.pyplot as plt

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
controller_gain_matrix = np.array([
    [1, sampling_time, -sampling_time],
    [integral_gain, proportional_gain, -proportional_gain]
])

# Unicycle model matrices
transformation_matrix = np.array([
    [wheel_radius / 2, wheel_radius / 2],
    [wheel_radius / wheel_base, -wheel_radius / wheel_base]
])
inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

# Read reference trajectory from files
reference_x = np.loadtxt('./Reference Trajectories/xr.txt', delimiter=",")
reference_y = np.loadtxt('./Reference Trajectories/yr.txt', delimiter=",")
reference_x_velocity = np.loadtxt('./Reference Trajectories/xdr.txt', delimiter=",")
reference_y_velocity = np.loadtxt('./Reference Trajectories/ydr.txt', delimiter=",")
reference_theta = np.loadtxt('./Reference Trajectories/thetar.txt', delimiter=",")

# Robot initial conditions
current_x = 0.0
current_y = 0.0
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

# Controller states (integral components)
integral_state_x = np.zeros(num_time_steps + 1)
integral_state_y = np.zeros(num_time_steps + 1)

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

    # Controller input vectors for x and y directions
    controller_input_vector_x = np.array([
        integral_state_x[k],
        ref_x + lookahead_distance * np.cos(ref_theta),
        current_x + lookahead_distance * np.cos(current_theta)
    ])
    controller_input_vector_y = np.array([
        integral_state_y[k],
        ref_y + lookahead_distance * np.sin(ref_theta),
        current_y + lookahead_distance * np.sin(current_theta)
    ])

    # Controller output vectors for x and y directions
    controller_output_vector_x = controller_gain_matrix @ controller_input_vector_x
    controller_output_vector_y = controller_gain_matrix @ controller_input_vector_y

    # Update integral states
    integral_state_x[k + 1] = controller_output_vector_x[0]
    integral_state_y[k + 1] = controller_output_vector_y[0]

    # Control inputs
    u_x = controller_output_vector_x[1]
    u_y = controller_output_vector_y[1]
    control_input_x.append(u_x)
    control_input_y.append(u_y)

    # Compute linear and angular velocities
    transformation_inverse = np.array([
        [np.cos(current_theta), np.sin(current_theta)],
        [-np.sin(current_theta) / lookahead_distance, np.cos(current_theta) / lookahead_distance]
    ])
    control_vector = np.array([u_x, u_y])
    velocities = transformation_inverse @ control_vector
    linear_velocity = velocities[0]
    angular_velocity = velocities[1]

    # Saturate velocities to robot's physical limits
    max_linear_velocity = wheel_radius * max_wheel_speed
    max_angular_velocity = 2 * max_wheel_speed * wheel_radius / wheel_base
    linear_velocity = np.clip(linear_velocity, -max_linear_velocity, max_linear_velocity)
    angular_velocity = np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity)

    # Update robot's state
    current_x += sampling_time * linear_velocity * np.cos(current_theta)
    current_y += sampling_time * linear_velocity * np.sin(current_theta)
    current_theta += sampling_time * angular_velocity

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