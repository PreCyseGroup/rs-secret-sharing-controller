import numpy as np
import matplotlib.pyplot as plt
from RSSecretSharing import RSSecretSharing
import math
# Global Quantization Parameters
QU_BASE = 10
QU_GAMMA = 2
QU_DELTA = 1
#Global Parameters / Sharing 
PRIME = 2053
NUM_SHARES = 5
T_POLY_DEGREE = 1 
MAX_MISSING = 1
MAX_MANIPULATED = 1
SEED = 11
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

# System parameters
P_m = 1.0  # mass
P_k = 1.0  # spring constant
P_b = 0.2  # damping coefficient
T_s = 0.1  # sampling period
k_p = 2.0  # proportional gain for position
k_v = 1.0  # proportional gain for velocity
# Discrete-time state-space matrices
SYS_A = np.array([[1, T_s], [-T_s * P_k / P_m, 1 - T_s * P_b / P_m]])
SYS_B = np.array([[0], [T_s / P_m]])
K_P = np.array([k_p, k_v])
K_I = np.array([0.1, 0.5])
K_P_qu = np.array([custom_quantize(k_p), custom_quantize(k_v)])
K_I_qu = np.array([custom_quantize(K_I[0]), custom_quantize(K_I[1])])
# Simulation parameters
num_steps = 100
x = np.zeros((2, num_steps))
x_re = np.zeros((2, num_steps))
x_qu = np.zeros((2, num_steps))
x_shares = np.zeros((2, NUM_SHARES))
u = np.zeros(num_steps)
u_re = np.zeros(num_steps)
u_de_re = np.zeros(num_steps)
u_shares = np.zeros(NUM_SHARES)
x_PI = x = np.zeros((2, num_steps))
x_PI_re = np.zeros((2, num_steps))
x_PI_qu = np.zeros((2, num_steps))
x_PI_shares = np.zeros((2, NUM_SHARES))
u_PI = np.zeros(num_steps)
u_PI_re = np.zeros(num_steps)
u_PI_de_re = np.zeros(num_steps)
u_PI_shares = np.zeros(NUM_SHARES)
# Initial conditions
x[:, 0] = [2.0, 0.0]  # initial position and velocity
x_re[:, 0] = x[:, 0]
x_qu[:, 0] = [custom_quantize(x[0,0]), custom_quantize(x[1,0])]
x_shares = rscode.shares_of_vector(x_qu[:, 0])
x_PI[:, 0] = x[:, 0]
x_PI_re[:, 0] = x[:, 0]
x_PI_qu[:, 0] = [custom_quantize(x_PI[0,0]), custom_quantize(x_PI[1,0])]
x_PI_shares = rscode.shares_of_vector(x_PI_qu[:, 0])
PI_integral_term = np.zeros(2)
PI_integral_term_re = np.zeros(2)
PI_integral_term_qu = [custom_quantize(PI_integral_term[0]), custom_quantize(PI_integral_term[1])]
PI_integral_term_shares = rscode.shares_of_vector(PI_integral_term_qu)
# Simulation loop
for k in range(num_steps - 1):
    #### P Controller 
    u[k] = -K_P @ x[:, k]
    x[:, k + 1] = SYS_A @ x[:, k] + SYS_B.flatten() * u[k]
    #### P Controller RSCode 
    for share in range(x_shares.shape[1]): 
        u_shares[share] = (-K_P_qu @ x_shares[:, share]) % PRIME
    u_shares = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in u_shares]
    u_re[k], error_indices = rscode.shamir_robust_reconstruct(u_shares)
    u_de_re[k] = decode_quantized_value(decode_quantized_value(u_re[k]))
    x_re[:, k + 1] = SYS_A @ x_re[:, k] + SYS_B.flatten() * u_de_re[k]
    x_qu[:, k + 1] = [custom_quantize(x_re[0, k + 1]), custom_quantize(x_re[1, k + 1])]
    x_shares = rscode.shares_of_vector(x_qu[:, k + 1])
    #### PI Controller
    PI_integral_term += x_PI[:, k] * T_s
    u_PI[k] = -K_P @ x_PI[:, k] - K_I @ PI_integral_term
    x_PI[:, k+1] = SYS_A @ x_PI[:, k] + SYS_B.flatten() * u_PI[k]
    #### PI Controller RSCode 
    PI_integral_term_re += x_PI_re[:, k] * T_s
    PI_integral_term_qu = [custom_quantize(PI_integral_term_re[0]), custom_quantize(PI_integral_term_re[1])]
    PI_integral_term_shares = rscode.shares_of_vector(PI_integral_term_qu)
    for share in range(x_PI_shares.shape[1]): 
        u_PI_shares[share] = (-K_P_qu @ x_PI_shares[:, share] - K_I_qu @ PI_integral_term_shares[:, share]) % PRIME
    u_PI_shares = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in u_PI_shares]
    print("PI Controller Control Input: \n", u_PI_shares)
    u_PI_re[k], error_indices = rscode.shamir_robust_reconstruct(u_PI_shares)
    u_PI_de_re[k] = decode_quantized_value(decode_quantized_value(u_PI_re[k]))
    x_PI_re[:, k + 1] = SYS_A @ x_PI_re[:, k] + SYS_B.flatten() * u_PI_de_re[k]
    x_PI_qu[:, k + 1] = [custom_quantize(x_PI_re[0, k + 1]), custom_quantize(x_PI_re[1, k + 1])]
    x_PI_shares = rscode.shares_of_vector(x_PI_qu[:, k + 1])
    print("PI Controller State of System: \n", x_PI_shares)

# Plotting the results
time = np.arange(num_steps) * T_s
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, x[0, :], label='Position')
plt.ylabel('Position')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, x[1, :], label='Velocity')
plt.ylabel('Velocity')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, u, label='Control Input')
plt.ylabel('Control Input')
plt.xlabel('Time [s]')
plt.legend()

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, x_re[0, :], label='Position')
plt.ylabel('Position')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, x_re[1, :], label='Velocity')
plt.ylabel('Velocity')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, u_de_re, label='Control Input')
plt.ylabel('Control Input')
plt.xlabel('Time [s]')
plt.legend()

#### PI Controller Plots
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, x_PI[0, :], label='Position')
plt.ylabel('Position')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, x_PI[1, :], label='Velocity')
plt.ylabel('Velocity')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, u_PI, label='Control Input')
plt.ylabel('Control Input')
plt.xlabel('Time [s]')
plt.legend()

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, x_PI_re[0, :], label='Position')
plt.ylabel('Position')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, x_PI_re[1, :], label='Velocity')
plt.ylabel('Velocity')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, u_PI_de_re, label='Control Input')
plt.ylabel('Control Input')
plt.xlabel('Time [s]')
plt.legend()

plt.show()