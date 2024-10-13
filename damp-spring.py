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
rscode = RSSecretSharing(PRIME, 1, NUM_SHARES, T_POLY_DEGREE, MAX_MISSING, MAX_MANIPULATED)

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
K = np.array([k_p, k_v])
K_qu = np.array([custom_quantize(k_p), custom_quantize(k_v)])
# Simulation parameters
num_steps = 100
x = np.zeros((2, num_steps))
x_ = np.zeros((2, num_steps))
x_re = np.zeros((2, num_steps))
x_qu = np.zeros((2, num_steps))
u = np.zeros(num_steps)
u_qu = np.zeros(num_steps)
u_de = np.zeros(num_steps)
u_re = np.zeros(num_steps)
u_de_re = np.zeros(num_steps)
x_shares = np.zeros((2, NUM_SHARES))
u_shares = np.zeros(NUM_SHARES)
# Initial conditions
x[:, 0] = [2.0, 0.0]  # initial position and velocity
x_[:, 0] = x[:, 0]  # initial position and velocity
x_re[:, 0] = x[:, 0]
x_qu[:, 0] = [custom_quantize(x_[0,0]), custom_quantize(x_[1,0])]
x_shares = rscode.shares_of_vector(x_qu[:, 0])
# Simulation loop
for k in range(num_steps - 1):
    for share in range(x_shares.shape[1]): 
        u_shares[share] = (-K_qu @ x_shares[:, share]) % PRIME
    u_shares = [None if isinstance(x, (float, np.floating)) and math.isnan(x) else x for x in u_shares]
    u_re[k], error_indices = rscode.shamir_robust_reconstruct(u_shares)
    u_de_re[k] = decode_quantized_value(decode_quantized_value(u_re[k]))
    u[k] = -K @ x[:, k]
    u_qu[k] = (-K_qu @ x_qu[:, k]) % PRIME
    u_de[k] = decode_quantized_value(decode_quantized_value(u_qu[k]))
    x[:, k + 1] = SYS_A @ x[:, k] + SYS_B.flatten() * u[k]
    x_[:, k + 1] = SYS_A @ x_[:, k] + SYS_B.flatten() * u_de[k]
    x_re[:, k + 1] = SYS_A @ x_re[:, k] + SYS_B.flatten() * u_de_re[k]
    x_qu[:, k + 1] = [custom_quantize(x_re[0, k + 1]), custom_quantize(x_re[1, k + 1])]
    x_shares = rscode.shares_of_vector(x_qu[:, k + 1])

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
plt.plot(time, x_[0, :], label='Position')
plt.ylabel('Position')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, x_[1, :], label='Velocity')
plt.ylabel('Velocity')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, u_de, label='Control Input')
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

plt.show()