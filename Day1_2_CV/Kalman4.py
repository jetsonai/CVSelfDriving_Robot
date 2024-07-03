import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

np.random.seed(0)

def get_pos_vel(itime):
    """Return Measured Velocity and True Position."""
    v = np.random.normal(0, np.sqrt(10))  # v: measurement noise.

    vel_true = 80                         # nominal velocity = 80 [m/s]. no system noise here.
    pos_true = vel_true * (itime * dt)    # pos_true: true position. 
    z_vel_meas = vel_true + v             # z_vel_meas: measured velocity (observable) 

    return z_vel_meas, pos_true

def kalman_filter(z_meas, x_esti, P):
    """Kalman Filter Algorithm."""
    # (1) Prediction.
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q

    # (2) Kalman Gain.
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    # (3) Estimation.
    x_esti = x_pred + K @ (z_meas - H @ x_pred)

    # (4) Error Covariance.
    P = P_pred - K @ H @ P_pred

    return x_esti, P

# Input parameters.
time_end = 4
dt = 0.1

# Initialization for system model.
# Matrix: A, H, Q, R, P_0
# Vector: x_0
A = np.array([[1, dt],
              [0, 1]])
H = np.array([[0, 1]])
Q = np.array([[1, 0],
              [0, 3]])
R = np.array([[10]])

# Initialization for estimation.
x_0 = np.array([0, 20])  # position and velocity
P_0 = 5 * np.eye(2)

time = np.arange(0, time_end, dt)
n_samples = len(time)
vel_meas_save = np.zeros(n_samples)
pos_true_save = np.zeros(n_samples)
pos_esti_save = np.zeros(n_samples)
vel_esti_save = np.zeros(n_samples)

x_esti, P = None, None
for i in range(n_samples):
    z_meas, pos_true = get_pos_vel(i)
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = kalman_filter(z_meas, x_esti, P)

    vel_meas_save[i] = z_meas
    pos_true_save[i] = pos_true
    pos_esti_save[i] = x_esti[0]
    vel_esti_save[i] = x_esti[1]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(time, vel_esti_save, 'bo-', label='Estimation (KF)')
plt.plot(time, vel_meas_save, 'r*--', label='Measurements', markersize=10)
plt.legend(loc='lower right')
plt.title('Velocity: Meas. v.s. Esti. (KF)')
plt.xlabel('Time [sec]')
plt.ylabel('Velocity [m/s]')

plt.subplot(1, 2, 2)
plt.plot(time, pos_esti_save, 'bo-', label='Estimation (KF)')
plt.plot(time, pos_true_save, 'g*--', label='True', markersize=10)
plt.legend(loc='upper left')
plt.title('Position: True v.s. Esti. (KF)')
plt.xlabel('Time [sec]')
plt.ylabel('Position [m]')
plt.show()