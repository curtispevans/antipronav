import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.ekf_unknownA_discrete import kalman_update
from models.mav_dynamics import MavDynamics

bearings = np.load('data/bearing.npy')
pixel_sizes = np.load('data/pixel_sizes.npy')
true_distance = np.load('data/distances.npy')
control = np.load('data/control.npy')
relative_velocities = np.load('data/relative_velocity.npy')
own_velocities = np.load('data/own_velocity.npy')

Ts = 1/30
intruder_vel = np.array([0., 30.])
intruder_heading = np.pi/2

los_n = jnp.cos(bearings[0]-np.radians(0.1))
los_e = jnp.sin(bearings[0]+np.radians(0.1))
pixel_size = pixel_sizes[0]
c_n = 30*np.cos(intruder_heading)
c_e = 30.*np.sin(intruder_heading)
eta = 1./true_distance[0]
A = 20

mu20 = jnp.array([los_n, los_e, pixel_size, c_n, c_e, eta, 20])
mu15 = jnp.array([los_n, los_e, pixel_size, c_n, c_e, eta, 15])
mu10 = jnp.array([los_n, los_e, pixel_size, c_n, c_e, eta, 10])
sigma20 = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01)), jnp.cos(jnp.radians(0.01)), 1, 0.1, 0.1, 0.01, 5]))
sigma15 = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01)), jnp.cos(jnp.radians(0.01)), 1, 0.1, 0.1, 0.01, 5]))
sigma10 = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01)), jnp.cos(jnp.radians(0.01)), 1, 0.1, 0.1, 0.01, 5]))

Q = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01)), jnp.cos(jnp.radians(0.01)), 0.01, 0.01, 0.01, 0.01, 10]))
R = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01)), jnp.cos(jnp.radians(0.01)), 0.01, 0.0001]))

est_dist20 = []
est_bearing20 = []
est_pixel_size20 = []
est_A20 = []
std_A20 = []

est_dist15 = []
est_bearing15 = []
est_pixel_size15 = []
est_A15 = []
std_A15 = []

est_dist10 = []
est_bearing10 = []
est_pixel_size10 = []
est_A10 = []
std_A10 = []

for bearing, pixel_size, own_vel in zip(bearings, pixel_sizes, own_velocities):
    measurement = jnp.array([jnp.cos(bearing), jnp.sin(bearing), pixel_size, 0.0])

    mu20, sigma20 = kalman_update(mu20, sigma20, own_vel, measurement, Q, R, Ts)
    est_dist20.append(mu20[5])
    est_bearing20.append(np.arctan2(mu20[1], mu20[0]))
    est_pixel_size20.append(mu20[2])
    est_A20.append(mu20[6])
    std_A20.append(sigma20[6, 6])

    mu15, sigma15 = kalman_update(mu15, sigma15, own_vel, measurement, Q, R, Ts)
    est_dist15.append(mu15[5])
    est_bearing15.append(np.arctan2(mu15[1], mu15[0]))
    est_pixel_size15.append(mu15[2])
    est_A15.append(mu15[6])
    std_A15.append(sigma15[6, 6])

    mu10, sigma10 = kalman_update(mu10, sigma10, own_vel, measurement, Q, R, Ts)
    est_dist10.append(mu10[5])
    est_bearing10.append(np.arctan2(mu10[1], mu10[0]))
    est_pixel_size10.append(mu10[2])
    est_A10.append(mu10[6])
    std_A10.append(sigma10[6, 6])





plt.subplot(221)
plt.plot(bearings, label='True Bearing')
plt.plot(est_bearing20, label='Est A=20')
plt.plot(est_bearing15, label='Est A=15')
plt.plot(est_bearing10, label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Bearing')
plt.legend()
plt.title('Bearing between Mavs')

plt.subplot(222)
plt.plot(pixel_sizes, label='True Pixel Size')
plt.plot(est_pixel_size20, label='Est A=20')
plt.plot(est_pixel_size15, label='Est A=15')
plt.plot(est_pixel_size10, label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Pixel Size')
plt.title('Pixel Size')
plt.legend()


plt.subplot(223)
plt.plot(1/true_distance, label='True inverse distance')
plt.plot(est_dist20, label='Est A=20')
plt.plot(est_dist15, label='Est A=15')
plt.plot(est_dist10, label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Inverse Distance')
plt.title('Inverse Distance between Mavs')
plt.legend()


plt.subplot(224)
plt.plot(15*np.ones(len(est_A20)), label='True A')
plt.plot(est_A20, label='Est A=20')
plt.plot(est_A15, label='Est A=15')
plt.plot(est_A10, label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Wingspan')
plt.title('A')
plt.legend()

# plt.subplot(224)
# plt.plot(np.ones(len(est_relative_velocity_x))*intruder_vel[0], label='True Relative Velocity N')
# plt.plot(np.ones(len(est_relative_velocity_y))*intruder_vel[1], label='True Relative Velocity E')
# plt.plot(est_relative_velocity_x, label='Estimated Intruder Vel N')
# plt.plot(est_relative_velocity_y, label='Estimated Intruder Vel E')
# plt.xlabel('Time')
# plt.ylabel('Relative Velocity')
# plt.title('Relative Velocity')
# plt.legend()

plt.tight_layout()
plt.show()

plt.plot(std_A20, label='Est A=20')
plt.plot(std_A15, label='Est A=15')
plt.plot(std_A10, label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Standard Deviation A')
plt.title('Standard Deviation A')
plt.legend()
plt.show()




