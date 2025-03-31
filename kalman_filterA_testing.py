import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.kalman_filter_A import kalman_update
from models.mav_dynamics import MavDynamics

bearings = np.load('data/bearing.npy')
pixel_sizes = np.load('data/pixel_sizes.npy')
true_distance = np.load('data/distances.npy')
control = np.load('data/control.npy')
relative_velocities = np.load('data/relative_velocity.npy')
own_velocities = np.load('data/own_velocity.npy')

Ts = 1/30
intruder_vel = np.array([0., 30.])

mu = jnp.array([jnp.cos(bearings[0]-np.radians(0.1)), jnp.sin(bearings[0]+np.radians(0.1)), 0.001, 0., 0., 1./true_distance[0], 20])
sigma = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01)), jnp.cos(jnp.radians(0.01)), 1, .1, 0.1, 0.01, 10]))

Q = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01)), jnp.cos(jnp.radians(0.01)), 1, 0.01, 0.01, 0.1, 0.1]))
R = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01)), jnp.cos(jnp.radians(0.01)), 1]))
R_psuedo = jnp.diag(jnp.array([0.000001]))

est_dist = []
est_bearing = []
est_pixel_size = []
est_relative_velocity_x = []
est_relative_velocity_y = []
est_A = []


for bearing, pixel_size, own_vel in zip(bearings, pixel_sizes, own_velocities):
    measurement = jnp.array([jnp.cos(bearing), jnp.sin(bearing), pixel_size])
    measurement_psuedo = jnp.array([0])
    mu, sigma = kalman_update(mu, sigma, own_vel, measurement, Q, R, R_psuedo, Ts)
    est_dist.append(mu[5])
    est_bearing.append(np.arctan2(mu[1], mu[0]))
    est_pixel_size.append(mu[2])
    est_relative_velocity_x.append(mu[3])
    est_relative_velocity_y.append(mu[4])
    est_A.append(mu[6])
    



plt.subplot(221)
plt.plot(bearings, label='True Bearing')
plt.plot(est_bearing, label='Estimated Bearing')
plt.xlabel('Time')
plt.ylabel('Bearing')
plt.legend()
plt.title('Bearing between Mavs')

plt.subplot(222)
plt.plot(pixel_sizes, label='True Pixel Size')
plt.plot(est_pixel_size, label='Estimated Pixel Size')
plt.xlabel('Time')
plt.ylabel('Pixel Size')
plt.title('Pixel Size')
plt.legend()


plt.subplot(223)
plt.plot(1/true_distance, label='True inverse distance')
plt.plot(est_dist, label='Estimated inverse distance')
plt.xlabel('Time')
plt.ylabel('Inverse Distance')
plt.title('Inverse Distance between Mavs')
plt.legend()

plt.subplot(224)
plt.plot(est_A, label='Estimated A')
plt.plot(np.ones(len(est_A))*20, label='True A')
plt.xlabel('Time')
plt.ylabel('A')
plt.title('A')
plt.legend()


# plt.subplot(224)
# plt.plot(true_distance, label='True Distance')
# plt.plot(1/np.array(est_dist), label='Estimated Distance')
# plt.xlabel('Time')
# plt.ylabel('Distance')
# plt.title('Distance between Mavs')
# plt.legend()


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


