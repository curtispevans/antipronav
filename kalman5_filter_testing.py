import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.kalman_filter_5_states import kalman_update
from models.mav_dynamics import MavDynamics

bearings = np.load('data/bearing.npy')
pixel_sizes = np.load('data/pixel_sizes.npy')
true_distance = np.load('data/distances.npy')
control = np.load('data/control.npy')
relative_velocities = np.load('data/relative_velocity.npy')
own_velocities = np.load('data/own_velocity.npy')

Ts = 1/30
intruder_vel = np.array([0., 30.])

mu = jnp.array([bearings[0], 0.001, 0., 30., 1./true_distance[0]])
sigma = jnp.diag(jnp.array([jnp.radians(0.1), 1, .0000001, 0.000001, 0.01]))

R = jnp.diag(jnp.array([jnp.radians(0.1), 1, 0.000001, 0.0000001, 1e-10]))
Q = jnp.diag(jnp.array([jnp.radians(0.1), 1]))

est_dist = []
est_bearing = []
est_pixel_size = []
est_relative_velocity_x = []
est_relative_velocity_y = []

for bearing, pixel_size, own_vel in zip(bearings, pixel_sizes, own_velocities):
    measurement = jnp.array([bearing, pixel_size])
    mu, sigma = kalman_update(mu, sigma, own_vel, measurement, R, Q, Ts)
    est_dist.append(mu[4])
    est_bearing.append(mu[0])
    est_pixel_size.append(mu[1])
    est_relative_velocity_x.append(mu[2])
    est_relative_velocity_y.append(mu[3])



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
plt.plot(true_distance, label='True Distance')
plt.plot(1/np.array(est_dist), label='Estimated Distance')
plt.xlabel('Time')
plt.ylabel('Distance')
plt.title('Distance between Mavs')
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


