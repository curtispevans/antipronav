import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.kalman_filter import kalman_update
from models.mav_dynamics import MavDynamics

bearings = np.load('data/bearing.npy')
pixel_sizes = np.load('data/pixel_sizes.npy')
true_distance = np.load('data/distances.npy')
control = np.load('data/control.npy')
relative_velocities = np.load('data/relative_velocity.npy')

Ts = 1/30

mu = jnp.array([jnp.cos(bearings[0]), jnp.sin(bearings[0]), pixel_sizes[0], relative_velocities[0,0], relative_velocities[0,1], 1./true_distance[0]])
sigma = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01)), jnp.cos(jnp.radians(0.01)), 0.01, 0.01, 0.01, 0.001]))

R = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.001)), jnp.sin(jnp.radians(0.001)), 1, 0.1, 0.1, 0.1]))
Q = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.001)), jnp.sin(jnp.radians(0.001)), 0.1]))

est_dist = []
est_bearing = []
est_pixel_size = []
est_relative_velocity_x = []
est_relative_velocity_y = []
inv_dist_std = []

for bearing, pixel_size, u in zip(bearings[1:], pixel_sizes[1:], control[1:]):
    measurement = jnp.array([jnp.cos(bearing), jnp.sin(bearing), pixel_size])
    mu, sigma = kalman_update(mu, sigma, u, measurement, R, Q, Ts)
    est_dist.append(mu[5])
    inv_dist_std.append(jnp.sqrt(sigma[-1,-1]))
    est_bearing.append(np.arctan2(mu[1], mu[0]))
    est_pixel_size.append(mu[2])
    est_relative_velocity_x.append(mu[3])
    est_relative_velocity_y.append(mu[4])



plt.subplot(221)
plt.plot(bearings, label='True Bearing')
plt.plot(est_bearing, label='Estimated Bearing')
plt.xlabel('Time')
plt.ylabel('Bearing')
plt.legend()
plt.grid()
plt.title('Bearing between Mavs')

plt.subplot(222)
plt.plot(pixel_sizes, label='True Pixel Size')
plt.plot(est_pixel_size, label='Estimated Pixel Size')
plt.xlabel('Time')
plt.ylabel('Pixel Size')
plt.title('Pixel Size')
plt.grid()
plt.legend()


plt.subplot(223)
plt.plot(1/true_distance, label='True Distance')
plt.plot(est_dist, label='Estimated Distance')
# plt.plot(np.array(est_dist) + np.array(inv_dist_std), '--', color='black', label='Estimated Distance + Std')
# plt.plot(np.array(est_dist) - np.array(inv_dist_std), '--', color='black', label='Estimated Distance - Std')
plt.xlabel('Time')
plt.ylabel('1/Distance')
plt.title('1/Distance between Mavs')
plt.grid()
plt.legend()

plt.subplot(224)
plt.plot(relative_velocities[:,0], label='True Relative Velocity X')
plt.plot(relative_velocities[:,1], label='True Relative Velocity Y')
plt.plot(est_relative_velocity_x, label='Estimated Relative Velocity X')
plt.plot(est_relative_velocity_y, label='Estimated Relative Velocity Y')
plt.xlabel('Time')
plt.ylabel('Relative Velocity')
plt.title('Relative Velocity')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


