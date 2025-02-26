import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.kalman_filter import kalman_update
from models.mav_dynamics import MavDynamics

bearings = np.load('data/bearing.npy')
pixel_sizes = np.load('data/pixel_sizes.npy')
true_distance = np.load('data/distances.npy')
control = np.load('data/control.npy')

Ts = 1/30

mu = jnp.array([jnp.cos(bearings[0]), jnp.sin(bearings[0]), pixel_sizes[0], 10, 10., 1./true_distance[0]])
sigma = jnp.eye(6) * 0.1

R = jnp.eye(6) * 0.1
Q = jnp.eye(3) * 0.1

est_dist = []
est_bearing = []
est_pixel_size = []

for bearing, pixel_size, u in zip(bearings, pixel_sizes, control):
    measurement = jnp.array([jnp.cos(bearing), jnp.sin(bearing), pixel_size])
    mu, sigma = kalman_update(mu, sigma, u, measurement, R, Q, Ts)
    est_dist.append(1/mu[5])
    est_bearing.append(np.arctan2(mu[1], mu[0]))
    est_pixel_size.append(mu[2])



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
plt.plot(true_distance, label='True Distance')
plt.plot(est_dist, label='Estimated Distance')
plt.xlabel('Time')
plt.ylabel('Distance')
plt.title('Distance between Mavs')
plt.legend()

plt.tight_layout()
plt.show()


