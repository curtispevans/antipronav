import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.ekf_knownA_continuous import kalman_update
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
A = 15

mu = jnp.array([jnp.cos(bearings[0]), jnp.sin(bearings[0]), pixel_sizes[0], 30*np.cos(intruder_heading), 30.*np.sin(intruder_heading), 1./true_distance[0]])
sigma = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01)), jnp.sin(jnp.radians(0.01)), 1, 0.1, 0.1, 0.01]))**2

Q = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.1)), jnp.sin(jnp.radians(0.1)), 0.1, 0.001, 0.001, 0.1]))**2
# Q = jnp.eye(6)*0.1
R = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01)), jnp.sin(jnp.radians(0.01)), 0.01, 0.0001, 0.0001]))**2
R_psuedo = jnp.diag(jnp.array([0.000001]))

est_dist = []
est_bearing = []
est_cos_sin = []
est_pixel_size = []
est_relative_velocity_x = []
est_relative_velocity_y = []
det_bearing = []
std_pixel_size = []
std_inverse_distance = []

for bearing, pixel_size, own_vel in zip(bearings, pixel_sizes, own_velocities):
    measurement = jnp.array([jnp.cos(bearing), jnp.sin(bearing), pixel_size, 0.0, 0.0])
    mu, sigma = kalman_update(mu, sigma, own_vel, measurement, Q, R, R_psuedo, Ts, A)
    # print(np.linalg.norm(mu[:2]))
    est_dist.append(mu[5])
    est_bearing.append(np.arctan2(mu[1], mu[0]))
    est_cos_sin.append(mu[:2])
    est_pixel_size.append(mu[2])
    est_relative_velocity_x.append(mu[3])
    est_relative_velocity_y.append(mu[4])
    det_bearing.append(np.linalg.det(sigma[:2, :2]))
    std_pixel_size.append(np.sqrt(sigma[2, 2]))
    std_inverse_distance.append(np.sqrt(sigma[5, 5]))


plt.figure(1)
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

plt.figure(2)
plt.subplot(221)
plt.plot(det_bearing, label='Det Bearing')
plt.xlabel('Time')
plt.ylabel('Det Bearing')
plt.title('Det Bearing')

plt.subplot(222)
plt.plot(std_pixel_size, label='Std Pixel Size')
plt.xlabel('Time')
plt.ylabel('Std Pixel Size')
plt.title('Std Pixel Size')

plt.subplot(223)
plt.plot(std_inverse_distance, label='Std Inverse Distance')
plt.xlabel('Time')
plt.ylabel('Std Inverse Distance')
plt.title('Std Inverse Distance')

plt.tight_layout()

plt.figure(3)
plt.subplot(121)
plt.plot(np.cos(bearings), label='True Cos')
plt.plot(np.array(est_cos_sin)[:, 0], label='Estimated Cos')
plt.title('State 1 Cos')
plt.legend()

plt.subplot(122)
plt.plot(np.sin(bearings), label='True Sin')
plt.plot(np.array(est_cos_sin)[:, 1], label='Estimated Sin')
plt.title('State 2 Sin')
plt.legend()

plt.tight_layout()
plt.show()


