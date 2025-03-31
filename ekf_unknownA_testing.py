import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.ekf_unknownA_continuous import kalman_update
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

los_n = jnp.cos(bearings[0])
los_e = jnp.sin(bearings[0])
pixel_size = pixel_sizes[0]
c_n = 30*np.cos(intruder_heading)
c_e = 30.*np.sin(intruder_heading)
eta = 1./true_distance[0]
A = 20

mu20 = jnp.array([los_n, los_e, pixel_size, c_n, c_e, eta, 20])
mu15 = jnp.array([los_n, los_e, pixel_size, c_n, c_e, eta, 15])
mu10 = jnp.array([los_n, los_e, pixel_size, c_n, c_e, eta, 10])
sigma20 = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01))**2, jnp.sin(jnp.radians(0.01))**2, 1**2, 
                              0.1**2, 0.1**2, 0.01**2, 3**2]))
sigma15 = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01))**2, jnp.sin(jnp.radians(0.01))**2, 1**2, 
                              0.1**2, 0.1**2, 0.01**2, 3**2]))
sigma10 = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.01))**2, jnp.sin(jnp.radians(0.01))**2, 1**2, 
                              0.1**2, 0.1**2, 0.01**2, 3**2]))

Q = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.1))**2, jnp.sin(jnp.radians(0.1))**2, 0.1**2, 
                        0.1**2, 0.1**2, 0.1**2, 0.1**2]))
R = jnp.diag(jnp.array([jnp.cos(jnp.radians(0.1))**2, jnp.sin(jnp.radians(0.1))**2, 0.01**2, 0.0001**2, 0.0001**2]))

est_dist20 = []
est_bearing20 = []
est_pixel_size20 = []
est_A20 = []
std_A20 = []
std_pixel_size20 = []
std_inverse_distance20 = []
det_bearing20 = []

est_dist15 = []
est_bearing15 = []
est_pixel_size15 = []
est_A15 = []
std_A15 = []
std_pixel_size15 = []
std_inverse_distance15 = []
det_bearing15 = []

est_dist10 = []
est_bearing10 = []
est_pixel_size10 = []
est_A10 = []
std_A10 = []
std_pixel_size10 = []
std_inverse_distance10 = []
det_bearing10 = []

for bearing, pixel_size, own_vel in zip(bearings, pixel_sizes, own_velocities):
    measurement = jnp.array([jnp.cos(bearing), jnp.sin(bearing), pixel_size, 0.0, 0.0])

    mu20, sigma20 = kalman_update(mu20, sigma20, own_vel, measurement, Q, R, Ts)
    est_dist20.append(mu20[5])
    est_bearing20.append(np.arctan2(mu20[1], mu20[0]))
    est_pixel_size20.append(mu20[2])
    est_A20.append(mu20[6])
    std_A20.append(sigma20[6, 6])
    std_pixel_size20.append(sigma20[2, 2])
    std_inverse_distance20.append(sigma20[5, 5])
    det_bearing20.append(np.linalg.det(sigma20[:2,:2]))

    mu15, sigma15 = kalman_update(mu15, sigma15, own_vel, measurement, Q, R, Ts)
    est_dist15.append(mu15[5])
    est_bearing15.append(np.arctan2(mu15[1], mu15[0]))
    est_pixel_size15.append(mu15[2])
    est_A15.append(mu15[6])
    std_A15.append(sigma15[6, 6])
    std_pixel_size15.append(sigma15[2, 2])
    std_inverse_distance15.append(sigma15[5, 5])
    det_bearing15.append(np.linalg.det(sigma15[:2,:2]))

    mu10, sigma10 = kalman_update(mu10, sigma10, own_vel, measurement, Q, R, Ts)
    est_dist10.append(mu10[5])
    est_bearing10.append(np.arctan2(mu10[1], mu10[0]))
    est_pixel_size10.append(mu10[2])
    est_A10.append(mu10[6])
    std_A10.append(sigma10[6, 6])
    std_pixel_size10.append(sigma10[2, 2])
    std_inverse_distance10.append(sigma10[5, 5])
    det_bearing10.append(np.linalg.det(sigma10[:2,:2]))




plt.figure(1)
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

plt.tight_layout()

plt.figure(2)
plt.subplot(221)
plt.plot(bearings - np.array(est_bearing20), label='Est A=20')
plt.plot(bearings - np.array(est_bearing15), label='Est A=15')
plt.plot(bearings - np.array(est_bearing10), label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Bearing Error')
plt.title('Bearing Error')
plt.legend()

plt.subplot(222)
plt.plot(pixel_sizes - np.array(est_pixel_size20), label='Est A=20')
plt.plot(pixel_sizes - np.array(est_pixel_size15), label='Est A=15')
plt.plot(pixel_sizes - np.array(est_pixel_size10), label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Pixel Size Error')
plt.title('Pixel Size Error')
plt.legend()

plt.subplot(223)
plt.plot(true_distance - 1/np.array(est_dist20), label='Est A=20')
plt.plot(true_distance - 1/np.array(est_dist15), label='Est A=15')
plt.plot(true_distance - 1/np.array(est_dist10), label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Distance Error')
plt.title('Distance Error')
plt.legend()

plt.subplot(224)
plt.plot(15 - np.array(est_A20), label='Est A=20')
plt.plot(15 - np.array(est_A15), label='Est A=15')
plt.plot(15 - np.array(est_A10), label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Wingspan Error')
plt.title('A Error')
plt.legend()

plt.tight_layout()

plt.figure(3)
plt.subplot(221)
plt.plot(std_A20, label='Est A=20')
plt.plot(std_A15, label='Est A=15')
plt.plot(std_A10, label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Standard Deviation A')
plt.title('Standard Deviation A')
plt.legend()

plt.subplot(222)
plt.plot(std_pixel_size20, label='Est A=20')
plt.plot(std_pixel_size15, label='Est A=15')
plt.plot(std_pixel_size10, label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Standard Deviation Pixel Size')
plt.title('Standard Deviation Pixel Size')
plt.legend()

plt.subplot(223)
plt.plot(std_inverse_distance20, label='Est A=20')
plt.plot(std_inverse_distance15, label='Est A=15')
plt.plot(std_inverse_distance10, label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Standard Deviation Inverse Distance')
plt.title('Standard Deviation Inverse Distance')
plt.legend()

plt.subplot(224)
plt.plot(det_bearing20, label='Est A=20')
plt.plot(det_bearing15, label='Est A=15')
plt.plot(det_bearing10, label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Det Bearing')
plt.title('Det Bearing')
plt.legend()

plt.tight_layout()
plt.show()




