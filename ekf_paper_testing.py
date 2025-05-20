import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.ekf_paper import kalman_update
from models.mav_dynamics import MavDynamics

def get_own_pose_and_intruder_pose(mav1, distance, bearing):
    # Get the position of the intruder
    intruder_pos = mav1[0:2] + distance * np.array([np.cos(bearing + mav1[2]), np.sin(bearing + mav1[2])])

    return mav1[0:2], intruder_pos



def get_all_own_poses_and_intruder_poses(mav1_list, distances, bearings):
    # Get the position of the intruder
    mav1_poses = []
    intruder_poses = []
    for mav1, distance, bearing in zip(mav1_list, distances, bearings):
        mav1_pose, intruder_pose = get_own_pose_and_intruder_pose(mav1, distance, bearing)
        mav1_poses.append(mav1_pose)
        intruder_poses.append(intruder_pose)
    return mav1_poses, intruder_poses

bearings = np.load('data/bearing.npy')
pixel_sizes = np.load('data/pixel_sizes.npy')
true_distance = np.load('data/distances.npy')
control = np.load('data/control.npy')
relative_velocities = np.load('data/relative_velocity.npy')
own_velocities = np.load('data/own_velocity.npy')
mav_states = np.load('data/mav_state.npy')
us = np.load('data/us.npy')

Ts = 1/30

mu = np.array([0, 0, bearings[0],1/true_distance[0]])
G = np.array([[10/300**2, 0, 1/300, 0],
              [0, -6010/300**2, 0, -1/300],
              [1/300, 0, 0, 0],
              [0, -1/300, 0, 0]])
P = np.diag([1, 1, 1, 1])**2
# sigma = G @ P @ G.T
sigma = np.diag(np.array([np.radians(0.1), 0.001, np.radians(0.1), 0.1]))**2

Q = np.diag(np.array([np.radians(0.001), 1e-5, np.radians(0.001), 1e-5]))**2
# Q = jnp.eye(6)*0.1
R = np.diag(np.array([[np.radians(0.0001)]]))**2
R_psuedo = np.diag(np.array([0.000001]))

est_dist = []
est_bearing = []
std_bearing = []
std_pixel_size = []
std_int_vel = []
std_inverse_distance = []

for bearing, pixel_size, own_mav, u in zip(bearings[:], pixel_sizes[:], mav_states[:], us[:]):
    measurement = np.array([bearing])
    mu, sigma = kalman_update(mu, sigma, own_mav, u, measurement, Q, R, Ts)
    est_dist.append(mu[3])
    est_bearing.append(mu[2])
    std_bearing.append(np.sqrt(sigma[2, 2]))
    std_inverse_distance.append(np.sqrt(sigma[3, 3]))


plt.figure(1)
plt.subplot(121)
plt.plot(bearings, label='True Bearing')
plt.plot(est_bearing, label='Estimated Bearing')
plt.xlabel('Time')
plt.ylabel('Bearing')
plt.legend()
plt.title('Bearing between Mavs')


plt.subplot(122)
plt.plot(1/true_distance, label='True inverse distance')
plt.plot(est_dist, label='Estimated inverse distance')
plt.xlabel('Time')
plt.ylabel('Inverse Distance')
plt.title('Inverse Distance between Mavs')
plt.legend()

plt.tight_layout()

plt.figure(2)
plt.subplot(121)
plt.plot(std_bearing, label='Std Bearing')
plt.xlabel('Time')
plt.ylabel('Std Bearing')
plt.title('Std Bearing')

plt.subplot(122)
plt.plot(std_inverse_distance, label='Std Inverse Distance')
plt.xlabel('Time')
plt.ylabel('Std Inverse Distance')
plt.title('Std Inverse Distance')
plt.tight_layout()

plt.figure(3)
mav_poses, intruder_poses = get_all_own_poses_and_intruder_poses(mav_states, true_distance, bearings)
plt.scatter([mav[1] for mav in mav_poses], [mav[0] for mav in mav_poses], c='r', marker='o', label='Mav 1')
plt.scatter([intruder[1] for intruder in intruder_poses], [intruder[0] for intruder in intruder_poses], c='b', marker='o', label='Intruder True')
mav_poses, intruder_poses_est = get_all_own_poses_and_intruder_poses(mav_states[:], 1/np.array(est_dist), est_bearing)
plt.scatter([intruder[1] for intruder in intruder_poses_est], [intruder[0] for intruder in intruder_poses_est], c='g', marker='o', label='Intruder Est')


for i in range(0, len(intruder_poses),30):
    x_mav, y_mav = mav_poses[i][1], mav_poses[i][0]
    x_true, y_true = intruder_poses[i][1], intruder_poses[i][0]
    x_est, y_est = intruder_poses_est[i][1], intruder_poses_est[i][0]
    plt.plot([x_mav, x_true], [y_mav, y_true], 'r--', linewidth=1)
    plt.plot([x_true, x_est], [y_true, y_est], 'k--', linewidth=1)

plt.legend()

plt.tight_layout()
plt.show()



