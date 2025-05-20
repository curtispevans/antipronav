import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.ekf_modified_polar_coordinates_unknownA import kalman_update
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
intruder_vel = np.array([0., 0.])
intruder_heading = np.pi/2
A = 10

mu = np.array([0, 0, bearings[0], 1./true_distance[0], A])
sigma = np.diag(np.array([np.radians(0.1), 0.0001, np.radians(0.1), 0.1, 10]))**2

Q = np.diag(np.array([np.radians(0.1), 1e-5, np.radians(0.1), 1e-2, 0.1]))**2
R = np.diag(np.array([np.radians(1), 0.1]))**2

est_dist = []
est_bearing = []
est_pixel_size = []
est_cn = []
est_ce = []
std_bearing = []
std_pixel_size = []
std_int_vel = []
std_inverse_distance = []
std_A = []
est_A = []

for bearing, pixel_size, own_mav, u in zip(bearings[1:], pixel_sizes[1:], mav_states[1:], us[1:]):
    measurement = np.array([bearing, pixel_size])
    # measurement = np.array([bearing, pixel_size])
    mu, sigma = kalman_update(mu, sigma, own_mav, u, measurement, Q, R, Ts)
    # print(np.linalg.norm(mu[:2]))
    est_dist.append(mu[3])
    est_bearing.append(mu[2])
    # est_pixel_size.append(mu[4])
    est_A.append(mu[4])
    std_bearing.append(np.sqrt(sigma[2, 2]))
    # std_pixel_size.append(np.sqrt(sigma[4, 4]))
    std_inverse_distance.append(np.sqrt(sigma[3, 3])) 
    std_A.append(np.sqrt(sigma[4, 4]))
    


plt.figure(1)
plt.subplot(221)
plt.plot(bearings, label='True Bearing')
plt.plot(est_bearing, label='Estimated Bearing')
plt.xlabel('Time')
plt.ylabel('Bearing')
plt.legend()
plt.title('Bearing between Mavs')

# plt.subplot(222)
# plt.plot(pixel_sizes, label='True Pixel Size')
# plt.plot(est_pixel_size, label='Estimated Pixel Size')
# plt.xlabel('Time')
# plt.ylabel('Pixel Size')
# plt.title('Pixel Size')
# plt.legend()


plt.subplot(223)
plt.plot(1/true_distance, label='True inverse distance')
plt.plot(est_dist, label='Estimated inverse distance')
plt.xlabel('Time')
plt.ylabel('Inverse Distance')
plt.title('Inverse Distance between Mavs')
plt.legend()

plt.subplot(224)
plt.plot(20*np.ones(len(est_A)), label='True A')
plt.plot(est_A, label='Estimated A')
plt.xlabel('Time')
plt.ylabel('Wingspan')
plt.title('A')
plt.legend()

plt.tight_layout()

plt.figure(2)
plt.subplot(221)
plt.plot(std_bearing, label='Std Bearing')
plt.xlabel('Time')
plt.ylabel('Std Bearing')
plt.title('Std Bearing')

# plt.subplot(222)
# plt.plot(std_pixel_size, label='Std Pixel Size')
# plt.xlabel('Time')
# plt.ylabel('Std Pixel Size')
# plt.title('Std Pixel Size')

plt.subplot(223)
plt.plot(std_inverse_distance, label='Std Inverse Distance')
plt.xlabel('Time')
plt.ylabel('Std Inverse Distance')
plt.title('Std Inverse Distance')

plt.subplot(224)
plt.plot(std_A, label='Std A')
plt.xlabel('Time')
plt.ylabel('Std A')
plt.title('Std A')
plt.legend()
plt.tight_layout()

plt.figure(3)
mav_poses, intruder_poses = get_all_own_poses_and_intruder_poses(mav_states, true_distance, bearings)
plt.scatter([mav[1] for mav in mav_poses], [mav[0] for mav in mav_poses], c='r', marker='o', label='Ownship')
plt.scatter([intruder[1] for intruder in intruder_poses], [intruder[0] for intruder in intruder_poses], c='b', marker='o', label='Intruder True')
mav_poses, intruder_poses_est = get_all_own_poses_and_intruder_poses(mav_states[1:], 1/np.array(est_dist), est_bearing)
plt.scatter([intruder[1] for intruder in intruder_poses_est], [intruder[0] for intruder in intruder_poses_est], c='g', marker='o', label='Intruder Est')


for i in range(0, len(intruder_poses),30):
    x_mav, y_mav = mav_poses[i][1], mav_poses[i][0]
    x_true, y_true = intruder_poses[i][1], intruder_poses[i][0]
    x_est, y_est = intruder_poses_est[i][1], intruder_poses_est[i][0]
    plt.plot([x_mav, x_true], [y_mav, y_true], 'r--', linewidth=1)
    plt.plot([x_true, x_est], [y_true, y_est], 'k--', linewidth=1)

plt.legend()

# save the figure as a png file
plt.xlabel('East')
plt.ylabel('North')
plt.title('Mav Position')

plt.grid()
plt.tight_layout()
plt.savefig('ekf_modified_polar_coordinates_unknownA.png', dpi=300)

plt.show()


