import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.kalman_filter_5_states import kalman_update
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

Ts = 1/30
intruder_vel = np.array([0., 0.])
intruder_heading = np.pi/2
A = 15

mu = jnp.array([bearings[0], pixel_sizes[0], 0*np.cos(intruder_heading), 0.*np.sin(intruder_heading), 1./true_distance[0]])
sigma = jnp.diag(jnp.array([jnp.radians(0.1), 1, .001, .001, 0.01]))**2

Q = jnp.diag(jnp.array([jnp.radians(0.01), 0.1, 0.000001, 0.000001, 0.1]))**2
# Q = jnp.eye(6)*0.1
R = jnp.diag(jnp.array([jnp.radians(0.01), 0.01, 0.0001]))**2
R_psuedo = jnp.diag(jnp.array([0.000001]))

est_dist = []
est_bearing = []
est_pixel_size = []
est_cn = []
est_ce = []
std_bearing = []
std_pixel_size = []
std_int_vel = []
std_inverse_distance = []

for bearing, pixel_size, own_vel in zip(bearings, pixel_sizes, own_velocities):
    measurement = jnp.array([bearing, pixel_size, 0.0])
    mu, sigma = kalman_update(mu, sigma, own_vel, measurement, Q, R, Ts, A)
    # print(np.linalg.norm(mu[:2]))
    est_dist.append(mu[4])
    est_bearing.append(mu[0])
    est_pixel_size.append(mu[1])
    est_cn.append(mu[2])
    est_ce.append(mu[3])
    std_bearing.append(np.sqrt(sigma[0, 0]))
    std_pixel_size.append(np.sqrt(sigma[1, 1]))
    std_int_vel.append(np.linalg.det(sigma[2:4, 2:4]))
    std_inverse_distance.append(np.sqrt(sigma[4, 4]))


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


# plt.subplot(224)
# plt.plot(true_distance, label='True Distance')
# plt.plot(1/np.array(est_dist), label='Estimated Distance')
# plt.xlabel('Time')
# plt.ylabel('Distance')
# plt.title('Distance between Mavs')
# plt.legend()

plt.subplot(224)
plt.plot(np.ones(len(est_cn))*intruder_vel[0], label='True Relative Velocity N')
plt.plot(np.ones(len(est_ce))*intruder_vel[1], label='True Relative Velocity E')
plt.plot(est_cn, label='Estimated Intruder Vel N')
plt.plot(est_ce, label='Estimated Intruder Vel E')
plt.xlabel('Time')
plt.ylabel('Relative Velocity')
plt.title('Relative Velocity')
plt.legend()

plt.tight_layout()

plt.figure(2)
plt.subplot(221)
plt.plot(std_bearing, label='Std Bearing')
plt.xlabel('Time')
plt.ylabel('Std Bearing')
plt.title('Std Bearing')

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
# plt.tight_layout()

plt.subplot(224)
plt.plot(std_int_vel, label='Std Intruder Vel')
plt.xlabel('Time')
plt.ylabel('Std Intruder Vel')
plt.title('Std Intruder Vel')
plt.legend()
plt.tight_layout()

plt.figure(3)
mav_poses, intruder_poses = get_all_own_poses_and_intruder_poses(mav_states, true_distance, bearings)
plt.plot([mav[1] for mav in mav_poses], [mav[0] for mav in mav_poses], 'ro', label='Mav 1')
plt.plot([intruder[1] for intruder in intruder_poses], [intruder[0] for intruder in intruder_poses], 'bo', label='Intruder True')
mav_poses, intruder_poses = get_all_own_poses_and_intruder_poses(mav_states, 1/np.array(est_dist), est_bearing)
plt.plot([intruder[1] for intruder in intruder_poses], [intruder[0] for intruder in intruder_poses], 'go', label='Intruder Est')
plt.legend()

plt.tight_layout()
plt.show()


