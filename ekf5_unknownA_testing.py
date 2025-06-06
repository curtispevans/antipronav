import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.ekf5_unknownA import kalman_update
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
intruder_vel = np.array([0., 10.])
intruder_heading = np.pi/2

bearing = bearings[0]
pixel_size = pixel_sizes[0]
c_n = 10*np.cos(intruder_heading)
c_e = 10.*np.sin(intruder_heading)
eta = 1./true_distance[0]
A = 20

mu20 = jnp.array([bearing, c_n, c_e, eta, 20])
mu15 = jnp.array([bearing, c_n, c_e, eta, 15])
mu10 = jnp.array([bearing, c_n, c_e, eta, 10])
sigma20 = jnp.diag(jnp.array([np.radians(1)**2, 0.1**2, 0.1**2, 0.01**2, 0.1**2]))
sigma15 = jnp.diag(jnp.array([np.radians(1)**2, 0.1**2, 0.1**2, 0.01**2, 15**2]))
sigma10 = jnp.diag(jnp.array([np.radians(1)**2, 0.1**2, 0.1**2, 0.01**2, 10**2]))

Q = jnp.diag(jnp.array([np.radians(0.01), 0.001**2, 0.001**2, 0.01**2, 0.1**2]))
R = jnp.diag(jnp.array([np.radians(0.1), 0.01**2]))

est_dist20 = [mu20[3]]
est_bearing20 = [mu20[0]]
# est_pixel_size20 = [mu20[1]]
est_A20 = [mu20[4]]
std_A20 = [sigma20[4, 4]]
# std_pixel_size20 = [sigma20[1,1]]
std_inverse_distance20 = [sigma20[3, 3]]
det_bearing20 = [sigma20[0,0]]

est_dist15 = [mu15[3]]
est_bearing15 = [mu15[0]]
# est_pixel_size15 = [mu15[1]]
est_A15 = [mu15[4]]
std_A15 = [sigma15[4, 4]]
# std_pixel_size15 = [sigma15[1, 1]]
std_inverse_distance15 = [sigma15[3, 3]]
det_bearing15 = [sigma15[0,0]]

est_dist10 = [mu10[3]]
est_bearing10 = [mu10[0]]
# est_pixel_size10 = [mu10[1]]
est_A10 = [mu10[4]]
std_A10 = [sigma10[4, 4]]
# std_pixel_size10 = [sigma10[1, 1]]
std_inverse_distance10 = [sigma10[3, 3]]
det_bearing10 = [sigma10[0,0]]

for bearing, pixel_size, own_vel in zip(bearings[1:], pixel_sizes[1:], own_velocities[1:]):
    measurement = jnp.array([bearing, pixel_size])

    mu20, sigma20 = kalman_update(mu20, sigma20, own_vel, measurement, Q, R, Ts)
    est_dist20.append(mu20[3])
    est_bearing20.append(mu20[0])
    # est_pixel_size20.append(mu20[1])
    est_A20.append(mu20[4])
    std_A20.append(sigma20[4, 4])
    # std_pixel_size20.append(sigma20[1, 1])
    std_inverse_distance20.append(sigma20[3, 3])
    det_bearing20.append(sigma20[0,0])

    mu15, sigma15 = kalman_update(mu15, sigma15, own_vel, measurement, Q, R, Ts)
    est_dist15.append(mu15[3])
    est_bearing15.append(mu15[0])
    # est_pixel_size15.append(mu15[1])
    est_A15.append(mu15[4])
    std_A15.append(sigma15[4, 4])
    # std_pixel_size15.append(sigma15[1, 1])
    std_inverse_distance15.append(sigma15[3, 3])
    det_bearing15.append(sigma15[0,0])

    mu10, sigma10 = kalman_update(mu10, sigma10, own_vel, measurement, Q, R, Ts)
    est_dist10.append(mu10[3])
    est_bearing10.append(mu10[0])
    # est_pixel_size10.append(mu10[1])
    est_A10.append(mu10[4])
    std_A10.append(sigma10[4, 4])
    # std_pixel_size10.append(sigma10[1, 1])
    std_inverse_distance10.append(sigma10[3, 3])
    det_bearing10.append(sigma10[0,0])




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

# plt.subplot(222)
# plt.plot(pixel_sizes, label='True Pixel Size')
# plt.plot(est_pixel_size20, label='Est A=20')
# plt.plot(est_pixel_size15, label='Est A=15')
# plt.plot(est_pixel_size10, label='Est A=10')
# plt.xlabel('Time')
# plt.ylabel('Pixel Size')
# plt.title('Pixel Size')
# plt.legend()


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
plt.plot(20*np.ones(len(est_A20)), label='True A')
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
plt.plot(np.abs(bearings - np.array(est_bearing20)), label='Est A=20')
plt.plot(np.abs(bearings - np.array(est_bearing15)), label='Est A=15')
plt.plot(np.abs(bearings - np.array(est_bearing10)), label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Bearing Error')
plt.title('Bearing Error')
plt.legend()

# plt.subplot(222)
# plt.plot(np.abs(pixel_sizes - np.array(est_pixel_size20)), label='Est A=20')
# plt.plot(np.abs(pixel_sizes - np.array(est_pixel_size15)), label='Est A=15')
# plt.plot(np.abs(pixel_sizes - np.array(est_pixel_size10)), label='Est A=10')
# plt.xlabel('Time')
# plt.ylabel('Pixel Size Error')
# plt.title('Pixel Size Error')
# plt.legend()

plt.subplot(223)
plt.plot(np.abs(true_distance - 1/np.array(est_dist20)), label='Est A=20')
plt.plot(np.abs(true_distance - 1/np.array(est_dist15)), label='Est A=15')
plt.plot(np.abs(true_distance - 1/np.array(est_dist10)), label='Est A=10')
plt.xlabel('Time')
plt.ylabel('Distance Error')
plt.title('Distance Error')
plt.legend()

plt.subplot(224)
plt.plot(np.abs(20 - np.array(est_A20)), label='Est A=20')
plt.plot(np.abs(20 - np.array(est_A15)), label='Est A=15')
plt.plot(np.abs(20 - np.array(est_A10)), label='Est A=10')
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

# plt.subplot(222)
# plt.plot(std_pixel_size20, label='Est A=20')
# plt.plot(std_pixel_size15, label='Est A=15')
# plt.plot(std_pixel_size10, label='Est A=10')
# plt.xlabel('Time')
# plt.ylabel('Standard Deviation Pixel Size')
# plt.title('Standard Deviation Pixel Size')
# plt.legend()

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

plt.figure(4)
mav_poses, intruder_poses = get_all_own_poses_and_intruder_poses(mav_states, true_distance, bearings)
plt.plot([mav_pose[1] for mav_pose in mav_poses], [mav_pose[0] for mav_pose in mav_poses], 'ro')
plt.plot([intruder_pose[1] for intruder_pose in intruder_poses], [intruder_pose[0] for intruder_pose in intruder_poses], 'bo', label='Intruder True')
mav_poses, intruder_poses = get_all_own_poses_and_intruder_poses(mav_states, 1/np.array(est_dist20), est_bearing20)
plt.plot([intruder_pose[1] for intruder_pose in intruder_poses], [intruder_pose[0] for intruder_pose in intruder_poses], 'go', label='Intruder Est A=20')
mav_poses, intruder_poses = get_all_own_poses_and_intruder_poses(mav_states, 1/np.array(est_dist15), est_bearing15)
plt.plot([intruder_pose[1] for intruder_pose in intruder_poses], [intruder_pose[0] for intruder_pose in intruder_poses], 'co', label='Intruder Est A=15')
mav_poses, intruder_poses = get_all_own_poses_and_intruder_poses(mav_states, 1/np.array(est_dist10), est_bearing10)
plt.plot([intruder_pose[1] for intruder_pose in intruder_poses], [intruder_pose[0] for intruder_pose in intruder_poses], 'yo', label='Intruder Est A=10')
plt.tight_layout()
plt.legend()
plt.show()




