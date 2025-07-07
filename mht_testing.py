import models.mht_A as mht
import numpy as np
import matplotlib.pyplot as plt

bearings = np.load('data/bearing.npy')
pixel_sizes = np.load('data/pixel_sizes.npy')
true_distance = np.load('data/distances.npy')
control = np.load('data/control.npy')
relative_velocities = np.load('data/relative_velocity.npy')
own_velocities = np.load('data/own_velocity.npy')
mav_states = np.load('data/mav_state.npy')
us = np.load('data/us.npy')

Ts = 1/30
min_A = 2
max_A = 40

mu_inverse_distance = np.array([0, 0, bearings[0], 1/true_distance[0]])
sigma_inverse_distance = np.diag(np.array([np.radians(0.1), 0.001, np.radians(0.1), 0.01]))**2
Q_inverse_distance = np.diag(np.array([np.radians(0.01), 1e-5, np.radians(0.01), 1e-5]))**2
R_inverse_distance = np.diag(np.array([np.radians(0.0001), 0.0001]))**2

Q_tmp = np.eye(2)*0.01**2
Q_nearly_constant_accel = np.block([[Ts**5/20*Q_tmp, Ts**4/8*Q_tmp, Ts**3/6*Q_tmp],
                                    [Ts**4/8*Q_tmp, Ts**3/3*Q_tmp, Ts**2/2*Q_tmp],
                                    [Ts**3/6*Q_tmp, Ts**2/2*Q_tmp, Ts*Q_tmp]])
# print(np.round(Q_nearly_constant_accel, 3))
# Q_nearly_constant_accel = np.eye(6)*0.01**2
# print(Q_nearly_constant_accel)
R_nearly_constant_accel = np.diag(np.array([1e-5, 1e-5]))**2

intruders_dict = {}




for i in range(min_A, max_A):
    # Get the position of the intruder
    distance = i/pixel_sizes[0]  # distance in meters
    bearing = bearings[i]
    own_pose = mav_states[i][0:2]  # own position
    own_heading = mav_states[i][2]  # own heading in radians
    los = np.array([np.cos(bearing + own_heading), np.sin(bearing + own_heading)])
    int_x = own_pose[0] + distance * los[0]
    int_y = own_pose[1] + distance * los[1]
    vel_x = relative_velocities[i][0] + own_velocities[i][0]
    vel_y = relative_velocities[i][1] + own_velocities[i][1]

    mu_nearly_constant_accel = np.array([int_x, int_y, vel_x, vel_y, 0, 0])
    sigma_nearly_constant_accel = np.eye(6)*1**2
    filter_counter = 0
    intruders_dict[i] = [mu_inverse_distance.copy(), sigma_inverse_distance.copy(), mu_nearly_constant_accel.copy(), sigma_nearly_constant_accel.copy(), filter_counter]

full_inverse_distance = []
partial_inverse_distance = []

intruder_poses = {i:[] for i in range(min_A, max_A)}
inv_distances = {i:[] for i in range(min_A, max_A)}
for i in range(100):
    bearing = bearings[i+1]
    pixel_size = pixel_sizes[i+1]
    u = us[i+1]
    own_mav = mav_states[i+1]


    measurement = np.array([bearing, pixel_size])

    
    
    
    # Propagate candidates for inverse distance
    intruders_dict = mht.propagate_candidates_inverse_distance(intruders_dict, own_mav, u, measurement, Ts, Q_inverse_distance, R_inverse_distance)
    
    # Propagate candidates for nearly constant acceleration
    intruders_dict = mht.propagate_candidates_intruder_pos(intruders_dict, own_mav, Ts, Q_nearly_constant_accel, R_nearly_constant_accel)

    # Filter candidates
    if i > 30:
        # intruders_dict = mht.filter_candidates(intruders_dict, vel_threshold=150, g_force_threshold=0.01)
        # intruders_dict = mht.filter_candidates_probabilistic(intruders_dict, prob_threshold=-3)
        # intruders_dict = mht.filter_state_measurement_probabilistic(intruders_dict, measurement, R_inverse_distance, mahalanobis_dist=np.inf)
        intruders_dict = mht.filter_pose_measurement_probabilistic(intruders_dict, own_mav, R_nearly_constant_accel, 1)


    

    # Plot candidates

    # print(np.linalg.norm(intruders_dict[18][2][4:])/9.81, np.linalg.norm(intruders_dict[18][2][2:4]))  # Print g-force of candidate 2
    
    for A in intruders_dict.keys():
        # intruder_state = intruders_dict[A][2]
        intruder_state = intruders_dict[A][2][:2]
        intruder_poses[A].append(intruder_state[0:2])
        inv_distances[A].append(intruders_dict[A][0][3])

print('Finished processing candidates.')

print(f'Number of candidates: {len(intruders_dict)}')
print(f"Remaining Candidates for A:\n", intruders_dict.keys())
for A in intruders_dict.keys():
    print(A, intruders_dict[A][4])

# print(f"intruder poses: {intruder_poses}")

print('Plotting candidates...')
# Plotting the intruder positions
plt.figure(1)
for A in intruder_poses.keys():
    # print(intruder_poses[A])
    intruder_pos = np.array(intruder_poses[A])
    plt.plot(intruder_pos[:, 1], intruder_pos[:, 0], 'ko', label=f'Candidate {A}')

print('Plotting own MAV position...')
plt.plot(mav_states[:, 1], mav_states[:, 0], 'ro', label='Own Mav')


# plt.figure(2)
# for A in inv_distances.keys():
#     # print(A)
#     plt.plot(np.abs(pixel_sizes[1:] - A*np.array(inv_distances[A])), label=f'Candidate {A}')
# # plt.plot(1/true_distance, label='True Inverse Distance', linestyle='--')
# plt.xlabel('Time Step')
# plt.ylabel('True Pixel Size - Estimated Pixel Size')

# plt.legend()
# plt.tight_layout()

plt.figure(3)
for A in inv_distances.keys():
    plt.plot(inv_distances[A], label=f'Candidate {A}')
plt.plot(1/true_distance[1:], label='True Inverse Distance', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Inverse Distance (m)')
plt.legend()
plt.tight_layout()

plt.figure(4)
for A in inv_distances.keys():
    plt.plot(A*np.array(inv_distances[A]), label=f'Candidate {A}')
plt.plot(pixel_sizes[1:], label='True Pixel Size', linestyle='--')
plt.plot(10/true_distance[1:], label='True Pixel Size (10m)', linestyle='--')
plt.title('Estimated Pixel Size vs True Pixel Size')
plt.xlabel('Time Step')
plt.ylabel('Estimated Pixel Size (m)')
plt.legend()
plt.tight_layout()


plt.show()



