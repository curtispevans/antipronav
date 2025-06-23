import models.mht_full as mht
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

mu_inverse_distance = np.array([0, 0, bearings[0], 1/true_distance[0]])
sigma_inverse_distance = np.diag(np.array([np.radians(0.1), 0.001, np.radians(0.1), 0.01]))**2
Q_inverse_distance = np.diag(np.array([np.radians(0.01), 1e-5, np.radians(0.01), 1e-5]))**2
R_inverse_distance = np.diag(np.array([np.radians(0.001), 0.001]))**2

Q_tmp = np.eye(2)*0.01**2
Q_nearly_constant_accel = np.block([[Ts**5/20*Q_tmp, Ts**4/8*Q_tmp, Ts**3/6*Q_tmp],
                                    [Ts**4/8*Q_tmp, Ts**3/3*Q_tmp, Ts**2/2*Q_tmp],
                                    [Ts**3/6*Q_tmp, Ts**2/2*Q_tmp, Ts*Q_tmp]])
# print(np.round(Q_nearly_constant_accel, 3))
# Q_nearly_constant_accel = np.eye(6)*0.01**2
# print(Q_nearly_constant_accel)
R_nearly_constant_accel = np.diag(np.array([1e-5, 1e-5]))**2

intruders_dict_full_state = {}

# Q_full_state = np.diag(np.array([np.radians(0.01), 1e-5, np.radians(0.01), 1e-5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]))**2
Q_full_state = np.block([[Q_inverse_distance, np.zeros((4,6))],
                         [np.zeros((6,4)), Q_nearly_constant_accel]])
# R_full_state = np.diag(np.array([np.radians(1e-5), 1e-5, 1e-10, 1e-10]))**2
R_full_state = np.diag(np.array([np.radians(1e-10), 1e-10]))**2


for i in range(2, 40):
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
    intruders_dict_full_state[i] = [np.array([*mu_inverse_distance.copy(), *mu_nearly_constant_accel.copy()]),
                                    np.block([[sigma_inverse_distance.copy(), np.zeros((4,6))],
                                              [np.zeros((6,4)), sigma_nearly_constant_accel.copy()]])]

full_inverse_distance = []
partial_inverse_distance = []

intruder_poses = {i:[] for i in range(2, 40)}
for i in range(len(bearings[1:])):
    bearing = bearings[i+1]
    pixel_size = pixel_sizes[i+1]
    u = us[i+1]
    own_mav = mav_states[i+1]


    measurement = np.array([bearing, pixel_size])

    # Propagate candidates for full state
    intruders_dict_full_state = mht.propagate_full_state(intruders_dict_full_state, own_mav, u, measurement, Ts, Q_full_state, R_full_state)


    # Filter candidates
    if i > 30:
        intruders_dict_full_state = mht.filter_full_state_probabilistic(intruders_dict_full_state, own_mav, measurement, R_inverse_distance, m_dist_thres=1e-4)

    # Plot candidates

    # print(np.linalg.norm(intruders_dict[18][2][4:])/9.81, np.linalg.norm(intruders_dict[18][2][2:4]))  # Print g-force of candidate 2
    
    for A in intruders_dict_full_state.keys():
        # intruder_state = intruders_dict[A][2]
        intruder_state = intruders_dict_full_state[A][0][4:]
        intruder_poses[A].append(intruder_state[0:2])

print('Finished processing candidates.')

print(f'Number of candidates: {len(intruders_dict_full_state)}')
print(f"Remaining Candidates for A:\n", intruders_dict_full_state.keys())

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


plt.figure(2)
plt.plot(full_inverse_distance, label='Full State EKF Inverse Distance')
plt.plot(partial_inverse_distance, label='Partial Inverse Distance')
plt.xlabel('Time Step')
plt.ylabel('Inverse Distance (1/m)')
plt.title('Inverse Distance Over Time')
plt.legend()

plt.show()


