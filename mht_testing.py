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

mu_inverse_distance = np.array([0, 0, bearings[0], 0.5])
sigma_inverse_distance = np.diag(np.array([np.radians(0.1), 0.001, np.radians(0.1), 0.01]))**2
Q_inverse_distance = np.diag(np.array([np.radians(0.01), 1e-5, np.radians(0.01), 1e-5]))**2
R_inverse_distance = np.diag(np.array([np.radians(0.0000001), 0.00001]))**2

Q_tmp = np.ones((2,2))*0.1**2
Q_nearly_constant_accel = np.block([[Ts**5/20*Q_tmp, Ts**4/8*Q_tmp, Ts**3/6*Q_tmp],
                                    [Ts**4/8*Q_tmp, Ts**3/3*Q_tmp, Ts**2/2*Q_tmp],
                                    [Ts**3/6*Q_tmp, Ts**2/2*Q_tmp, Ts*Q_tmp]])
print(Q_nearly_constant_accel)
Q_nearly_constant_accel = np.eye(6)*0.01**2
print(Q_nearly_constant_accel)
R_nearly_constant_accel = np.diag(np.array([1e-5, 1e-5]))**2

intruders_dict = {}

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
    sigma_nearly_constant_accel = np.eye(6)*0.1**2
    intruders_dict[i] = [mu_inverse_distance.copy(), sigma_inverse_distance.copy(), mu_nearly_constant_accel.copy(), sigma_nearly_constant_accel.copy()]

intruder_poses = {i:[] for i in range(2, 40)}
for bearing, pixel_size, own_mav, u in zip(bearings[1:], pixel_sizes[1:], mav_states[1:], us[1:]):
    measurement = np.array([bearing, pixel_size])
    
    
    # Propagate candidates for inverse distance
    intruders_dict = mht.propagate_candidates_inverse_distance(intruders_dict, own_mav, u, measurement, Ts, Q_inverse_distance, R_inverse_distance)
   
    # Propagate candidates for nearly constant acceleration
    intruders_dict = mht.propagate_candidates_intruder_pos(intruders_dict, own_mav, Ts, Q_nearly_constant_accel, R_nearly_constant_accel)

    # Filter candidates
    intruders_dict = mht.filter_candidates(intruders_dict, vel_threshold=100, g_force_threshold=.05)
    # Plot candidates

    # print(np.linalg.norm(intruders_dict[2][2][4:])/9.81, np.linalg.norm(intruders_dict[2][2][2:4]))  # Print g-force of candidate 2
    
    for A in intruders_dict.keys():
        intruder_state = intruders_dict[A][2]
        intruder_poses[A].append(intruder_state[0:2])

print('Finished processing candidates.')

print('Plotting candidates...')
# Plotting the intruder positions
for A in intruder_poses.keys():
    intruder_pos = np.array(intruder_poses[A])
    plt.plot(intruder_pos[:, 1], intruder_pos[:, 0], 'ko', label=f'Candidate {A}')

print('Plotting own MAV position...')
plt.plot(mav_states[:, 1], mav_states[:, 0], 'ro', label='Own Mav')
plt.show()

print(f'Number of candidates: {len(intruders_dict)}')
print(f"Remaining Candidates for A:\n", intruders_dict.keys())