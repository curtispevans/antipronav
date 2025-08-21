import models.mht_A as mht
import numpy as np
import matplotlib.pyplot as plt
from monte_carlo_simulations import get_simulated_data
from tqdm import tqdm

Ts = 1/30
num_scenarios = 1
num_frames = 500
plotting = False

all_bearings, all_pixel_sizes, all_true_distance, all_us, all_mav_states, true_As_vels, own_vels = get_simulated_data(Ts, num_scenarios, num_frames, False)
min_A = 5
max_A = 40

range_A = np.linspace(min_A, max_A, 5)

num_scenarios = len(all_bearings)  # Number of scenarios is the number of bearings
poses_from_inverse_distance = []
poses_from_nca = []

gradients = np.load('gradients.npy')

for i in tqdm(range(num_scenarios)):
    bearings = all_bearings[i]
    mav_states = all_mav_states[i]
    pixel_sizes = all_pixel_sizes[i]
    true_distance = all_true_distance[i]
    us = all_us[i]
    true_A, intruder_vel = true_As_vels[i]

    est_intruder_poses = []

    mu_inverse_distance = np.array([0, 0, bearings[0], 1/true_distance[0]])
    sigma_inverse_distance = np.diag(np.array([np.radians(0.1), 0.001, np.radians(0.1), 0.01]))**2
    Q_inverse_distance = 1*np.diag(np.array([np.radians(0.00001), 1e-5, np.radians(0.00001), 1e-5]))**2
    # R_inverse_distance = np.diag(np.array([np.radians(0.04), np.radians(0.12)]))**2
    R_inverse_distance = np.diag(np.array([np.radians(0.001), np.radians(0.001)]))**2  

    

    Q_tmp = np.eye(2)*0.001**2
    Q_nearly_constant_accel = np.block([[Ts**5/20*Q_tmp, Ts**4/8*Q_tmp, Ts**3/6*Q_tmp],
                                        [Ts**4/8*Q_tmp, Ts**3/3*Q_tmp, Ts**2/2*Q_tmp],
                                        [Ts**3/6*Q_tmp, Ts**2/2*Q_tmp, Ts*Q_tmp]]) 
    R_nearly_constant_accel = np.diag(np.array([1, 1]))**2

    intruders_dict = {'mah_dist_sorted':[]}


    for k in range_A:
        # Get the position of the intruder
        distance = k/pixel_sizes[0]  # distance in meters
        bearing = bearings[0]
        own_pose = mav_states[0][0:2]  # own position
        own_heading = mav_states[0][2]  # own heading in radians
        los = np.array([np.cos(bearing + own_heading), np.sin(bearing + own_heading)])
        int_x = own_pose[0] + distance * los[0]
        int_y = own_pose[1] + distance * los[1]
        # vel_x = relative_velocities[i][0] + own_velocities[i][0]
        # vel_y = relative_velocities[i][1] + own_velocities[i][1]

        mu_nearly_constant_accel = np.array([int_x, int_y, 0, 0, 0, 0])
        sigma_nearly_constant_accel = np.eye(6)*1**2
        filter_counter = 0
        intruders_dict[k] = [mu_inverse_distance.copy(), sigma_inverse_distance.copy(), mu_nearly_constant_accel.copy(), sigma_nearly_constant_accel.copy(), filter_counter]

    full_inverse_distance = []
    partial_inverse_distance = []

    intruder_poses = {i:[] for i in range_A}
    inv_distances = {i:[] for i in range_A}

    for j in range(len(bearings) - 1):
        bearing = bearings[j+1] 
        pixel_size = pixel_sizes[j+1]
        u = us[j+1]
        own_mav = mav_states[j+1]

        measurement = np.array([bearing, pixel_size])
        
        # Propagate candidates for inverse distance
        intruders_dict = mht.propagate_candidates_inverse_distance(intruders_dict, own_mav, u, measurement, Ts, Q_inverse_distance, R_inverse_distance)
        
        # Propagate candidates for nearly constant acceleration
        intruders_dict = mht.propagate_candidates_intruder_pos(intruders_dict, own_mav, Ts, Q_nearly_constant_accel, R_nearly_constant_accel)

        plt.plot(own_mav[1], own_mav[0], 'bo', markersize=2)
        for k in range_A:
            mu_inverse_distance, sigma_inverse_distance, mu_nearly_constant_accel, sigma_nearly_constant_accel, filter_counter = intruders_dict[k]
            pose_inverse_dist = mht.get_position_of_intruder(mu_inverse_distance, own_mav)
            plt.plot(pose_inverse_dist[1], pose_inverse_dist[0], 'ro', markersize=2)
            plt.plot(mu_nearly_constant_accel[1], mu_nearly_constant_accel[0], 'go', markersize=2)

        plt.plot(own_mav[1] + true_distance[j+1] * np.sin(bearing + own_mav[2]), own_mav[0] + true_distance[j+1] * np.cos(bearing + own_mav[2]), 'mo', markersize=2)
        plt.pause(0.01)

plt.show()