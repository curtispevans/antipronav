import models.mht_A as mht
import numpy as np
import matplotlib.pyplot as plt
from monte_carlo_simulations import get_simulated_data
from tqdm import tqdm

Ts = 1/30
num_scenarios = 5
num_frames = 300

all_bearings, all_pixel_sizes, all_true_distance, all_us, all_mav_states, true_As, own_vels = get_simulated_data(Ts, num_scenarios, num_frames, True)
min_A = 5
max_A = 40

num_scenarios = len(all_bearings)  # Number of scenarios is the number of bearings minus one
predicted_As = []
for i in tqdm(range(num_scenarios)):
    bearings = all_bearings[i]
    mav_states = all_mav_states[i]
    pixel_sizes = all_pixel_sizes[i]
    true_distance = all_true_distance[i]
    us = all_us[i]
    true_A = true_As[i]


    mu_inverse_distance = np.array([0, 0, bearings[0], 1/true_distance[0]])
    sigma_inverse_distance = np.diag(np.array([np.radians(0.1), 0.001, np.radians(0.1), 0.01]))**2
    Q_inverse_distance = np.diag(np.array([np.radians(0.01), 1e-5, np.radians(0.01), 1e-5]))**2
    R_inverse_distance = np.diag(np.array([np.radians(0.0001), 0.0001]))**2   

    Q_tmp = np.eye(2)*0.01**2
    Q_nearly_constant_accel = np.block([[Ts**5/20*Q_tmp, Ts**4/8*Q_tmp, Ts**3/6*Q_tmp],
                                        [Ts**4/8*Q_tmp, Ts**3/3*Q_tmp, Ts**2/2*Q_tmp],
                                        [Ts**3/6*Q_tmp, Ts**2/2*Q_tmp, Ts*Q_tmp]]) 
    R_nearly_constant_accel = np.diag(np.array([1e-5, 1e-5]))**2

    intruders_dict = {}

    for k in range(min_A, max_A):
        # Get the position of the intruder
        distance = k/pixel_sizes[0]  # distance in meters
        bearing = bearings[k]
        own_pose = mav_states[k][0:2]  # own position
        own_heading = mav_states[k][2]  # own heading in radians
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

    intruder_poses = {i:[] for i in range(min_A, max_A)}
    inv_distances = {i:[] for i in range(min_A, max_A)}

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

        # Filter candidates
        if j > 30:
            intruders_dict = mht.filter_pose_measurement_probabilistic(intruders_dict, own_mav, R_nearly_constant_accel, 1)
        
        for A in intruders_dict.keys():
            # intruder_state = intruders_dict[A][2]
            intruder_state = intruders_dict[A][2][:2]
            intruder_poses[A].append(intruder_state[0:2])
            inv_distances[A].append(intruders_dict[A][0][3])

    # print(f'Number of candidates: {len(intruders_dict)}')
    # print(f"Remaining Candidates for A:\n", intruders_dict.keys())
    
    highest_counter_list = []
    for A in intruders_dict.keys():
        highest_counter_list.append(intruders_dict[A][4])
        # print(A, intruders_dict[A][4])
    sorted_idx = np.argsort(np.array(highest_counter_list))[::-1]
    ordered_candidates = np.array(list(intruders_dict.keys()))[sorted_idx]
    predicted_A = ordered_candidates[0]
    predicted_As.append(predicted_A)
    if np.abs(true_A - predicted_A) > 5:
        tqdm.write(f"Own Vel {own_vels[i]}")
        tqdm.write(f"True A {true_A}")
        tqdm.write(f"Highest counters: ")
        tqdm.write(f"{ordered_candidates[:3]}\n")

    # print(f"intruder poses: {intruder_poses}")

    # print('Plotting candidates...')
    # # Plotting the intruder positions
    # plt.figure(1)
    # for A in intruder_poses.keys():
    #     # print(intruder_poses[A])
    #     intruder_pos = np.array(intruder_poses[A])
    #     plt.plot(intruder_pos[:, 1], intruder_pos[:, 0], 'ko', label=f'Candidate {A}')

    # print('Plotting own MAV position...')
    # plt.plot(mav_states[:, 1], mav_states[:, 0], 'ro', label='Own Mav')


    # plt.figure(2)
    # for A in inv_distances.keys():
    #     # print(A)
    #     plt.plot(np.abs(pixel_sizes[1:] - A*np.array(inv_distances[A])), label=f'Candidate {A}')
    # # plt.plot(1/true_distance, label='True Inverse Distance', linestyle='--')
    # plt.xlabel('Time Step')
    # plt.ylabel('True Pixel Size - Estimated Pixel Size')

    # plt.legend()
    # plt.tight_layout()

    # plt.figure(3)
    # for A in inv_distances.keys():
    #     plt.plot(inv_distances[A], label=f'Candidate {A}')
    # plt.plot(1/true_distance[1:], label='True Inverse Distance', linestyle='--')
    # plt.xlabel('Time Step')
    # plt.ylabel('Inverse Distance (m)')
    # plt.legend()
    # plt.tight_layout()

    # plt.figure(4)
    # for A in inv_distances.keys():
    #     plt.plot(A*np.array(inv_distances[A]), label=f'Candidate {A}')
    # plt.plot(pixel_sizes[1:], label='True Pixel Size', linestyle='--')
    # plt.plot(10/true_distance[1:], label='True Pixel Size (10m)', linestyle='--')
    # plt.title('Estimated Pixel Size vs True Pixel Size')
    # plt.xlabel('Time Step')
    # plt.ylabel('Estimated Pixel Size (m)')
    # plt.legend()
    # plt.tight_layout()


    # plt.show()

plt.figure(1)
plt.plot(np.abs(true_As - np.array(predicted_As)))
plt.xlabel('Simulation Iteration')
plt.ylabel('Error of true A')
plt.title("Error Plot")
plt.tight_layout()
plt.show()