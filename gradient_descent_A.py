import models.gradient_based_utilities as gbu
import numpy as np
import matplotlib.pyplot as plt
from monte_carlo_simulations import get_simulated_data
from tqdm import tqdm

Ts = 1/30
num_scenarios = 1
num_frames = 100
plotting = False

all_bearings, all_pixel_sizes, all_true_distance, all_us, all_mav_states, true_As_vels, own_vels = get_simulated_data(Ts, num_scenarios, num_frames, False)

initial_A = 1
eps = 1e-5
eta = 1e-1

bearings = all_bearings[0]
pixel_sizes = all_pixel_sizes[0]
true_distance = all_true_distance[0]
us = all_us[0]
mav_states = all_mav_states[0]
true_As_vels = true_As_vels[0]
own_vels = own_vels[0]

distance = initial_A/pixel_sizes[0]  # distance in meters
mu_mpc = np.array([0, 0, bearings[0], 1/true_distance[0]])
sigma_mpc = np.diag(np.array([np.radians(0.1), 0.001, np.radians(0.1), 0.01]))**2
Q_mpc = 1*np.diag(np.array([np.radians(0.01), 1e-5, np.radians(0.01), 1e-5]))**2
R_mpc = np.diag(np.array([np.radians(0.0001), np.radians(0.0001)]))**2  

Q_tmp = np.eye(2)*0.01**2
Q_nca = np.block([[Ts**5/20*Q_tmp, Ts**4/8*Q_tmp, Ts**3/6*Q_tmp],
                                        [Ts**4/8*Q_tmp, Ts**3/3*Q_tmp, Ts**2/2*Q_tmp],
                                        [Ts**3/6*Q_tmp, Ts**2/2*Q_tmp, Ts*Q_tmp]]) 
R_nca = np.diag(np.array([1e-5, 1e-5]))**2


# Get the position of the intruder

bearing = bearings[0]
own_pose = mav_states[0][0:2]  # own position
own_heading = mav_states[0][2]  # own heading in radians
los = np.array([np.cos(bearing + own_heading), np.sin(bearing + own_heading)])
int_x = own_pose[0] + distance * los[0]
int_y = own_pose[1] + distance * los[1]

mu_nca = np.array([int_x, int_y, 0, 0, 0, 0])
sigma_nca = np.eye(6)*1**2

mus_sigmas = [(mu_mpc, sigma_mpc), (mu_nca, sigma_nca)]
Qs_Rs = [(Q_mpc, R_mpc), (Q_nca, R_nca)]

distance_A_eps = (initial_A + eps)/pixel_sizes[0]  # distance in meters
int_x_eps = own_pose[0] + distance_A_eps * los[0]
int_y_eps = own_pose[1] + distance_A_eps * los[1]

mu_mpc_eps = np.array([0, 0, bearings[0], 1/distance_A_eps])

mu_nca_eps = np.array([int_x_eps, int_y_eps, 0, 0, 0, 0])
sigma_nca_eps = np.eye(6)*1**2

mus_sigmas_eps = [(mu_mpc_eps, sigma_mpc), (mu_nca_eps, sigma_nca_eps)]

Ak = initial_A

measurements = []
mus_sigmas_list = []
mus_sigmas_eps_list = []

for i in tqdm(range(len(bearings) - 1)):
    bearing = bearings[i+1]
    pixel_size = pixel_sizes[i+1]
    u = us[i+1]
    mav_state = mav_states[i+1]

    measurement = np.array([bearing, pixel_size])
    measurements.append(measurement)

    mus_sigmas_k1, D2_Ak = gbu.update_all_filters(mus_sigmas, Qs_Rs, measurement, Ts, mav_state, u, Ak)
    mus_sigmas_list.append(mus_sigmas_k1)

    Ak_eps = Ak + eps

    mus_sigmas_eps, D2_Ak_eps = gbu.update_all_filters(mus_sigmas_eps, Qs_Rs, measurement, Ts, mav_state, u, Ak_eps)
    mus_sigmas_eps_list.append(mus_sigmas_eps)

    
    gradient = (D2_Ak_eps - D2_Ak)/eps

    print(Ak, D2_Ak_eps, D2_Ak, gradient)

    mus_sigmas = mus_sigmas_k1.copy()

    window = 1
    if i > 30:
        if i % window == 0:
            update_window = 60
            
            Ak = Ak - eta * gradient
            Ak_eps = Ak + eps

            print(Ak, 'step', len(measurements))

            mus_sigmas_init = gbu.initialize_filters(bearings[0], pixel_sizes[0], mav_states[0], Ak)
            mus_sigmas_init_eps = gbu.initialize_filters(bearings[0], pixel_sizes[0], mav_states[0], Ak_eps)
            # print('first Ak')
            mus_sigmas_k1, D2s_k1 = gbu.update_new_filter(mus_sigmas_init, Qs_Rs, measurements, Ts, mav_states[:i+1], us[:i+1], Ak)
            # print('first Ak_eps')
            mus_sigmas_eps, D2s_eps = gbu.update_new_filter(mus_sigmas_init_eps, Qs_Rs, measurements, Ts, mav_states[:i+1], us[:i+1], Ak_eps)
            # measurements.clear()
            print((np.array(D2s_eps) - np.array(D2s_k1))/eps)
            mus_sigmas = mus_sigmas_k1.copy()
            mus_sigmas_eps = mus_sigmas_eps.copy()

    # print((D2_Ak_eps - D2_Ak) / eps)
    
    # print(Ak, gradient)
    # if i < 30:
    #     Ak = Ak
    # else:
    # print(Ak, gradient)

    # Ak = Ak - eta*gradient

    

