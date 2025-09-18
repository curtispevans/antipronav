import models.gradient_based_utilities as gbu
import numpy as np
import matplotlib.pyplot as plt
from monte_carlo_simulations import get_simulated_data
from tqdm import tqdm

Ts = 1/30
num_scenarios = 1
num_frames = 500
plotting = False

bearing_std = (2*np.pi/8192)*0.5
pixel_size_std = (2*np.pi/8192)*1

all_bearings, all_pixel_sizes, all_true_distance, all_us, all_mav_states, true_As_vels, own_vels = get_simulated_data(Ts, num_scenarios, num_frames, False)

initial_A = 10
eps = 1e-5
eta = 1e-1
tol = 1e-3

bearings = all_bearings[0]
pixel_sizes = all_pixel_sizes[0]
true_distance = all_true_distance[0]
us = all_us[0]
mav_states = all_mav_states[0]
true_As_vels = true_As_vels[0]
own_vels = own_vels[0]


Q_mpc = 1*np.diag(np.array([np.radians(0.01), 1e-5, np.radians(0.01), 1e-5]))**2
R_mpc = np.diag(np.array([np.radians(0.0001), np.radians(0.0001)]))**2  

Q_tmp = np.eye(2)*0.01**2
Q_nca = np.block([[Ts**5/20*Q_tmp, Ts**4/8*Q_tmp, Ts**3/6*Q_tmp],
                                        [Ts**4/8*Q_tmp, Ts**3/3*Q_tmp, Ts**2/2*Q_tmp],
                                        [Ts**3/6*Q_tmp, Ts**2/2*Q_tmp, Ts*Q_tmp]]) 
R_nca = np.diag(np.array([1e-5, 1e-5]))**2

Ak = initial_A
Ak_eps = Ak + eps

Qs_Rs = [(Q_mpc, R_mpc), (Q_nca, R_nca)]

mus_sigmas = gbu.initialize_filters(bearings[0], pixel_sizes[0], mav_states[0], Ak)
mus_sigmas_eps = gbu.initialize_filters(bearings[0], pixel_sizes[0], mav_states[0], Ak_eps)


measurements = []

for i in tqdm(range(len(bearings) - 1)):
    bearing = bearings[i+1] + np.random.normal(0, bearing_std)
    pixel_size = pixel_sizes[i+1] + np.random.normal(0, pixel_size_std)
    u = us[i+1]
    mav_state = mav_states[i+1]

    measurement = np.array([bearing, pixel_size])
    measurements.append(measurement)

    mus_sigmas, D2_Ak = gbu.update_all_filters(mus_sigmas, Qs_Rs, measurement, Ts, mav_state, u, Ak)

    Ak_eps = Ak + eps

    mus_sigmas_eps, D2_Ak_eps = gbu.update_all_filters(mus_sigmas_eps, Qs_Rs, measurement, Ts, mav_state, u, Ak_eps)
    
    gradient = (D2_Ak_eps - D2_Ak)/eps

    print(Ak, D2_Ak_eps, D2_Ak, gradient)

    window = 1
    diff = np.abs(Ak - (Ak - eta * gradient))
    if i >= 60:# and diff > tol:
        Ak = Ak - eta * gradient
        Ak_eps = Ak + eps
    
        mus_sigmas_init = gbu.initialize_filters(bearings[0], pixel_sizes[0], mav_states[0], Ak)
        mus_sigmas_init_eps = gbu.initialize_filters(bearings[0], pixel_sizes[0], mav_states[0], Ak_eps)
        
        mus_sigmas, D2s_k1 = gbu.update_new_filter(mus_sigmas_init, Qs_Rs, measurements, Ts, mav_states[1:i+2], us[1:i+2], Ak)
        mus_sigmas_eps, D2s_eps = gbu.update_new_filter(mus_sigmas_init_eps, Qs_Rs, measurements, Ts, mav_states[1:i+2], us[1:i+2], Ak_eps)
       

    

