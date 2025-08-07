import models.mht_A_jax as mht
from monte_carlo_simulations import get_simulated_data
import numpy as np
from jax import jacfwd
import matplotlib.pyplot as plt
from jax import config
config.update('jax_enable_x64', True)

Ts = 1/30
num_scenarios = 1
num_frames = 300
plotting = False
A = 16.42857142857143
# A = 5

all_bearings, all_pixel_sizes, all_true_distance, all_us, all_mav_states, true_As_vels, own_vels = get_simulated_data(Ts, num_scenarios, num_frames, False)

bearings = all_bearings[0]
pixel_sizes = all_pixel_sizes[0]
true_distance = all_true_distance[0]
us = all_us[0]
mav_states = all_mav_states[0]


mu_mpc = np.array([0, 0, bearings[0], 1/true_distance[0]])
sigma_mpc = np.diag(np.array([np.radians(0.1), 0.001, np.radians(0.1), 0.01]))**2
Q_mpc = np.diag(np.array([np.radians(0.01), 1e-5, np.radians(0.01), 1e-5]))**2
R_mpc = np.diag(np.array([np.radians(0.0001), 0.0001]))**2

Q_tmp = np.eye(2)*0.01**2
Q_nca = np.block([[Ts**5/20*Q_tmp, Ts**4/8*Q_tmp, Ts**3/6*Q_tmp],
                                    [Ts**4/8*Q_tmp, Ts**3/3*Q_tmp, Ts**2/2*Q_tmp],
                                    [Ts**3/6*Q_tmp, Ts**2/2*Q_tmp, Ts*Q_tmp]]) 
R_nca = np.diag(np.array([1e-5, 1e-5]))**2

distance = A/pixel_sizes[0]  # distance in meters
bearing = bearings[0]
own_pose = mav_states[0][0:2]  # own position
own_heading = mav_states[0][2]  # own heading in radians
los = np.array([np.cos(bearing + own_heading), np.sin(bearing + own_heading)])
int_x = own_pose[0] + distance * los[0]
int_y = own_pose[1] + distance * los[1]

mu_nca = np.array([int_x, int_y, 0, 0, 0, 0])
sigma_nca = np.eye(6)*1**2

J = jacfwd(mht.wrapper_update_all_filters, argnums=12)

gradients = []

for j in range(len(bearings)-1):
    bearing = bearings[j+1]
    pixel_size = pixel_sizes[j+1]
    u = us[j+1]
    own_mav = mav_states[j+1]

    measurement = np.array([bearing, pixel_size])

    mu_mpc, sigma_mpc, mu_nca, sigma_nca, D2 = mht.update_all_filters(
        mu_mpc, sigma_mpc, mu_nca, sigma_nca, Q_mpc, R_mpc, Q_nca, R_nca, measurement, Ts, own_mav, u, A)
    
    jacobian_A = J(mu_mpc, sigma_mpc, mu_nca, sigma_nca, Q_mpc, R_mpc, Q_nca, R_nca, measurement, Ts, own_mav, u, A)

    # plt.plot(own_mav[1], own_mav[0], 'ro', label='Ownship Position')
    # plt.plot(mu_nca[1], mu_nca[0], 'bo', label='NCA Estimate Position')
    # true_pose = own_mav[:2] + np.array([np.cos(bearing + own_mav[2]), np.sin(bearing + own_mav[2])]) * true_distance[j+1]
    # plt.plot(true_pose[1], true_pose[0], 'go', label='True Intruder Position')
    # # plt.legend()
    # plt.title(f'Frame {j+1} - Intruder Position Estimate')
    # plt.xlabel('X Position (m)')
    # plt.ylabel('Y Position (m)')
    # plt.pause(0.01)

    
    gradients.append(jacobian_A)
    # print(jacobian_A)

# plt.show()
np.save('gradients.npy', np.array(gradients))


    