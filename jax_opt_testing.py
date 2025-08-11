import models.mht_A_jax as mht
from monte_carlo_simulations import get_simulated_data
import numpy as np
from jax import jacfwd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import config
config.update('jax_enable_x64', True)

Ts = 1/30
num_scenarios = 1
num_frames = 50
plotting = False

all_bearings, all_pixel_sizes, all_true_distance, all_us, all_mav_states, true_As_vels, own_vels = get_simulated_data(Ts, num_scenarios, num_frames, False)

bearings = all_bearings[0]
pixel_sizes = all_pixel_sizes[0]
true_distance = all_true_distance[0]
us = all_us[0]
mav_states = all_mav_states[0]

initial_A = 15.0

mu_mpc = np.array([0, 0, bearings[0], 1/true_distance[0]])
sigma_mpc = np.diag(np.array([np.radians(0.1), 0.001, np.radians(0.1), 0.01]))**2
Q_mpc = np.diag(np.array([np.radians(0.01), 1e-5, np.radians(0.01), 1e-5]))**2
R_mpc = np.diag(np.array([np.radians(0.0001), 0.0001]))**2

Q_tmp = np.eye(2)*0.01**2
Q_nca = np.block([[Ts**5/20*Q_tmp, Ts**4/8*Q_tmp, Ts**3/6*Q_tmp],
                                    [Ts**4/8*Q_tmp, Ts**3/3*Q_tmp, Ts**2/2*Q_tmp],
                                    [Ts**3/6*Q_tmp, Ts**2/2*Q_tmp, Ts*Q_tmp]]) 
R_nca = np.diag(np.array([1e-5, 1e-5]))**2

distance = initial_A/pixel_sizes[0]  # distance in meters
bearing = bearings[0]
own_pose = mav_states[0][0:2]  # own position
own_heading = mav_states[0][2]  # own heading in radians
los = np.array([np.cos(bearing + own_heading), np.sin(bearing + own_heading)])
int_x = own_pose[0] + distance * los[0]
int_y = own_pose[1] + distance * los[1]

mu_nca = np.array([int_x, int_y, 0, 0, 0, 0])
sigma_nca = np.eye(6)*1**2

initial_states = (mu_mpc, sigma_mpc, mu_nca, sigma_nca)

measurements = jnp.array([jnp.array([bearing, pixel_size]) for bearing, pixel_size in zip(bearings[1:], pixel_sizes[1:])])
us = jnp.array(us[1:])
mav_states = jnp.array(mav_states[1:])

for i in range(30):
    mu_mpc, sigma_mpc, mu_nca, sigma_nca, D2 = mht.update_all_filters(
        mu_mpc, sigma_mpc, mu_nca, sigma_nca,
        Q_mpc, R_mpc, Q_nca, R_nca,
        measurements[i], Ts, mav_states[i], us[i], initial_A
    )


initial_states = (mu_mpc, sigma_mpc, mu_nca, sigma_nca)


A_opt, history = mht.optimize_A(
    initial_states,
    measurements[30:],
    mav_states[30:],
    us[30:], 
    Ts,
    Q_mpc,
    R_mpc,
    Q_nca,
    R_nca,
    num_frames - 30,
    A_init=initial_A,
    learning_rate=0.01,
    num_iters=10,
)

print(f'Optimized A: {A_opt}')