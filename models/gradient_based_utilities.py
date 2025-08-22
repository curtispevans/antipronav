import numpy as np
from models.ekf_modified_polar_coordinates_knownA import kalman_update as ekf_mpc_update
from models.nearly_constant_accel_kf import kalman_update as kf_nca_update

def get_position_of_intruder(state, mav):
    distance = 1/state[3]
    bearing = state[2]
    own_pose = mav[0:2]  # own position
    own_heading = mav[2]  # own heading in radians
    los = np.array([np.cos(bearing + own_heading), np.sin(bearing + own_heading)])
    intruder_pose = own_pose + distance*los
    return intruder_pose

def get_mahalanobis_distance_intruder_state(state, sigma, measurement, R):
    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])
    hx = C @ state
    innovation = measurement - hx
    
    S = C @ sigma @ C.T + R
    
    D2 = innovation.T @ np.linalg.inv(S) @ innovation
    
    return D2

def update_all_filters(mus_sigmas, Qs_Rs, measurement, Ts, mav, u, A):
    '''
    Updates the modified polar coordinate filter and then the nearly constant acceleration filter

    Parameters:
        mus_sigmas: list of tuples (mu, sigma) for each filter
        Qs_Rs: list of tuples (Q, R) for each filter
        measurement: [bearing, pixel_size]
        Ts: list of time steps for each filter
        mav: current state of the MAV
        u: control input
        A: state transition matrix

    Returns:
        mus_sigmas_updated: list of tuples (mu, sigma) for each filter after update
        D2: Mahalanobis distance for the nearly constant acceleration filter
    '''

    mu_mpc, sigma_mpc = mus_sigmas[0]
    mu_nca, sigma_nca = mus_sigmas[1]

    Q_mpc, R_mpc = Qs_Rs[0]
    Q_nca, R_nca = Qs_Rs[1]


    # Update the modified polar coordinate filter
    mu_mpc, sigma_mpc = ekf_mpc_update(mu_mpc, sigma_mpc, mav, u, measurement, Q_mpc, R_mpc, Ts, A)

    measurement_pose = get_position_of_intruder(mu_mpc, mav)

    # Update the nearly constant acceleration filter
    mu_nca, sigma_nca = kf_nca_update(mu_nca, sigma_nca, measurement_pose, Q_nca, R_nca, Ts)


    # Compute the Mahalanobis distance for the nearly constant acceleration filter
    D2 = get_mahalanobis_distance_intruder_state(mu_nca, sigma_nca, measurement_pose, R_nca)

    mus_sigmas_updated = [(mu_mpc, sigma_mpc), (mu_nca, sigma_nca)]

    return mus_sigmas_updated, D2