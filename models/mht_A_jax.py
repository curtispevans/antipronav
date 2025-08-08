import numpy as np
from jax import numpy as jnp
import jax
from models.ekf_modified_polar_coordinates_knownA_jax import kalman_update as ekf_mpc_update
from models.nearly_constant_accel_kf_jax import kalman_update as kf_nca_update
from jax import config
config.update('jax_enable_x64', True)


def get_position_of_intruder(state, mav):
    distance = 1/state[3]  # inverse distance to distance
    bearing = state[2]  # bearing in radians
    own_pose = mav[0:2]  # own position
    own_heading = mav[2]  # own heading in radians
    los = jnp.array([jnp.cos(bearing + own_heading), jnp.sin(bearing + own_heading)])
    intruder_pos = own_pose + distance * los
    return intruder_pos

def update_all_filters(mu_mpc, sigma_mpc, mu_nca, sigma_nca, Q_mpc, R_mpc, Q_nca, R_nca, bearing_pixel_measurement, Ts, mav, u, A):
    '''
    Update both MPC and NCA filters with the same measurement.
    mu_mpc, sigma_mpc: state and covariance for MPC
    mu_nca, sigma_nca: state and covariance for NCA
    bearing_pixel_measurement: measurement to update with
    Ts: time step
    A: constant used in the MPC measurement model
    '''
    
    # Update MPC filter
    mu_mpc, sigma_mpc = ekf_mpc_update(mu_mpc, sigma_mpc, mav, u, bearing_pixel_measurement, Q_mpc, R_mpc, Ts, A)
    

    # Get the position of the intruder using the MPC state
    intruder_pos = get_position_of_intruder(mu_mpc, mav)
    
    # Update NCA filter
    mu_nca, sigma_nca = kf_nca_update(mu_nca, sigma_nca, intruder_pos, Q_nca, R_nca, Ts)

    # compute mahalanobis distance
    D2 = get_mahalanobis_distance_intruder_state(mu_nca, sigma_nca, intruder_pos, R_nca)

    return mu_mpc, sigma_mpc, mu_nca, sigma_nca, D2

def wrapper_update_all_filters(mu_mpc, sigma_mpc, mu_nca, sigma_nca, Q_mpc, R_mpc, Q_nca, R_nca, measurement, Ts, mav, u, A):
    '''
    Wrapper function to update all filters with the same measurement.
    This is used to compute the Jacobian of the update function.
    '''
    return update_all_filters(mu_mpc, sigma_mpc, mu_nca, sigma_nca, Q_mpc, R_mpc, Q_nca, R_nca, measurement, Ts, mav, u, A)[-1]



def get_mahalanobis_distance_intruder_state(state, sigma, measurement, R):
    C = jnp.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])
    hx = C @ state
    innovation = measurement - hx
    
    S = C @ sigma @ C.T + R
    
    D2 = innovation.T @ jnp.linalg.inv(S) @ innovation
    # print(D2)
    return D2

