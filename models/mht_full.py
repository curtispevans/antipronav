import numpy as np
from scipy.stats import halfnorm, norm, multivariate_normal
# from models.full_state_ekf import kalman_update as full_state_ekf_update
# from models.full_state_ekf import measurement_model as full_state_measurement_model
# from models.full_state_ekf import jacobian_measurement_model as full_state_jacobian_measurement_model
# from models.full_state_ekf import wrap
from models.full_state_ekf_discrete import kalman_update as full_state_ekf_update
from models.full_state_ekf_discrete import measurement_model as full_state_measurement_model
from models.full_state_ekf_discrete import jacobian_measurement_model as full_state_jacobian_measurement_model
from models.full_state_ekf_discrete import wrap

def velocity_mean_function(wingspan):
    beta = np.load('data/regression_coefficients.npy')
    if wingspan < 8:
        return 25
    else:
        return beta[0] * wingspan**3 + beta[1] * wingspan**2 + beta[2] * wingspan + beta[3]


def get_position_of_intruder(state, mav):
    distance = 1/state[3] # inverse distance to distance
    bearing = state[2] # bearing in radians
    own_pose = mav[0:2] # own position
    own_heading = mav[2] # own heading in radians
    los = np.array([np.cos(bearing + own_heading), np.sin(bearing + own_heading)])
    intruder_pos = own_pose + distance * los
    return intruder_pos


def propagate_full_state(intruders_dict, mav, u, measurement, Ts, Q, R):
    '''
    intruders_dict: {A : [full_state_A, full_sigma_A]}
    '''
    for A in intruders_dict.keys():
        state = intruders_dict[A][0]
        sigma = intruders_dict[A][1]

        # Update the state using the EKF
        state, sigma = full_state_ekf_update(state, sigma, mav, u, measurement, Q , R, Ts, A)

        intruders_dict[A][0] = state
        intruders_dict[A][1] = sigma

        # print(A, np.linalg.norm(state[6:8]), np.linalg.norm(state[8:])/9.81)

    return intruders_dict


def filter_candidates(intruders_dict, vel_threshold=100, g_force_threshold=1):
    '''
    Filter candidates based on velocity and g-force thresholds.
    '''
    filtered_dict = {}
    for A in intruders_dict.keys():
        state = intruders_dict[A][0]
        sigma = intruders_dict[A][1]
        intruder_state = intruders_dict[A][2]
        intruder_sigma = intruders_dict[A][3]

        # Calculate the velocity of the intruder
        velocity = np.linalg.norm(intruder_state[2:4])
        # Calculate the g-force
        g_force = np.linalg.norm(intruder_state[4:]) / 9.81
        if velocity < vel_threshold and g_force < g_force_threshold:
            filtered_dict[A] = [state, sigma, intruder_state, intruder_sigma]
    return filtered_dict



def filter_full_state_probabilistic(intruders_dict, own_mav, measurement, R_full, m_dist_thres=0):
    '''
    Filter candidates based on p(measurement | state) > prob_threshold.
    '''
    filtered_dict = {}
    for A in intruders_dict.keys():
        state = intruders_dict[A][0]
        sigma = intruders_dict[A][1]

        # Calculate the probability of the measurement given the state
        log_prob = get_measurement_log_probability(state, own_mav, sigma, measurement, R_full, A)
        m_dist = get_mahalanobis_distance(state, sigma, own_mav, measurement, R_full, A)
        print(A, log_prob, np.linalg.norm(state[6:8]), m_dist)

        if m_dist < m_dist_thres:
            filtered_dict[A] = [state, sigma]

    return filtered_dict


def get_g_force_probability(g_force):
    # Assuming g-force follows a half-normal distribution
    return halfnorm.logpdf(g_force, scale=0.1**0.5)  # scale can be adjusted based on expected g-force values

def get_measurement_log_probability(state, own_mav, sigma, measurement, R_full, A):
    '''
    Calculate the probability of the measurement given the state using 
    the kalman filter measurement model.
    '''
    hx = full_state_measurement_model(state, own_mav, A)
    innovation_mean = measurement - hx
    innovation_mean[0] = wrap(innovation_mean[0]) 

    H = full_state_jacobian_measurement_model(full_state_measurement_model, state, own_mav, A)
    S = H @ sigma @ H.T + R_full

    log_prob = multivariate_normal.logpdf(measurement, mean=hx, cov=S)

    return log_prob

def get_mahalanobis_distance(state, sigma, own_mav, measurement, R, A):
    hx = full_state_measurement_model(state, own_mav, A)
    innovation_mean = measurement - hx
    innovation_mean[0] = wrap(innovation_mean[0])

    H = full_state_jacobian_measurement_model(full_state_measurement_model, state, own_mav, A)
    S = H @ sigma @ H.T + R

    return innovation_mean.T @ np.linalg.inv(S) @ innovation_mean




