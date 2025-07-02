import numpy as np
from scipy.stats import halfnorm, norm, multivariate_normal
from models.nearly_constant_accel_kf import kalman_update as nearly_constant_accel_kf_update
from models.ekf_modified_polar_coordinates_knownA import kalman_update as ekf_modified_polar_knownA_update
from models.ekf_modified_polar_coordinates_knownA import measurement_model as ekf_modified_polar_measurement_model
from models.ekf_modified_polar_coordinates_knownA import wrap

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


def propagate_candidates_intruder_pos(intruders_dict, mav, Ts, Q, R):
    '''
    intruders_dict: {A : [candidate_state_A, candidate_sigma_A, intruder_state, intruder_sigma]}
    '''

    for A in intruders_dict.keys():
        state = intruders_dict[A][0]
        intruder_state = intruders_dict[A][2]
        intruder_sigma = intruders_dict[A][3]
        # Get the position of the intruder
        measurement_pos = get_position_of_intruder(state, mav)

        # update the intruder state with the measurement
        intruder_state, sigma = nearly_constant_accel_kf_update(intruder_state, intruder_sigma, measurement_pos, Q, R, Ts)

        intruders_dict[A][2] = intruder_state
        intruders_dict[A][3] = sigma

    return intruders_dict

def propagate_candidates_inverse_distance(intruders_dict, mav, u, measurement, Ts, Q, R):
    '''
    intruders_dict: {A : [candidate_state_A, candidate_sigma_A, intruder_state, intruder_sigma]}
    '''
    for A in intruders_dict.keys():
        state = intruders_dict[A][0]
        sigma = intruders_dict[A][1]

        # Update the state using the EKF
        state, sigma = ekf_modified_polar_knownA_update(state, sigma, mav, u, measurement, Q, R, Ts, A)

        intruders_dict[A][0] = state
        intruders_dict[A][1] = sigma

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

def filter_candidates_probabilistic(intruders_dict, prob_threshold=0.5):
    '''
    Filter candidates based on a probabilistic threshold.
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

        # Calculate the probability of the velocity and g-force
        mean_velocity = velocity_mean_function(A)
        if A < 8:
            std = 20**0.5
        elif 8 <= A < 15:
            std = 150**0.5
        elif 15 <= A < 21:
            std = 75**0.5
        else:
            std = 80**0.5
        velocity_prob = norm.logpdf(velocity, loc=mean_velocity, scale=std)
        g_force_prob = get_g_force_probability(g_force)
        # Combine the probabilities
        combined_prob = velocity_prob + g_force_prob
        # print(A, combined_prob, velocity_prob, g_force_prob, mean_velocity)
        if combined_prob > prob_threshold:
            filtered_dict[A] = [state, sigma, intruder_state, intruder_sigma]
    return filtered_dict


def filter_state_measurement_probabilistic(intruders_dict, measurement, R, mahalanobis_dist = 1e-5):
    '''
    Filter candidstes based on p(measurement | state) > prob_threshold, NOT FULL STATE
    '''

    filtered_dict = {}
    for A in intruders_dict.keys():
        state = intruders_dict[A][0]
        sigma = intruders_dict[A][1]
        intruder_state = intruders_dict[A][2]
        intruder_sigma = intruders_dict[A][3]
        # Calculate the probability of the measurement given the state
        log_prob = get_measurement_log_probability_not_full_state(state, sigma, measurement, R, A)
        D2 = get_mahalanobis_distance(state, sigma, measurement, R, A)
        pixel_mh = get_mahalanobis_distance_pixel_size(state, sigma, measurement, R, A)
        print(A) #, D2, sigma[-1,-1], pixel_mh)

        if D2 < mahalanobis_dist:
            filtered_dict[A] = [state, sigma, intruder_state, intruder_sigma]

    return filtered_dict


def get_g_force_probability(g_force):
    # Assuming g-force follows a half-normal distribution
    return halfnorm.logpdf(g_force, scale=0.1**0.5)  # scale can be adjusted based on expected g-force values


def get_measurement_log_probability_not_full_state(state, sigma, measurement, R, A):
    '''
    Calculate the probability of the measurement given the state using 
    the kalman filter measurement model.
    '''
    hx = ekf_modified_polar_measurement_model(state, A)
    innovation_mean = measurement - hx
    innovation_mean[0] = wrap(innovation_mean[0]) 

    H = np.array([[0, 0, 1, 0],
                  [0, 0, 0, A]])
    S = H @ sigma @ H.T + R

    log_prob = multivariate_normal.logpdf(measurement, mean=hx, cov=S)

    return log_prob

def get_mahalanobis_distance(state, sigma, measurement, R, A):
    '''
    Calculate the probability of the measurement given the state using 
    the kalman filter measurement model.
    '''
    hx = ekf_modified_polar_measurement_model(state, A)
    innovation_mean = measurement - hx
    innovation_mean[0] = wrap(innovation_mean[0]) 
    # print(innovation_mean)

    H = np.array([[0, 0, 1, 0],
                  [0, 0, 0, A]])
    S = H @ sigma @ H.T + R

    D2 = innovation_mean.T @ np.linalg.inv(S) @ innovation_mean

    return D2 + np.log(np.linalg.det(S))  # Add log determinant for numerical stability

def get_mahalanobis_distance_pixel_size(state, sigma, measurement, R, A):
    '''
    Calculate the probability of the measurement given the state using 
    the kalman filter measurement model.
    '''
    hx = ekf_modified_polar_measurement_model(state, A)
    innovation_mean = (measurement[1] - hx[1]).reshape(1,1)

    H = np.array([[0, 0, 0, A]])
    S = H @ sigma @ H.T + R[-1,-1]

    D2 = innovation_mean.T @ np.linalg.inv(S) @ innovation_mean

    return D2





