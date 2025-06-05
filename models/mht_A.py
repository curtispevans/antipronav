import numpy as np
from scipy.stats import halfnorm
from models.nearly_constant_accel_kf import kalman_update as nearly_constant_accel_kf_update
from models.ekf_modified_polar_coordinates_knownA import kalman_update as ekf_modified_polar_knownA_update


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


        
    
    return filtered_dict

def get_g_force_probability(g_force):
    pass