import numpy as np
from scipy.stats import halfnorm, norm, multivariate_normal
from models.nearly_constant_accel_kf import kalman_update as nearly_constant_accel_kf_update
from models.ekf_modified_polar_coordinates_knownA import kalman_update as ekf_modified_polar_knownA_update
from models.ekf_modified_polar_coordinates_knownA import measurement_model as ekf_modified_polar_measurement_model
from models.ekf_modified_polar_coordinates_unknownA import kalman_update as ekf_modified_polar_unknownA_update
from models.ekf_modified_polar_coordinates_knownA import wrap
import matplotlib.pyplot as plt

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

    for A in list(intruders_dict.keys())[1:]:
        state = intruders_dict[A][0]
        intruder_state = intruders_dict[A][2]
        intruder_sigma = intruders_dict[A][3]
        # Get the position of the intruder
        measurement_pos = get_position_of_intruder(state, mav)
        # print(f'{np.round(A, 2)} vel: {np.round(np.linalg.norm(intruder_state[2:4]), 2)} m/s, g-force: {np.round(np.linalg.norm(intruder_state[4:])/9.81, 2)} g')
        # update the intruder state with the measurement
        intruder_state, sigma = nearly_constant_accel_kf_update(intruder_state, intruder_sigma, measurement_pos, Q, R, Ts)

        intruders_dict[A][2] = intruder_state
        intruders_dict[A][3] = sigma

    return intruders_dict

def propagate_candidates_inverse_distance(intruders_dict, mav, u, measurement, Ts, Q, R):
    '''
    intruders_dict: {A : [candidate_state_A, candidate_sigma_A, intruder_state, intruder_sigma]}
    '''
    for A in list(intruders_dict.keys())[1:]:
        state = intruders_dict[A][0]
        sigma = intruders_dict[A][1]

        # Update the state using the EKF
        R_tmp = R.copy()
        # R_tmp[1,1] = A * R[1,1]
        state, sigma = ekf_modified_polar_knownA_update(state, sigma, mav, u, measurement, Q, R_tmp, Ts, A)

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
        # print(A) #, D2, sigma[-1,-1], pixel_mh)

        if D2 < mahalanobis_dist:
            filtered_dict[A] = [state, sigma, intruder_state, intruder_sigma]

    return filtered_dict

def filter_pose_measurement_probabilistic(intruders_dict, mav, R_inv, R_nca, mahalanobis_dist, gradient, measurement=None):
    filtered_dict = {}
    mah_dists = []
    for A in list(intruders_dict.keys())[1:]:
        state = intruders_dict[A][0]
        sigma = intruders_dict[A][1]
        intruder_state = intruders_dict[A][2]
        intruder_sigma = intruders_dict[A][3]

        measurement_pos = get_position_of_intruder(state, mav)

        print_inno = False
        # if 16 < A and A < 17:
        #     x = np.linspace(5, 40, 100)
        #     y = gradient*(x - A) + D2

        # print(A)
        D2 = get_mahalanobis_distance_intruder_state(intruder_state, intruder_sigma, measurement_pos, R_nca, print_inno)
        D2 += get_mahalanobis_distance_pixel_size(state, sigma, measurement, R_inv, A)
        # D2 = get_mahalanobis_distance_intruder_state_normalized(intruder_state, intruder_sigma, measurement_pos, R_nca, state[-1])
        # print(A, '\n', np.round(intruder_sigma, 5))
        mah_dists.append(D2)

        # if 10 <= A <= 12:
        #     print(A, D2)

        # if D2 < mahalanobis_dist:
        #     filtered_dict[A] = [state, sigma, intruder_state, intruder_sigma]
    plt.figure(-2)
    plt.plot(list(intruders_dict.keys())[1:], mah_dists, 'b-', label='Mahalanobis distances', alpha=0.05)
    # plt.plot(x, y, 'g-', alpha=0.5)
    plt.xlabel('Candidate A')
    plt.ylabel('Mahalanobis distance')
    plt.title('Mahalanobis distances of candidates')
    # plt.xlim(4, 40)
    # plt.ylim(-0.01, 0.5)
    plt.pause(0.01)       
    # plt.show()


    sorted_As = np.argsort(np.array(mah_dists))
    
    lowest_As = np.array(list(intruders_dict.keys())[1:])[sorted_As]
    intruders_dict['mah_dist_sorted'] = lowest_As

    for A in lowest_As[:1]:
        intruders_dict[A][4] += 1

    return intruders_dict
        
def get_best_estimated_intruder_pose(intruders_dict):
    highest = -1
    best_state = None
    for A in list(intruders_dict.keys())[1:]:
        if intruders_dict[A][4] > highest:
            highest = intruders_dict[A][4]
            best_state = intruders_dict[A][2][:2]
    # print('mah_dist_sorted', intruders_dict['mah_dist_sorted'])
    # best_A = intruders_dict['mah_dist_sorted'][0]
    # best_state = intruders_dict[best_A][2][:2]
    return best_state

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
    hx = np.array([state[2], A*state[3]])
    innovation_mean = measurement - hx
    innovation_mean[0] = wrap(innovation_mean[0]) 
    print(innovation_mean)

    H = np.array([[0, 0, 1, 0],
                  [0, 0, 0, A]])
    S = H @ sigma @ H.T + R

    D2 = innovation_mean.T @ np.linalg.inv(S) @ innovation_mean

    return D2 #+ np.log(np.linalg.det(S))  # Add log determinant for numerical stability

def get_mahalanobis_distance_pixel_size(state, sigma, measurement, R, A):
    '''
    Calculate the probability of the measurement given the state using 
    the kalman filter measurement model.
    '''
    hx = np.array([A*state[3]])
    innovation_mean = (measurement[1] - hx).reshape(1,1)

    H = np.array([[0, 0, 0, A]])
    S = H @ sigma @ H.T + R[-1,-1]

    D2 = innovation_mean.T @ np.linalg.inv(S) @ innovation_mean
    # print(innovation_mean[0,0], D2[0,0], S)
    return D2[0,0]

def get_mahalanobis_distance_intruder_state(state, sigma, measurement, R, print_inno):
    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])
    hx = C @ state
    innovation = measurement - hx
    
    S = C @ sigma @ C.T + R
    D2 = innovation.T @ np.linalg.inv(S) @ innovation

    # if print_inno:
    #     print(D2)
    return D2

def propagate_mpc_unknownA(mu, sigma, own_mav, u, measurement, Q, R, Ts):
    '''
    Propagate the MPC filter with unknown A.
    '''
    mu, sigma = ekf_modified_polar_unknownA_update(mu, sigma, own_mav, u, measurement, Q, R, Ts)

    return mu, sigma

def get_mu_sigma_from_mosted_voted_A(intruders_dict):
    '''
    Get the mu and sigma from the most voted A.
    '''
    most_voted_A = max(list(intruders_dict.keys())[1:], key=lambda k: intruders_dict[k][4])
    mu = intruders_dict[most_voted_A][0]
    sigma = intruders_dict[most_voted_A][1]
    return mu, sigma, most_voted_A


def get_mahalanobis_distance_intruder_state_normalized(state, sigma, measurement, R, inverse_dist):
    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])
    hx = C @ state
    innovation = (inverse_dist**2)*(measurement - hx)

    S = C @ sigma @ C.T + R
    D2 = innovation.T @ np.linalg.inv(S) @ innovation

    return D2

