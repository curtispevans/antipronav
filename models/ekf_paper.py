import numpy as np

def f(x, own_mav, u):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    '''
    beta_dot, r_dot_over_r, beta, one_over_r = x
    an = -own_mav[3]*u*np.sin(own_mav[2])
    ae = own_mav[3]*u*np.cos(own_mav[2])
    f_ = np.array([-2*beta_dot*r_dot_over_r + one_over_r*(-ae*np.cos(beta) - -an*np.sin(beta)),
                    beta_dot**2 - r_dot_over_r**2 + one_over_r*(-ae*np.sin(beta) + -an*np.cos(beta)),
                    beta_dot,
                    -r_dot_over_r * one_over_r])
    return f_


def jacobian_f(x, own_mav, u):
    beta_dot, r_dot_over_r, beta, one_over_r = x
    an = -own_mav[3]*u*np.sin(own_mav[2])
    ae = own_mav[3]*u*np.cos(own_mav[2])
    J = np.array([[-2*r_dot_over_r, 2*beta_dot, 1, 0],
                  [-2*beta_dot, -2*r_dot_over_r, 0, -one_over_r],
                  [one_over_r*(ae*np.sin(beta) - -an*np.cos(beta)), one_over_r*(-ae*np.cos(beta) - -an*np.sin(beta)), 0, 0],
                  [-ae*np.cos(beta)- -an*np.sin(beta), -ae*np.sin(beta)+ -an*np.cos(beta), 0, -r_dot_over_r]]).T
    return J

def measurement_model(x):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    beta_dot, r_dot_over_r, beta, one_over_r = x
    return np.array([beta])

def jacobian_measurement_model(x):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    beta_dot, r_dot_over_r, beta, one_over_r = x
    H = np.array([[0, 0, 1, 0]])
    return H

def kalman_update(mu, sigma, own_mav, u, measurement, Q, R, delta_t):
    # Prediction
    mu = mu + delta_t*f(mu, own_mav, u)
    J = jacobian_f(mu, own_mav, u)
    Jd = np.eye(len(mu)) + delta_t*J + 0.5*delta_t**2*J@J
    # sigma = Jd @ sigma @ Jd.T + delta_t**2*Q
    sigma = Jd @ sigma @ Jd.T + Q

    mu_bar = mu
    sigma_bar = sigma

    # Update measurement
    z = measurement_model(mu_bar)
    H = jacobian_measurement_model(mu_bar)
    S = H@sigma_bar@H.T + R
    K = sigma_bar@H.T@np.linalg.inv(S)
    # innovation = wrap(measurement - z, dim=0)
    innovation = measurement - z
    innovation[0] = wrap(innovation[0])
    # innovation[2] = wrap(innovation[2])
    # print(innovation)
    # print(K @ innovation.flatten())
    mu_bar = mu_bar + K @ innovation
    # print(mu_bar)
    I = np.eye(len(K))
    # print(K.shape, R.shape, K.T.shape)
    sigma_bar = (I - K@H)@sigma_bar@(I - K@H).T + K@R.reshape(-1,1)@K.T

    mu = np.array(mu_bar)
    # mu[0] = wrap(mu[0])
    # mu[2] = wrap(mu[2])
    sigma = sigma_bar
    
    return mu, sigma 

def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle