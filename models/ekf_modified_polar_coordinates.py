import numpy as np
from jax import jacfwd
import jax.numpy as jnp


def f(x, own_mav, u, A=15):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    '''
    beta_dot, r_dot_over_r, beta, one_over_r, epsilon_A = x
    an = -own_mav[3]*u*np.sin(own_mav[2])
    ae = own_mav[3]*u*np.cos(own_mav[2])
    f_ = jnp.array([-2*beta_dot*r_dot_over_r + one_over_r*(-ae*jnp.cos(beta) - -an*jnp.sin(beta)),
                    beta_dot**2 - r_dot_over_r**2 + one_over_r*(-ae*jnp.sin(beta) + -an*jnp.cos(beta)),
                    beta_dot,
                   -r_dot_over_r * one_over_r,
                   -A*one_over_r*r_dot_over_r])
    return f_


def jacobian_f(x, own_mav, u, A=15):
    beta_dot, r_dot_over_r, beta, one_over_r, epsilon_A = x
    an = -own_mav[3]*u*np.sin(own_mav[2])
    ae = own_mav[3]*u*np.cos(own_mav[2])
    J = np.array([[-2*r_dot_over_r, 2*beta_dot, 1, 0, 0],
                  [-2*beta_dot, -2*r_dot_over_r, 0, -one_over_r, -A*one_over_r],
                  [one_over_r*(ae*np.sin(beta) - -an*np.cos(beta)), one_over_r*(-ae*np.cos(beta) - -an*np.sin(beta)), 0, 0, 0],
                  [-ae*np.cos(beta)- -an*np.sin(beta), -ae*np.sin(beta)+ -an*np.cos(beta), 0, -r_dot_over_r, -A*r_dot_over_r],
                  [0, 0, 0, 0, 0]]).T
    return J

def measurement_model(x, A=15):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    beta_dot, r_dot_over_r, beta, one_over_r, alpha = x
    return jnp.array([beta, alpha, alpha - A*one_over_r])

def jacobian_measurement_model(x, A=15):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    beta_dot, r_dot_over_r, beta, one_over_r, alpha = x
    H = np.array([[0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, -A, 1]])
    return H

def kalman_update(mu, sigma, own_mav, u, measurement, Q, R, delta_t, A=15):
    # Prediction
    mu = mu + delta_t*f(mu, own_mav, u, A)
    J = jacobian_f(mu, own_mav, u, A)
    Jd = np.eye(len(mu)) + delta_t*J + 0.5*delta_t**2*J@J
    # sigma = Jd @ sigma @ Jd.T + delta_t**2*Q
    sigma = Jd @ sigma @ Jd.T + Q

    mu_bar = mu
    sigma_bar = sigma

    # Update measurement
    z = measurement_model(mu_bar, A)
    H = jacobian_measurement_model(mu_bar, A)
    S = H@sigma_bar@H.T + R
    K = sigma_bar@H.T@np.linalg.inv(S)
    # innovation = wrap(measurement - z, dim=0)
    innovation = np.array(measurement - z)
    innovation[0] = wrap(innovation[0])
    # print(innovation)
    mu_bar = mu_bar + K@(innovation)
    I = jnp.eye(len(K))
    sigma_bar = (I - K@H)@sigma_bar@(I - K@H).T + K@R@K.T

    mu = np.array(mu_bar)
    mu[0] = wrap(mu[0])
    mu[2] = wrap(mu[2])
    sigma = sigma_bar
    
    return mu, sigma 

def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle