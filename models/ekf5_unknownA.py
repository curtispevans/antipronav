import numpy as np
from jax import jacfwd
import jax.numpy as jnp


def f(x, own_vel):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    '''
    theta, c_n, c_e, eta, A = x
    los_n = jnp.cos(theta)
    los_e = jnp.sin(theta)
    v_n = c_n - own_vel[0]
    v_e = c_e - own_vel[1]
    bearing_dot_relative_velocity = los_n*v_n + los_e*v_e
    f_ = jnp.array([eta*(-los_e*v_n + los_n*v_e),
                   0,
                   0,
                   -eta**2*bearing_dot_relative_velocity, 
                   0])
    return f_


def jacobian_f(x, own_vel):
    return jacfwd(f, argnums=0)(x, own_vel)

def measurement_model(x):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    theta, c_n, c_e, eta, A = x
    return jnp.array([theta, A*eta])

def jacobian_measurement_model(x):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    return jacfwd(measurement_model, argnums=0)(x)

def kalman_update(mu, sigma, own_vel, measurement, Q, R, delta_t):
    # Prediction
    mu = mu + delta_t*f(mu, own_vel)
    J = jacobian_f(mu, own_vel)
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
    # TODO fix the wrapping of the angle
    # innovation = wrap(measurement - z, dim=0)
    innovation = np.array(measurement - z)
    innovation[0] = wrap(innovation[0])
    # print(innovation)
    mu_bar = mu_bar + K@(innovation)
    I = jnp.eye(len(K))
    sigma_bar = (I - K@H)@sigma_bar@(I - K@H).T + K@R@K.T

    
    mu = np.array(mu_bar)
    mu[0] = wrap(mu[0])
    sigma = sigma_bar
    
    return mu, sigma 

def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle