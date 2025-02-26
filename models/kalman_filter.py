import numpy as np
from jax import jacfwd
import jax.numpy as jnp

def motion_model(x, u, delta_t):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    delta_t: time step
    '''
    los_x, los_y, pixel_size, v_x, v_y, eta = x
    bearing_dot_relative_velocity = los_x*v_x + los_y*v_y
    f = jnp.array([eta*(los_y**2*v_x - los_x*los_y*v_y),
                   eta*(-los_x*los_y*v_x + los_x**2*v_y),
                   -2*pixel_size*eta*bearing_dot_relative_velocity,
                   -u[0],
                   -u[1],
                   -eta**2*bearing_dot_relative_velocity])
    return x + f*delta_t

def jacobian_motion_model(x, u, delta_t):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    delta_t: time step
    '''
    return jacfwd(motion_model, argnums=0)(x, u, delta_t)

def measurement_model(x):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    H = jnp.array([[1,0,0,0,0,0],
                   [0,1,0,0,0,0],
                   [0,0,1,0,0,0]])
    return jnp.array([x[0], x[1], x[2]]), H

def kalman_update(mu, sigma, u, measurement, R, Q, delta_t):
    # Prediction
    mu_bar = motion_model(mu, u, delta_t)
    J = jacobian_motion_model(mu, u, delta_t)
    sigma_bar = J@sigma@J.T + R

    # Update
    z, H = measurement_model(mu_bar)
    S = H@sigma_bar@H.T + Q
    K = sigma_bar@H.T@np.linalg.inv(S)
    mu = mu_bar + K@(measurement - z)
    sigma = (jnp.eye(len(K)) - K@H)@sigma_bar

    return mu, sigma 