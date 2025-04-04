import numpy as np
from jax import jacfwd
import jax.numpy as jnp

def motion_model(x, own_vel, delta_t):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    delta_t: time step
    '''
    theta, pixel_size, c_n, c_e, eta = x
    los_n = jnp.cos(theta)
    los_e = jnp.sin(theta)
    v_n = c_n - own_vel[0]
    v_e = c_e - own_vel[1]
    bearing_dot_relative_velocity = los_n*v_n + los_e*v_e
    f = jnp.array([eta*(-los_e*v_n + los_n*v_e),
                   -2*pixel_size*eta*bearing_dot_relative_velocity,
                   0,
                   0,
                   -eta**2*bearing_dot_relative_velocity])
    return x + f*delta_t

def jacobian_motion_model(x, own_vel, delta_t):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    delta_t: time step
    '''
    return jacfwd(motion_model, argnums=0)(x, own_vel, delta_t)

def measurement_model(x):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    H = jnp.array([[1,0,0,0,0],
                   [0,1,0,0,0]])
    return jnp.array([x[0], x[1]]), H

def kalman_update(mu, sigma, own_vel, measurement, R, Q, delta_t):
    # Prediction
    mu_bar = motion_model(mu, own_vel, delta_t)
    J = jacobian_motion_model(mu, own_vel, delta_t)
    sigma_bar = J@sigma@J.T + R

    # Update
    z, H = measurement_model(mu_bar)
    S = H@sigma_bar@H.T + Q
    K = sigma_bar@H.T@np.linalg.inv(S)
    mu = mu_bar + K@(measurement - z)
    sigma = (jnp.eye(len(K)) - K@H)@sigma_bar

    return mu, sigma 