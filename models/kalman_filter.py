import numpy as np
from jax import jacfwd
import jax.numpy as jnp

def motion_model(x, own_vel, delta_t):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    delta_t: time step
    '''
    los_n, los_e, pixel_size, c_n, c_e, eta = x
    v_n = c_n - own_vel[0]
    v_e = c_e - own_vel[1]
    bearing_dot_relative_velocity = los_n*v_n + los_e*v_e
    f = jnp.array([eta*(los_e**2*v_n - los_n*los_e*v_e),
                   eta*(-los_n*los_e*v_n + los_n**2*v_e),
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

def measurement_model(x, own_vel):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    los_n, los_e, pixel_size, c_n, c_e, eta = x
    v_n = c_n - own_vel[0]
    v_e = c_e - own_vel[1]
    bearing_dot_relative_velocity = los_n*v_n + los_e*v_e
    return jnp.array([los_n, los_e, pixel_size, -pixel_size*bearing_dot_relative_velocity])

def jacobian_measurement_model(x, own_vel):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    ''' 
    return jacfwd(measurement_model, argnums=0)(x, own_vel)

def kalman_update(mu, sigma, own_vel, measurement, R, Q, delta_t):
    # Prediction
    mu_bar = motion_model(mu, own_vel, delta_t)
    J = jacobian_motion_model(mu, own_vel, delta_t)
    Ad = jnp.eye(len(mu)) + delta_t*J + 0.5*delta_t**2*J@J
    sigma_bar = Ad@sigma@Ad.T + delta_t**2*R
    # sigma_bar = J@sigma@J.T + R

    # Update
    z = measurement_model(mu_bar, own_vel)
    H = jacobian_measurement_model(mu_bar, own_vel)
    S = H@sigma_bar@H.T + Q
    K = sigma_bar@H.T@np.linalg.inv(S)
    mu = mu_bar + K@(measurement - z)
    I = jnp.eye(len(K))
    sigma = (I - K@H)@sigma_bar@(I - K@H).T + K@Q@K.T
    sigma = (jnp.eye(len(K)) - K@H)@sigma_bar
    
    return mu, sigma 