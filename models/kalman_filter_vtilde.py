import numpy as np
from jax import jacfwd
import jax.numpy as jnp
import jax

def f(x, own_vel):
    los_n, los_e, pixel_size, v_n, v_e = x
    bearing_dot_relative_velocity = los_n*v_n + los_e*v_e
    _f = jnp.array([(los_e**2*v_n - los_n*los_e*v_e),
                     (-los_n*los_e*v_n + los_n**2*v_e),
                     -2*pixel_size*bearing_dot_relative_velocity,
                     -v_n*bearing_dot_relative_velocity,
                     -v_e*bearing_dot_relative_velocity])
    return _f

def motion_model(x, own_vel, delta_t):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    delta_t: time step
    '''
    _f = f(x, own_vel)
    return x + _f * delta_t

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
    los_n, los_e, pixel_size, v_n, v_e = x
    bearing_dot_relative_velocity = los_n*v_n + los_e*v_e
    return jnp.array([los_n, los_e, pixel_size]) #, pixel_size*bearing_dot_relative_velocity])

def jacobian_measurement_model(x, own_vel):
    return jacfwd(measurement_model, argnums=0)(x, own_vel)

def kalman_update(mu, sigma, own_vel, measurement, Q, R, delta_t):
    # Prediction
    mu_bar = motion_model(mu, own_vel, delta_t)
    A = jacobian_motion_model(mu_bar, own_vel, delta_t)
    Ad = jnp.eye(len(mu_bar)) + delta_t*A + 0.5*delta_t**2*A@A
    sigma_bar = Ad@sigma@Ad.T + delta_t**2*Q


    # Update
    z = measurement_model(mu_bar, own_vel)
    H = jacobian_measurement_model(mu_bar, own_vel)
    S = H@sigma_bar@H.T + R
    K = sigma_bar@H.T@np.linalg.inv(S)
    mu = mu_bar + K@(measurement - z)
    I = jnp.eye(len(K))
    sigma = (I - K@H)@sigma_bar@(I - K@H).T + K@R@K.T
    
    return mu, sigma 