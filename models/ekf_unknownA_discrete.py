import numpy as np
from jax import jacfwd
import jax.numpy as jnp
import jax

def f(x, own_vel):
    los_n, los_e, pixel_size, c_n, c_e, eta, A = x
    v_n = c_n - own_vel[0]
    v_e = c_e - own_vel[1]
    bearing_dot_relative_velocity = los_n*v_n + los_e*v_e
    _f = jnp.array([eta*(los_e**2*v_n - los_n*los_e*v_e),
                     eta*(-los_n*los_e*v_n + los_n**2*v_e),
                     -pixel_size*eta*bearing_dot_relative_velocity,
                     0,
                     0,
                     -pixel_size*eta*bearing_dot_relative_velocity/A,
                     0])
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

def measurement_model(x):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    los_n, los_e, pixel_size, c_n, c_e, eta, A = x
    return jnp.array([los_n, los_e, pixel_size, pixel_size - A*eta])


def jacobian_measurement_model(x):
    return jacfwd(measurement_model, argnums=0)(x)

def kalman_update(mu, sigma, own_vel, measurement, Q, R, delta_t):
    # Prediction
    mu = motion_model(mu, own_vel, delta_t)
    J = jacobian_motion_model(mu, own_vel, delta_t)
    sigma = J @ sigma @ J.T + Q

    mu_bar = mu
    sigma_bar = sigma

    # Update measurement
    z = measurement_model(mu_bar)
    H = jacobian_measurement_model(mu_bar)
    S = H@sigma_bar@H.T + R
    K = sigma_bar@H.T@np.linalg.inv(S)
    mu_bar = mu_bar + K@(measurement - z)
    I = jnp.eye(len(K))
    sigma_bar = (I - K@H)@sigma_bar@(I - K@H).T + K@R@K.T

    mu = mu_bar
    sigma = sigma_bar
    
    return mu, sigma 