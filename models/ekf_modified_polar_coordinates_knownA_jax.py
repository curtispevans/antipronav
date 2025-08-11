from jax import jacfwd
import jax.numpy as jnp
import numpy as np
from jax import config
config.update('jax_enable_x64', True)


def f(x, own_mav, u, A=20):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    '''
    beta_dot, r_dot_over_r, beta, one_over_r = x
    ax = 0
    ay = own_mav[3]*u
    f_ = jnp.array([-2*beta_dot*r_dot_over_r + one_over_r*(-ay*jnp.cos(beta) - -ax*jnp.sin(beta)),
                    beta_dot**2 - r_dot_over_r**2 + one_over_r*(-ay*jnp.sin(beta) + -ax*jnp.cos(beta)),
                    beta_dot,
                   -r_dot_over_r * one_over_r])
    return f_

def jacobian_f(fun, x, own_mav, u, A=20):
    return jacfwd(fun, argnums=0)(x, own_mav, u, A)


def measurement_model(x, A=20):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    beta_dot, r_dot_over_r, beta, one_over_r = x
    return jnp.array([beta, A*one_over_r])


def jacobian_measurement_model(x, A=20):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    beta_dot, r_dot_over_r, beta, one_over_r = x
    H = jnp.array([[0, 0, 1, 0],
                  [0, 0, 0, A]])
    return H

def kalman_update(mu, sigma, own_mav, u, measurement, Q, R, delta_t, A=20):
    # Prediction
    mu = mu + delta_t*f(mu, own_mav, u, A)
    J = jacobian_f(f, mu, own_mav, u, A)
    Jd = jnp.eye(len(mu)) + delta_t*J + 0.5*delta_t**2*J@J
    sigma = Jd @ sigma @ Jd.T + Q

    mu_bar = mu
    sigma_bar = sigma

    # Update measurement
    z = measurement_model(mu_bar, A)
    H = jacobian_measurement_model(mu_bar, A)
    S = H@sigma_bar@H.T + R
    K = sigma_bar@H.T@jnp.linalg.inv(S)
    
    innovation = jnp.array(measurement - z)
    innovation.at[0].set(wrap(innovation[0]))  # wrap the bearing angle

    mu_bar = mu_bar + K@(innovation)
    mu = jnp.array(mu_bar)
    mu.at[0].set(wrap(mu_bar[0]))  # wrap the bearing angle velocity
    mu.at[2].set(wrap(mu_bar[2]))  # wrap the bearing angle

    I = jnp.eye(len(K))
    sigma_bar = (I - K@H)@sigma_bar@(I - K@H).T + K@R@K.T
    sigma = sigma_bar
    
    return mu, sigma 

def wrap(angle):
    angle -= 2*jnp.pi * jnp.floor((angle + jnp.pi) / (2*jnp.pi))
    return angle