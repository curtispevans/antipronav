import numpy as np
from jax import jacfwd
import jax.numpy as jnp
import jax

def f(x, own_vel, A=15):
    los_n, los_e, pixel_size, c_n, c_e, eta = x
    v_n = c_n - own_vel[0]
    v_e = c_e - own_vel[1]
    bearing_dot_relative_velocity = los_n*v_n + los_e*v_e
    _f = jnp.array([eta*(los_e**2*v_n - los_n*los_e*v_e),
                     eta*(-los_n*los_e*v_n + los_n**2*v_e),
                     -pixel_size*eta*bearing_dot_relative_velocity,
                     0,
                     0,
                     -pixel_size*eta*bearing_dot_relative_velocity/A])
    return _f


def jacobian_f(x, own_vel, A=15):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    delta_t: time step
    '''
    return jacfwd(f, argnums=0)(x, own_vel, A)

def measurement_model(x, A=15):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    '''
    los_n, los_e, pixel_size, c_n, c_e, eta = x
    return jnp.array([los_n, los_e, pixel_size, pixel_size - A*eta, (los_n**2 + los_e**2) - 1])


def jacobian_measurement_model(x, A=15):
    return jacfwd(measurement_model, argnums=0)(x, A)

def kalman_update(mu, sigma, own_vel, measurement, Q, R, R_psuedo, delta_t, A=15):
    # Prediction
    if mu[0] < 0.1:
        print(mu[0], f(mu, own_vel, A)[0])
    mu = mu + delta_t*f(mu, own_vel, A)
    J = jacobian_f(mu, own_vel, A)
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
    mu_bar = mu_bar + K@(measurement - z)
    I = jnp.eye(len(K))
    sigma_bar = (I - K@H)@sigma_bar@(I - K@H).T + K@R@K.T

    mu = mu_bar
    sigma = sigma_bar
    
    return mu, sigma 