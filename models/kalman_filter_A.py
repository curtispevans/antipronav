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
                     -eta**2*bearing_dot_relative_velocity,
                    -0*pixel_size*bearing_dot_relative_velocity])
    return _f

def motion_model(x, own_vel, delta_t):
    '''
    x: state vector x=[los_x, los_y, pixel_area, relative_velocity_x, relative_velocity_y, inverse_distance]
    u: control vector u=[acceleration_x, acceleration_y]
    delta_t: time step
    '''
    _f = f(x, own_vel)
    return x + _f * delta_t

def jacobian_f(x, own_vel):
    return jacfwd(f, argnums=0)(x, own_vel)

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
    los_n, los_e, pixel_size, c_n, c_e, eta, A = x
    v_n = c_n - own_vel[0]
    v_e = c_e - own_vel[1]
    bearing_dot_relative_velocity = los_n*v_n + los_e*v_e
    return jnp.array([los_n, los_e, pixel_size])

def psuedo_measurement_model(x, own_vel):
    los_n, los_e, pixel_size, c_n, c_e, eta, A = x
    return jnp.array([pixel_size - A*eta])

def jacobian_psuedo_measurement_model(x, own_vel):
    return jacfwd(psuedo_measurement_model, argnums=0)(x, own_vel)

def jacobian_measurement_model(x, own_vel):
    return jacfwd(measurement_model, argnums=0)(x, own_vel)

def kalman_update(mu, sigma, own_vel, measurement, Q, R, R_psuedo, delta_t):
    # Prediction
    mu_bar = motion_model(mu, own_vel, delta_t)
    A = jacobian_motion_model(mu_bar, own_vel, delta_t)

    # mu_bar = mu + delta_t * f(mu, own_vel)
    # A = jacobian_f(mu_bar, own_vel)
    Ad = jnp.eye(len(mu_bar)) + delta_t*A + 0.5*delta_t**2*A@A
    sigma_bar = Ad@sigma@Ad.T + delta_t**2*Q


    # Update measurement
    z = measurement_model(mu_bar, own_vel)
    H = jacobian_measurement_model(mu_bar, own_vel)
    S = H@sigma_bar@H.T + R
    K = sigma_bar@H.T@np.linalg.inv(S)
    mu = mu_bar + K@(measurement - z)
    I = jnp.eye(len(K))
    sigma = (I - K@H)@sigma_bar@(I - K@H).T + K@R@K.T

    # update psuedo measurement
    measurement_psuedo = psuedo_measurement_model(mu_bar, own_vel)
    H_psuedo = jacobian_psuedo_measurement_model(mu_bar, own_vel)
    S_psuedo = H_psuedo@sigma@H_psuedo.T + R_psuedo
    K_psuedo = sigma@H_psuedo.T@np.linalg.inv(S_psuedo)
    mu = mu + K_psuedo@(jnp.array([0.]) - measurement_psuedo)
    I_psuedo = jnp.eye(len(K_psuedo))
    sigma = (I_psuedo - K_psuedo@H_psuedo)@sigma@(I_psuedo - K_psuedo@H_psuedo).T + K_psuedo@R_psuedo@K_psuedo.T
    
    return mu, sigma 