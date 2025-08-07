from jax import numpy as jnp
from jax import config
config.update('jax_enable_x64', True)

def f(x, Ts):
    '''
    x: state vector x=[pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
    '''
    F = jnp.block([[jnp.eye(2), Ts * jnp.eye(2), Ts**2/2 * jnp.eye(2)],
                  [jnp.zeros((2,2)), jnp.eye(2), Ts * jnp.eye(2)],
                  [jnp.zeros((2,2)), jnp.zeros((2,2)), jnp.eye(2)]])
    
    return F @ x, F

def barrier_function(x, barrier=0.1):
    '''
    x: state vector x=[pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
    '''
    # Barrier function to ensure the intruder does not get too aggressive in acceleration
    acceleration = jnp.linalg.norm(x[4:6])
    # if acceleration > barrier:  # Maximum acceleration threshold
    #     # print('here')
    #     x[4:6] = barrier*x[4:6] / acceleration   # Barrier condition violated
    x.at[4:6].set(jnp.where(acceleration <= barrier, x[4:6], barrier*x[4:6] / acceleration))
    return x  # Barrier condition satisfied


def y(x):
    '''
    x: state vector x=[pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
    '''
    C = jnp.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])
    return C @ x, C

def kalman_update(mu, sigma, measurement, Q, R, Ts):
    # Prediction
    mu, F = f(mu, Ts)
    tol = jnp.linalg.det(sigma[-2:, -2:])
    # mu = barrier_function(mu, tol)
    sigma = F @ sigma @ F.T + Q

    # Measurement update
    z, C = y(mu)
    S = C @ sigma @ C.T + R
    K = sigma @ C.T @ jnp.linalg.inv(S)

    mu = mu + K @ (measurement - z)
    sigma = (jnp.eye(len(mu)) - K @ C) @ sigma @ (jnp.eye(len(mu)) - K @ C).T + K @ R @ K.T

    return mu, sigma