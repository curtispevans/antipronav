import numpy as np

def f(x, Ts):
    '''
    x: state vector x=[pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
    '''
    A = np.block([[np.eye(2), Ts * np.eye(2), Ts**2/2 * np.eye(2)],
                  [np.zeros((2,2)), np.eye(2), Ts * np.eye(2)],
                  [np.zeros((2,2)), np.zeros((2,2)), np.eye(2)]])
    
    return A @ x, A

def y(x):
    '''
    x: state vector x=[pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
    '''
    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])
    return C @ x, C

def kalman_update(mu, sigma, measurement, Q, R, Ts):
    # Prediction
    mu, A = f(mu, Ts)
    sigma = A @ sigma @ A.T + Q

    # Measurement update
    z, C = y(mu)
    S = C @ sigma @ C.T + R
    K = sigma @ C.T @ np.linalg.inv(S)
    
    mu = mu + K @ (measurement - z)
    sigma = (np.eye(len(mu)) - K @ C) @ sigma

    return mu, sigma