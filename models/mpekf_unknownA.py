import numpy as np

def f(x, own_mav, u, Ts):
    '''
    x: state vector x=[beta_dot, r_dot_over_r, beta, one_over_r]
    u: control vector u=angular_velocity
    '''
    y1, y2, y3, y4, A = x
    ay = 0
    ax = own_mav[3]*u
    w1 = -Ts*ax
    w2 = -Ts*ay
    w3 = -ax/2 * Ts**2
    w4 = -ay/2 * Ts**2 
    S1 = y1 + y4*(w1*np.cos(y3) - w2*np.sin(y3))
    S2 = y2 + y4*(w1*np.sin(y3) + w2*np.cos(y3))
    S3 = Ts*y1 + y4*(w3*np.cos(y3) - w4*np.sin(y3))
    S4 = 1  + Ts*y2 + y4*(w3*np.sin(y3) + w4*np.cos(y3))
    f_ = np.array([(S1*S4 - S2*S3)/(S3**2 + S4**2),
                   (S1*S3 + S2*S4)/(S3**2 + S4**2),
                   y3 + np.arctan(S3/S4),
                   y4/np.sqrt(S3**2 + S4**2),
                   A])
    return f_

def jacobian_f(fun, x, own_mav, u, Ts):
    fx = fun(x, own_mav, u, Ts)
    m = fx.shape[0]
    n = x.shape[0]
    eps = 0.0001
    J = np.zeros((m, n))
    for i in range(n):
        x_eps = np.copy(x)
        x_eps[i] += eps
        df = (fun(x_eps, own_mav, u, Ts) - fx) / eps
        J[:, i] = df
    return J

def measurement_model(x):
    '''
    x: state vector x=[beta_dot, r_dot_over_r, beta, one_over_r]
    '''
    beta_dot, r_dot_over_r, beta, one_over_r, A = x
    return np.array([beta, one_over_r*A])

def jacobian_measurement_model(x):
    '''
    x: state vector x=[beta_dot, r_dot_over_r, beta, one_over_r]
    '''
    beta_dot, r_dot_over_r, beta, one_over_r, A = x
    H = np.array([[0, 0, 1, 0, 0],
                  [0, 0, 0, A, one_over_r]])
    return H

def kalman_update(mu, sigma, own_mav, u, measurement, Q, R, delta_t):
    # Prediction
    N = 1
    Ts = delta_t/N
    for i in range(N):
        mu = f(mu, own_mav, u, Ts)
        J = jacobian_f(f, mu, own_mav, u, Ts)
        # Jd = np.eye(len(mu)) + Ts*J + 0.5*Ts**2*J@J
        sigma = J @ sigma @ J.T + Q

    mu_bar = mu
    sigma_bar = sigma

    # Update measurement
    z = measurement_model(mu_bar)
    H = jacobian_measurement_model(mu_bar)
    S = H@sigma_bar@H.T + R
    K = sigma_bar@H.T@np.linalg.inv(S)
    innovation = measurement - z
    innovation[0] = wrap(innovation[0])
    mu_bar = mu_bar + K @ innovation
    I = np.eye(len(K))
    sigma_bar = (I - K@H)@sigma_bar@(I - K@H).T + K@R@K.T

    mu = np.array(mu_bar)
    mu[0] = wrap(mu[0])
    mu[2] = wrap(mu[2])
    sigma = sigma_bar
    
    return mu, sigma 

def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle