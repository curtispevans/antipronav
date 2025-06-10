import numpy as np

def f(x, own_mav, u, Ts):
    y1, y2, y3, y4 = x[:4]
    intruder_state = x[4:]
    ax = 0
    ay = own_mav[3]*u
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
                   y4/np.sqrt(S3**2 + S4**2)])
    
    A = np.block([[np.eye(2), Ts * np.eye(2), Ts**2/2 * np.eye(2)],
                  [np.zeros((2,2)), np.eye(2), Ts * np.eye(2)],
                  [np.zeros((2,2)), np.zeros((2,2)), np.eye(2)]])
    
    f_intruder = A @ intruder_state

    return np.concatenate((f_, f_intruder))

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

def measurement_model(x, own_mav, A):
    beta_dot, r_dot_over_r, beta, one_over_r = x[:4]
    intruder_px, intruder_py, intruder_vx, intruder_vy, intruder_ax, intruder_ay = x[4:]
    # Calculate the relative position of the intruder
    return np.array([beta, A*one_over_r])

def jacobian_measurement_model(fun, x, own_mav, A):
    hx = fun(x, own_mav, A)
    m = hx.shape[0]
    n = x.shape[0]
    eps = 0.0001
    H = np.zeros((m, n))
    for i in range(n):
        x_eps = np.copy(x)
        x_eps[i] += eps
        dh = (fun(x_eps, own_mav, A) - hx) / eps
        H[:, i] = dh
    return H

def psuedo_measurement_model(x, own_mav, A):
    beta_dot, r_dot_over_r, beta, one_over_r = x[:4]
    intruder_px, intruder_py, intruder_vx, intruder_vy, intruder_ax, intruder_ay = x[4:]
    # Calculate the relative position of the intruder
    distance = 1 / one_over_r  # inverse distance to distance
    own_pose = own_mav[:2]  # own position
    own_heading = own_mav[2]  # own heading in radians
    los = np.array([np.cos(beta + own_heading), np.sin(beta + own_heading)])
    intruder_pos = own_pose + distance * los
    return np.array([intruder_px - intruder_pos[0], intruder_py - intruder_pos[1]])


def kalman_update(mu, sigma, own_mav, u, measurement, Q, R, Ts, A):
    # prediction
    mu_bar = f(mu, own_mav, u, Ts)
    J = jacobian_f(f, mu_bar, own_mav, u, Ts)
    sigma = J @ sigma @ J.T + Q

    mu_bar = mu
    sigma_bar = sigma

    # measurement update 
    z = measurement_model(mu_bar, own_mav, A)
    H = jacobian_measurement_model(measurement_model, mu_bar, own_mav, A)
    S = H@sigma_bar@H.T + R
    K = sigma_bar@H.T@np.linalg.inv(S)
    # innovation = wrap(measurement - z, dim=0)
    innovation = np.array(measurement - z)
    innovation[0] = wrap(innovation[0])
    # innovation[2] = wrap(innovation[2])
    # print(innovation)
    mu_bar = mu_bar + K@(innovation)
    I = np.eye(len(K))
    sigma_bar = (I - K@H)@sigma_bar@(I - K@H).T + K@R@K.T

    mu = np.array(mu_bar)
    mu[0] = wrap(mu[0])
    mu[2] = wrap(mu[2])
    sigma = sigma_bar

    # psuedo measurement update
    R = np.diag([1e-8, 1e-8])**2  # psuedo measurement noise
    z_psuedo = psuedo_measurement_model(mu_bar, own_mav, A)
    H_psuedo = jacobian_measurement_model(psuedo_measurement_model, mu_bar, own_mav, A)
    S_psuedo = H_psuedo@sigma_bar@H_psuedo.T + R
    K_psuedo = sigma_bar@H_psuedo.T@np.linalg.inv(S_psuedo)
    innovation_psuedo = np.array(measurement - z_psuedo)
    mu = mu_bar + K_psuedo@(innovation_psuedo)
    I_psuedo = np.eye(len(K_psuedo))
    sigma = (I_psuedo - K_psuedo@H_psuedo)@sigma_bar@(I_psuedo - K_psuedo@H_psuedo).T + K_psuedo@R@K_psuedo.T

    return mu, sigma 



def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle

