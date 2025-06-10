import numpy as np

def f(x, own_mav, u):
    beta_dot, r_dot_over_r, beta, one_over_r = x[:4]
    intruder_px, intruder_py, intruder_vx, intruder_vy, intruder_ax, intruder_ay = x[4:]
    ax = 0
    ay = own_mav[3] * u
    f_ = np.array([-2*beta_dot*r_dot_over_r + one_over_r*(-ay*np.cos(beta) - -ax*np.sin(beta)),
                    beta_dot**2 - r_dot_over_r**2 + one_over_r*(-ay*np.sin(beta) + -ax*np.cos(beta)),
                    beta_dot,
                   -r_dot_over_r * one_over_r])
    f_intruder = np.array([intruder_vx,
                           intruder_vy,
                           intruder_ax,
                           intruder_ay,
                           0, 
                           0])

    return np.concatenate((f_, f_intruder))


def jacobian_f(fun, x, own_mav, u):
    fx = fun(x, own_mav, u)
    m = fx.shape[0]
    n = x.shape[0]
    eps = 0.0001
    J = np.zeros((m, n))
    for i in range(n):
        x_eps = np.copy(x)
        x_eps[i] += eps
        df = (fun(x_eps, own_mav, u) - fx) / eps
        J[:, i] = df
    return J

def measurement_model(x, own_mav, A):
    beta_dot, r_dot_over_r, beta, one_over_r = x[:4]
    intruder_px, intruder_py, intruder_vx, intruder_vy, intruder_ax, intruder_ay = x[4:]
    # Calculate the relative position of the intruder
    distance = 1 / one_over_r  # inverse distance to distance
    own_pose = own_mav[:2]  # own position
    own_heading = own_mav[2]  # own heading in radians
    los = np.array([np.cos(beta + own_heading), np.sin(beta + own_heading)])
    intruder_pos = own_pose + distance * los
    return np.array([beta, A*one_over_r, intruder_px - intruder_pos[0], intruder_py - intruder_pos[1]])

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

def kalman_update(mu, sigma, own_mav, u, measurement, Q, R, Ts, A):
    # prediction
    mu_bar = mu + Ts * f(mu, own_mav, u)
    J = jacobian_f(f, mu_bar, own_mav, u)
    Jd = np.eye(len(mu)) + Ts*J + 0.5*Ts**2*J@J
    sigma = Jd @ sigma @ Jd.T + Q

    # measurement update
    z = measurement_model(mu_bar, own_mav, A)
    H = jacobian_measurement_model(measurement_model, mu_bar, own_mav, A)
    S = H @ sigma @ H.T + R
    K = sigma @ H.T @ np.linalg.inv(S)
    innovation = measurement - z
    innovation[0] = wrap(innovation[0])
    # innovation[2] = wrap(innovation[2])
    mu_bar = mu + K @ innovation

    I = np.eye(len(mu))
    sigma = (I - K @ H) @ sigma @ (I - K @ H).T + K @ R @ K.T

    mu[0] = wrap(mu[0])
    mu[2] = wrap(mu[2])

    return mu, sigma



def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle