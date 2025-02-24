import numpy as np

class MavDynamics:
    def __init__(self, x0, Ts):
                            # North, East, Theta, Velocity
        self._state = np.array([x0[0], x0[1], x0[2], x0[3]])
        self.Ts = Ts

    def update(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self.Ts
        k1 = self._derivatives(self._state[0:4], u)
        k2 = self._derivatives(self._state[0:4] + time_step / 2. * k1, u)
        k3 = self._derivatives(self._state[0:4] + time_step / 2. * k2, u)
        k4 = self._derivatives(self._state[0:4] + time_step * k3, u)
        self._state[0:4] += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state, u):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        north_dot = state[3] * np.cos(state[2])
        east_dot = state[3] * np.sin(state[2])
        theta_dot = u
        vel_dot = 0.
        # collect the derivative of the states
        x_dot = np.array([north_dot, east_dot, theta_dot, vel_dot])
        return x_dot