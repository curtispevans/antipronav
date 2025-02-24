import numpy as np
import matplotlib.pyplot as plt
from models.mav_dynamics import MavDynamics

Ts = 1/30

mav1 = MavDynamics([-300., 0., 0., 30.], Ts)
mav2 = MavDynamics([0., -300., np.pi/2, 30.], Ts)

u = 0.

bearings = []
bearings_vel = []

for i in range(500):
    mav1.update(u)
    mav2.update(u)

    plt.plot(mav1._state[1], mav1._state[0], 'ro')
    plt.plot(mav2._state[1], mav2._state[0], 'bo')
    bearing = np.arctan2(mav2._state[1] - mav1._state[1], mav2._state[2] - mav1._state[2])
    bearings.append(bearing)
    if i > 0:
        bearings_vel.append((bearings[-1] - bearings[-2]) / Ts)


plt.xlabel('East')
plt.ylabel('North')
plt.title('Mav Position')
plt.grid()
plt.show()


plt.plot(bearings)
plt.plot(bearings_vel)
plt.legend(['Bearing', 'Bearing Velocity'])
plt.xlabel('Time')
plt.ylabel('Bearing')
plt.title('Bearing between Mavs')
plt.grid()
plt.show()
