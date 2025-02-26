import numpy as np
import matplotlib.pyplot as plt
from models.mav_dynamics import MavDynamics

Ts = 1/30

mav1 = MavDynamics([-300., 0., 0., 30.], Ts)
mav2 = MavDynamics([0., -300., np.pi/2, 30.], Ts)

u = 0.

bearings = []
bearings_vel = []
pixel_sizes = []
distances = []
control = []

for i in range(500):
    mav1.update(-0.05)
    mav2.update(0.0)
    ux = -mav1._state[3]*-0.05*np.sin(mav1._state[2])
    uy = mav1._state[3]*-0.05*np.cos(mav1._state[2])
    control.append(np.array([ux, uy]))

    plt.plot(mav1._state[1], mav1._state[0], 'ro')
    plt.plot(mav2._state[1], mav2._state[0], 'bo')
    bearing = np.arctan2(mav2._state[1] - mav1._state[1], mav2._state[0] - mav1._state[0])
    relative_bearing = (bearing - mav1._state[2]) % (2*np.pi)
    if relative_bearing > np.pi:
        relative_bearing -= 2*np.pi
    bearings.append(relative_bearing - np.pi)
    if i > 0:
        bearings_vel.append((bearings[-1] - bearings[-2]) / Ts)

    rho = np.linalg.norm(mav2._state[0:2] - mav1._state[0:2])
    pixel_size = 10 / rho
    pixel_sizes.append(pixel_size)
    distances.append(rho)
    
    plt.title(f'Bearing between Mavs {np.rad2deg(relative_bearing)}')
    plt.pause(0.01)

plt.xlabel('East')
plt.ylabel('North')
plt.title('Mav Position')
plt.grid()
plt.show()

plt.subplot(221)
plt.plot(bearings)
# plt.plot(bearings_vel)
# plt.legend(['Bearing', 'Bearing Velocity'])
plt.xlabel('Time')
plt.ylabel('Bearing')
plt.title('Bearing between Mavs')
plt.grid()


plt.subplot(222)
plt.plot(pixel_sizes)
plt.xlabel('Time')
plt.ylabel('Pixel Size')
plt.title('Pixel Size')
plt.grid()

plt.subplot(223)
plt.plot(distances)
plt.xlabel('Time')
plt.ylabel('Distance')
plt.title('Distance between Mavs')
plt.grid()

plt.tight_layout()
plt.show()

np.save('data/bearing.npy', np.array(bearings))
np.save('data/pixel_sizes.npy', np.array(pixel_sizes))
np.save('data/distances.npy', np.array(distances))
np.save('data/control.npy', np.array(control))