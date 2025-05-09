import numpy as np
import matplotlib.pyplot as plt
from models.mav_dynamics import MavDynamics

Ts = 1/30
                # north, east, heading, speed
mav1 = MavDynamics([-1000., 0., 0, 30.], Ts)
mav2 = MavDynamics([0., -300., np.pi/2, 10.], Ts)

u = -0.07
A = 15

bearings = []
bearings_vel = []
pixel_sizes = []
distances = []
control = []
relative_velocities = []
own_velocities = []
mav_state = []
us = []

for i in range(1000): 
    if i < 100:
        u = -0.7
    elif i < 300:
        u = 0.7
    elif i < 500:
        u = -0.7 
    elif i < 700:
        u = 0.7
    elif i < 900:
        u = -0.7
    else:
        u = 0.7
    us.append(u)  
    mav1.update(u)
    mav_state.append(np.array([mav1._state[0], mav1._state[1], mav1._state[2], mav1._state[3]]))
    mav2.update(0.0)
    un = -mav1._state[3]*u*np.sin(mav1._state[2])
    ue = mav1._state[3]*u*np.cos(mav1._state[2])
    control.append(np.array([un, ue]))

    bearing = np.arctan2(mav2._state[1] - mav1._state[1], mav2._state[0] - mav1._state[0])
    relative_bearing = (bearing - mav1._state[2]) % (2*np.pi)
    if relative_bearing > np.pi:
        relative_bearing -= 2*np.pi
    bearings.append(relative_bearing)
    if i > 0:
        bearings_vel.append((bearings[-1] - bearings[-2]) / Ts)

    rho = np.linalg.norm(mav2._state[0:2] - mav1._state[0:2])
    pixel_size = A / rho 
    pixel_sizes.append(pixel_size)
    distances.append(rho)

    own_vel = mav1._state[3]*np.array([np.cos(mav1._state[2]), np.sin(mav1._state[2])])
    rel_vel_intruder = mav2._state[3]*np.array([np.cos(mav2._state[2]), np.sin(mav2._state[2])])
    relative_velocity = rel_vel_intruder - own_vel
    relative_velocities.append(relative_velocity)

    own_velocities.append(own_vel)

    plt.plot(mav1._state[1], mav1._state[0], 'ro')
    plt.plot(mav2._state[1], mav2._state[0], 'bo')
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
np.save('data/relative_velocity.npy', np.array(relative_velocities))
np.save('data/own_velocity.npy', np.array(own_velocities))
np.save('data/mav_state.npy', np.array(mav_state))
np.save('data/us.npy', np.array(us))