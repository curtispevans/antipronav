import numpy as np
import matplotlib.pyplot as plt
from models.mav_dynamics import MavDynamics
from scipy.interpolate import PchipInterpolator, PPoly

Ts = 1/30
                # north, east, heading, speed
mav1 = MavDynamics([-1000., 0., 0, 50.], Ts)
mav2 = MavDynamics([0., -1000., np.pi/6, 50], Ts)

u = 0.0
A = 20

bearings = []
bearings_vel = []
pixel_sizes = []
distances = []
control = []
relative_velocities = []
own_velocities = []
mav_state = []
us = []

x = [0, 50, 150, 250, 300, 350, 450]
y = [0, 0, 0.07, 0.07, 0.0, -0.07, -0.07]
f = PchipInterpolator(x, y)
t = np.linspace(0, 500, 500)
bearing_rate = f(t)

for i in range(1000): 
    if i < 50:
        u = 0.0
    elif 50 < i < 125:
        u = -0.3
    elif 125 < i < 200:
        u = 0.0
    elif 200 < i < 350:
        u = 0.3
    elif 350 < i < 500:
        u = 0.0
    elif 500 < i < 650:
        u = -0.3
    elif 650 < i < 800:
        u = 0.0
    elif 800 < i < 950:
        u = 0.3
    elif 950 < i < 1000:
        u = 0.0
    elif i < 900:
        u = 0.0
    # # else:
    # #     u = 0.7
    # if i < 50:
    #     u = 0.1
    # elif 50 < i < 125:
    #     u = -0.1
    # elif 125 < i < 200:
    #     u = 0.1
    # else:
    #     u = -0.1
    # if i < 10:
    #     u = 0.0
    # elif i % 200 < 100:
    #     u = 0.05
    # elif i % 200 > 100:
    #     u = -0.06
    # if 0 <= i % 500 < 200:
    #     u = -0.07
    # elif 200 < i % 500 < 400:
    #     u = 0.07
    # else:
    #     u = 0.0
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
    # bearings.append(relative_bearing + np.random.normal(0, np.radians(1)))
    if i > 0:
        bearings_vel.append((bearings[-1] - bearings[-2]) / Ts)

    rho = np.linalg.norm(mav2._state[0:2] - mav1._state[0:2])
    pixel_size = A / rho 
    pixel_sizes.append(pixel_size) 
    # pixel_sizes.append(pixel_size + np.random.normal(0, 0.01))
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

V = np.array([mav_state[i][3] for i in range(len(mav_state))])
dV = np.diff(V)

plt.subplot(221)
plt.plot(V)
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity of Mav 1')
plt.grid()

plt.subplot(222)
plt.plot(dV)
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.title('Acceleration of Mav 1')
plt.grid()

plt.subplot(223)
plt.plot(us)
plt.xlabel('Time')
plt.ylabel('Heading Rate')


dchi = np.diff(np.array([mav[2] for mav in mav_state]))/Ts
plt.subplot(224)
plt.plot(dchi)
plt.xlabel('Time')
plt.ylabel('Bearing Rate')
plt.title('Bearing Rate of Mav 1')
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