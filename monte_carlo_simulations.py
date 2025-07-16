import numpy as np
import matplotlib.pyplot as plt
from models.mav_dynamics import MavDynamics
from IPython.core.debugger import set_trace

np.random.seed(42)  # For reproducibility

def get_random_ownship_start_pose(x_min, x_max, y_min, y_max):
    start_pose_x = np.random.uniform(x_min, x_max)
    start_pose_y = np.random.uniform(y_min, y_max)
    start_pose =  np.array([start_pose_x, start_pose_y])
    return start_pose

def get_random_intruder(ownship : MavDynamics, intruder_velocity) -> MavDynamics:
    ownship_pose = ownship._state[:2]
    ownship_heading = ownship._state[2]
    ownship_velocity = ownship._state[3]
    # TODO make this more robust by using the rotation matrix and possibly doing the wedge thing JJ mentioned
    r =intruder_velocity * 10
    rand_theta = np.random.uniform(-np.pi, np.pi)
    intruder_start_pose = r * np.array([np.cos(rand_theta), np.sin(rand_theta)]) 
    rand_end_theta = np.random.uniform(0, np.pi/2)
    intruder_end_pose = r * np.array([np.cos(rand_end_theta), np.sin(rand_end_theta)])
    intruder_heading = np.arctan2(intruder_end_pose[1] - intruder_start_pose[1], intruder_end_pose[0] - intruder_start_pose[0])
    # intruder_pose = ownship_pose + np.random.uniform(-1000, 1000, size=2)
    # intruder_velocity = np.random.uniform(30, 100)
    # intruder_heading = np.random.uniform(-np.pi, np.pi)
    intruder = MavDynamics([*intruder_start_pose, intruder_heading, intruder_velocity], ownship.Ts)
    return intruder


wingspan_cruise_speed = np.load('data/wingspan_cruise_speed.npy')


def get_simulated_data(Ts, num_scenarios, num_frames, plot_scenarios=False):
    # Ts = 1/30
    # plot_scenarios = False  # Set to False to disable plotting

    # num_scenarios = 10
    # num_frames = 300
    x_min, x_max = -1000, 1000
    y_min, y_max = -1000, 1000

    all_bearings = []
    all_pixel_sizes = []
    all_distances = []
    all_us = []
    all_mav_states = []
    true_As_vels = []
    ownship_velocities = []

    for i in range(num_scenarios):
        if plot_scenarios:
            plt.clf()
        ownship_start_pose = np.array([0, 0])
        ownship_heading = 0
        ownship_velocity = np.random.choice(wingspan_cruise_speed[:,1])*0 + 15
        ownship_velocities.append(ownship_velocity)
        ownship = MavDynamics([*ownship_start_pose, ownship_heading, ownship_velocity], Ts)
        u = 0.07 + 0*np.random.uniform(-0.7, 0.7)  # Random control input for the ownship

        A, intruder_velocity = wingspan_cruise_speed[np.random.choice(np.arange(len(wingspan_cruise_speed)))]
        mav2 = get_random_intruder(ownship, intruder_velocity)
        
        current_scenario_bearings = []
        current_scenario_pixel_sizes = []
        current_scenario_distances = []
        current_scenario_us = []
        current_scenario_mav_states = []

        true_As_vels.append([A, intruder_velocity])
        # mav2._state[3] = intruder_velocity  # Set the intruder's velocity based on wingspan
        for j in range(num_frames):
            current_scenario_us.append(u)
            ownship.update(u)
            mav2.update(0)  # Assuming no control input for the intruder
            distance = np.linalg.norm(ownship._state[:2] - mav2._state[:2])
            bearing = np.arctan2(mav2._state[1] - ownship._state[1], mav2._state[0] - ownship._state[0])
            relative_bearing = (bearing - ownship._state[2]) % (2 * np.pi)
            if relative_bearing > np.pi:
                relative_bearing -= 2 * np.pi
            pixel_size = A / distance
            current_scenario_bearings.append(relative_bearing)
            current_scenario_pixel_sizes.append(pixel_size)
            current_scenario_distances.append(distance)
            current_scenario_mav_states.append(ownship._state.copy())

            # TODO just run the scenarios without having to save the data
            # probably a single function call

            if plot_scenarios:
                plt.plot(ownship._state[1], ownship._state[0], 'ro')
                plt.plot(mav2._state[1], mav2._state[0], 'bo')
                r = mav2._state[3] * 10
                t = np.linspace(0, 2 * np.pi, 100)
                plt.plot(r*np.cos(t), r*np.sin(t), 'k--', alpha=0.5)
                plt.title(f'Bearing between Mavs: {np.rad2deg(relative_bearing)}')
                plt.axis('equal')
                plt.pause(0.001)
        if plot_scenarios:
            plt.show()
        all_bearings.append(current_scenario_bearings)
        all_pixel_sizes.append(current_scenario_pixel_sizes)
        all_distances.append(current_scenario_distances)
        all_us.append(current_scenario_us)
        all_mav_states.append(current_scenario_mav_states)

    # Save the data for each all scenarios
    np.save('data/all_bearings.npy', all_bearings)
    np.save('data/all_pixel_sizes.npy', all_pixel_sizes)
    np.save('data/all_distances.npy', all_distances)
    np.save('data/all_us.npy', all_us)
    np.save('data/all_mav_states.npy', all_mav_states)
    np.save('data/true_As_vels.npy', true_As_vels)

    return all_bearings, all_pixel_sizes, all_distances, all_us, all_mav_states, true_As_vels, ownship_velocities




