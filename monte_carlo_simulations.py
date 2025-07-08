import numpy as np
import matplotlib.pyplot as plt
from models.mav_dynamics import MavDynamics

def get_random_ownship_start_pose(x_min, x_max, y_min, y_max):
    start_pose_x = np.random.uniform(x_min, x_max)
    start_pose_y = np.random.uniform(y_min, y_max)
    start_pose =  np.array([start_pose_x, start_pose_y])
    return start_pose

def get_random_intruder_start_pose(ownship):
    pass
    # return intruder



Ts = 1/30

num_scenarios = 10
num_frames = 300
x_min, x_max = -1000, 1000
y_min, y_max = -1000, 1000

all_bearings = []
all_pixel_sizes = []
all_distances = []
all_us = []
all_mav_states = []

for i in range(num_scenarios):
    ownship_start_pose = get_random_ownship_start_pose(x_min, x_max, y_min, y_max)
    ownship_heading = np.random.uniform(-np.pi, np.pi)
    ownship_velocity = np.random.uniform(30, 100)
    ownship = MavDynamics([*ownship_start_pose, ownship_heading, ownship_velocity])
    mav2_start_pose = get_random_intruder_start_pose(ownship)
    
    for j in range(num_frames):
        pass



