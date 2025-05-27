import numpy as np

def get_position_of_intruder(state, mav):
    distance = 1/state[3] # inverse distance to distance
    bearing = state[2] # bearing in radians
    own_pose = mav[0:2] # own position
    own_heading = mav[2] # own heading in radians
    los = np.array([np.cos(bearing + own_heading), np.sin(bearing + own_heading)])
    intruder_pos = own_pose + distance * los
    return intruder_pos


def get_intuder_velocity(intruder_pos, intruder_prev_pos, Ts):
    """
    Calculate the velocity of the intruder based on its current and previous position.
    """
    return (intruder_pos - intruder_prev_pos) / Ts

def get_intruder_acceleration(intruder_vel, intruder_prev_vel, Ts):
    """
    Calculate the acceleration of the intruder based on its current and previous velocity.
    """
    return (intruder_vel - intruder_prev_vel) / Ts

