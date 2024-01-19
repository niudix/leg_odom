import numpy as np


def euler_to_rot(euler: np.ndarray) -> np.ndarray:
    # listed by roll pitch yaw
    r = euler[0, 0]  # x axis
    p = euler[1, 0]  # y axis
    y = euler[2, 0]  # z axis

    rot_r = np.array([[1, 0, 0],
                      [0, np.cos(r), -np.sin(r)],
                      [0, np.sin(r), np.cos(r)]])

    rot_p = np.array([[np.cos(p), 0, np.sin(p)],
                      [0, 1, 0],
                      [-np.sin(p), 0, np.cos(p)]])

    rot_y = np.array([[np.cos(y), -np.sin(y), 0],
                      [np.sin(y), np.cos(y), 0],
                      [0, 0, 1]])

    rot = np.dot(rot_y, np.dot(rot_p, rot_r))
    return rot
