import numpy as np
import Data_Import as extData
import Forward_kinematics as fk
import euler_to_rot as eul



class Param:
    # process_noise
    proc_n_pos = 0.0005
    proc_n_vel_xy = 0.005
    proc_n_vel_z = 0.005
    proc_n_ang = 1e-7
    proc_n_foot_pos = 1e-4
    proc_n_ba = 1e-4  # White Gaussian noise covariance (random walk of accelerometer; w_f/Q_f in paper)
    proc_n_bg = 1e-5  # White Gaussian noise covariance (random walk of gyro bias terms; w_w/Q_w in paper)

    # control noise
    ctrl_n_acc = 1e-1
    ctrl_n_gyro = 1e-3

    # measurement noise
    meas_n_fk_pos = 0.001  # joint angle measurement
    meas_n_fk_vel = 0.01  # joint velocity measurement
    meas_n_foot_height = 0.001

    # control data
    num_leg = 4
    total_step, num_colum = extData.Accel_body_IMU.shape
    # total_start_idx = 200
    total_start_idx = 199
    total_end_idx = total_step - 1
    total_start_time = extData.Accel_body_IMU[total_start_idx, 0]
    meas_per_leg = 7  # joint angle, joint velocity, foot height (3 joint each leg) 3,3,1
    meas_size = meas_per_leg * num_leg


    state_size = 28
    # state x
    # - position(1: 3)
    # - velocity(4: 6)
    # - euler angle(7: 9)
    # - foot1(10: 12)
    # - foot2(13: 15)
    # - foot3(16: 18)
    # - foot4(19: 21)
    # - acc bias(22: 24)
    # - gyro bias(25: 27)
    # - time tk(28)
    ctrl_size = 7
    # control u
    # - w(1: 3) IMU angular velocity
    # - a(4: 6) IMU acceleration
    # - hk(7)
    R = 1e-2 * np.eye(meas_size)


# init data
class Init:
    init_position = extData.Pos_mocap[Param.total_start_idx, 1:]  # init data get from motion capture system
    init_euler = np.zeros((3, 1))  # Roll Pitch Yaw (X,Y,Z axis)
    R_er = eul.euler_to_rot(init_euler)  # Rotation matrix(convert coordinate from body frame to word frame)
    init_joint_angle = extData.Joint_ang[Param.total_start_idx, 1:]
    init_ft_pos = np.zeros((Param.num_leg * 3, 1))

    for i in range(Param.num_leg):
        ft_pos = fk.fK_function(init_joint_angle[i * 3:i * 3 + 3], i)
        init_ft_pos[i * 3:i * 3 + 3, :] = (ft_pos + init_position).reshape(-1, 1)

    init_position = init_position.reshape(-1, 1)  # reshape init_position into column vector
    init_state = np.vstack(
        (init_position, np.zeros((3, 1)), init_euler, init_ft_pos, np.zeros((3, 1)), np.zeros((3, 1)), [0]))
    # init_state is init state vector X0,size=state_size

    init_cov = 0.1
    init_bias_cov = 1e-4  # Randomly set a covariance, it will be updated in the Kalman filter.
    init_cov_matrix = init_cov * np.eye(Param.state_size)
    init_cov_matrix[21:27, 21:27] = init_bias_cov * np.eye(
        6)  # the covariance matrix P represents the uncertainties of the estimated state vector(eq.9)

    init_x = np.vstack((init_position, np.zeros((3, 1)), init_euler, init_ft_pos, np.zeros((7, 1))))
    # print(init_x)
    # print(np.size(init_x))
