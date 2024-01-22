import numpy as np
import os
import sys

sys.path.append('../Tools')

print("Current Working Directory:", os.getcwd())

import param_init
import cas_init
import Data_Import

import matplotlib.pyplot as plt

X0 = param_init.Init.init_x
N = param_init.Param.total_end_idx - param_init.Param.total_start_idx + 1
P0 = param_init.Init.init_cov * np.eye(param_init.Param.state_size)
P0[21:27, 21:27] = param_init.Init.init_bias_cov * np.eye(6)
cov_list = np.zeros((param_init.Param.state_size, param_init.Param.state_size, N + 1))
x_list = np.zeros((param_init.Param.state_size, N))
x_list[:, 0] = np.squeeze(X0)

cov_list[:, :, 0] = P0
for idx in range(param_init.Param.total_start_idx, param_init.Param.total_end_idx):
    k = idx - param_init.Param.total_start_idx
    dt = Data_Import.Accel_body_IMU[idx + 1, 0] - Data_Import.Accel_body_IMU[idx, 0]  # sampling interval of IMU
    # print(Data_Import.Gyro_body_IMU[idx, 1:])
    uk = np.array([Data_Import.Gyro_body_IMU[idx, 1:], Data_Import.Accel_body_IMU[idx, 1:], ])
    uk = uk.reshape((-1, 1))
    uk = np.vstack((uk, [dt]))
    X01 = np.array(cas_init.next_state_func(x_list[:, k], uk, dt))
    F = np.array(cas_init.process_jacobian_func(x_list[:, k], uk, dt))
    B = np.array(cas_init.control_jacobian_func(x_list[:, k], uk, dt))

    # noise mtx
    noise_array1 = np.concatenate([
        np.full(2, param_init.Param.proc_n_pos * dt),  # pos x y
        np.full(1, param_init.Param.proc_n_pos * dt),  # pos z
        np.full(1, param_init.Param.proc_n_vel_xy * dt),  # vel x
        np.full(1, param_init.Param.proc_n_vel_xy * dt),  # vel y
        np.full(1, param_init.Param.proc_n_vel_z * dt),  # vel z
        np.full(3, param_init.Param.proc_n_ang * dt),  # ang
        np.full(12, param_init.Param.proc_n_foot_pos * dt),  # foot pos
        np.full(3, param_init.Param.proc_n_ba * dt),  # acc bias random walk
        np.full(3, param_init.Param.proc_n_bg * dt),  # gyro bias random walk
        [0]
    ])
    Q1 = np.diag(noise_array1)

    noise_array2 = np.concatenate([
        np.full(3, param_init.Param.ctrl_n_acc * dt),  # acc noise should be large around contact
        np.full(3, param_init.Param.ctrl_n_gyro * dt),  # gyro noise
        [0],
    ])
    Q2 = np.diag(noise_array2)
    R = 1e-2 * np.eye(param_init.Param.meas_size)
    contact = Data_Import.Contact_mode[idx, 1:]
    # the noise parameter of a certain foothold is set to infinity (or to a very large value)
    # whenever it has no ground contact. (Q1, foot_position)
    for i in range(0, param_init.Param.num_leg):
        Q1[9 + i * 3:12 + i * 3, 9 + i * 3:12 + i * 3] = (
                (1 + (1 - contact[i]) * 1e5) * param_init.Param.proc_n_foot_pos * dt * np.eye(3))

        R[i * param_init.Param.meas_per_leg: i * param_init.Param.meas_per_leg + 3,
        i * param_init.Param.meas_per_leg: i * param_init.Param.meas_per_leg + 3] = (
                (1 + (1 - contact[i]) * 1e5) * param_init.Param.meas_n_fk_pos * dt * np.eye(3))
        R[i * param_init.Param.meas_per_leg + 3: i * param_init.Param.meas_per_leg + 6,
        i * param_init.Param.meas_per_leg + 3: i * param_init.Param.meas_per_leg + 6] = (
                (1 + (1 - contact[i]) * 1e5) * param_init.Param.meas_n_fk_vel * dt * np.eye(3))
        R[i * param_init.Param.meas_per_leg + 6, i * param_init.Param.meas_per_leg + 6] = (
                (1 + (1 - contact[i]) * 1e5) * param_init.Param.meas_n_foot_height * dt)

    P01 = F @ cov_list[:, :, k] @ F.T + Q1 + B @ Q2 @ B.T
    # print(P01)
    hat_IMU_gyro = Data_Import.Gyro_body_IMU[idx, 1:]
    hat_joint_ang = Data_Import.Joint_ang[idx, 1:]
    hat_joint_vel = Data_Import.Joint_vel[idx, 1:]
    # hat_yawk = Data_Import.Orient_mocap_euler[idx, 3]
    yk = np.array(cas_init.yk_Func(X01, hat_IMU_gyro, hat_joint_ang, hat_joint_vel))  # measurement residual(Eq:46)
    H = np.array(cas_init.H_jac_Func(X01, hat_IMU_gyro, hat_joint_ang, hat_joint_vel))
    # print(yk)
    S = H @ P01 @ H.T + R
    Kk = P01 @ H.T @ np.linalg.inv(S)
    update = Kk @ yk
    x_list[:, k + 1] = np.squeeze(X01 - update)
    cov_list[:, :, k + 1] = (np.eye(param_init.Param.state_size) - Kk @ H) @ P01
    print(k)

predict_state_list = x_list
predict_position_list = predict_state_list[0:3, :]
predict_position_list = predict_position_list.T
ground_truth_list = Data_Import.Pos_mocap[param_init.Param.total_start_idx:param_init.Param.total_end_idx + 1, 1:]
# print(predict_state_list)

# plot
# Plotting the graph
plt.figure(figsize=(10, 8))
# Plotting the real position
plt.plot(ground_truth_list[:, 0], ground_truth_list[:, 1], label='Ground Truth', color='blue')
# Plotting the predicted position
plt.plot(predict_position_list[:, 0], predict_position_list[:, 1], label='Predicted Position', color='red')
# Adding a legend
plt.legend()
# Adding a title and axis labels
plt.title('Ground Truth vs Predicted Position')
plt.xlabel('X Position')
plt.ylabel('Y Position')
# Displaying the graph
plt.show()
