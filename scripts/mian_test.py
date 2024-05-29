import numpy as np
import os
import sys

sys.path.append('../Tools')

print("Current Working Directory:", os.getcwd())

import param_init
import cas_init
import Data_Import
import Leg_odometry_func as LO

import matplotlib.pyplot as plt

X0 = param_init.Init.init_x     #init_x=0
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
    contact = Data_Import.Contact_mode[idx, 1:]
    hat_joint_ang = Data_Import.Joint_ang[idx, 1:]
    hat_joint_vel = Data_Import.Joint_vel[idx, 1:]

    x_list[:, k + 1], cov_list[:, :, k + 1] = LO.Lege_odom_func(x_list[:, k], uk, contact, cov_list[:, :, k],
                                                                hat_joint_ang, hat_joint_vel)



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
