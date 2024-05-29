#!/usr/bin/env python3

import numpy as np
import os
import sys

sys.path.append('../Tools')
sys.path.append('/home/marmot/dixiao/catkin_ws/src/odom/leg_odom/Tools')



print("Current Working Directory:", os.getcwd())

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
import message_filters


import param_init
import cas_init
import Data_Import
import Leg_odometry_func as LO

# rostopic to be subscrib
# /isaac_go1/IMU_ang
# /isaac_go1/IMU_vel
# /isaac_go1/foot_force
# /isaac_go1/joint_angle
# /isaac_go1/joint_vel

N = 8000
X0 = param_init.Init.init_x     #init_x=0
P0 = param_init.Init.init_cov * np.eye(param_init.Param.state_size)
P0[21:27, 21:27] = param_init.Init.init_bias_cov * np.eye(6)
cov_list = np.zeros((param_init.Param.state_size, param_init.Param.state_size, N + 1))
x_list = np.zeros((param_init.Param.state_size, N))
x_list[:, 0] = np.squeeze(X0)
k=0

cov_list[:, :, 0] = P0
def callback(imu_ang, imu_vel, foot_force, joint_angle, joint_vel,dt,sub_pos):
    # 在这里处理同步接收到的消息
    # print("IMU ang:", imu_ang.data)
    # print("IMU vel:", imu_vel.data)
    # print("Foot force:", foot_force.data)
    # # print("Joint angle:", joint_angle.data)
    # print("Joint vel:", joint_vel.data)
    global k  # 全局变量k
    global x_list
    global cov_list


    # reorder the data to meet the requirement of algorithm
    new_order_indexes = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    hat_joint_ang = np.array([joint_angle.data[i] for i in new_order_indexes])
    hat_joint_vel =np.array([joint_vel.data[i] for i in new_order_indexes])

    # hat_joint_ang = np.array(joint_angle.data)
    # hat_joint_vel =np.array(joint_vel.data)


    uk = np.array([imu_ang.data, imu_vel.data ])
    uk = uk.reshape((-1, 1))
    uk = np.vstack((uk, [dt.data]))

    # print(imu_vel.data)

    # contact mode（0 for unconatc, 1 for contact)
    contact_force=foot_force.data[0:4]
    contact_mode = np.array([int(force > 1) for force in contact_force])
    
    x_list[:, k + 1], cov_list[:, :, k + 1] = LO.Lege_odom_func(x_list[:, k], uk, contact_mode, cov_list[:, :, k],
                                                                    hat_joint_ang, hat_joint_vel)
    



    # print(x_list[0:3, k + 1])
    # print(cov_list[:, :, k])
    k=k+1
    print(sub_pos.data)
    

    # if k<4998:
    #     dt = Data_Import.Accel_body_IMU[idx + 1, 0] - Data_Import.Accel_body_IMU[idx, 0]  # sampling interval of IMU
    #     # print(Data_Import.Gyro_body_IMU[idx, 1:])
    #     uk = np.array([Data_Import.Gyro_body_IMU[idx, 1:], Data_Import.Accel_body_IMU[idx, 1:], ])
    #     uk = uk.reshape((-1, 1))
    #     uk = np.vstack((uk, [dt]))
    #     contact = Data_Import.Contact_mode[idx, 1:]
    #     hat_joint_ang = Data_Import.Joint_ang[idx, 1:]
    #     hat_joint_vel = Data_Import.Joint_vel[idx, 1:]

    #     x_list[:, k + 1], cov_list[:, :, k + 1] = LO.Lege_odom_func(x_list[:, k], uk, contact, cov_list[:, :, k],
    #                                                                 hat_joint_ang, hat_joint_vel)
    #     k=k+1


if __name__ == "__main__":
    rospy.init_node("main_node")
    rospy.loginfo("main_node init success")

    sub_IMU_ang = message_filters.Subscriber("/isaac_go1/IMU_ang", Float32MultiArray)
    sub_IMU_vel = message_filters.Subscriber("/isaac_go1/IMU_vel", Float32MultiArray)
    sub_foot_force = message_filters.Subscriber("/isaac_go1/foot_force", Float32MultiArray)
    sub_joint_angle = message_filters.Subscriber("/isaac_go1/joint_angle", Float32MultiArray)
    sub_joint_vel = message_filters.Subscriber("/isaac_go1/joint_vel", Float32MultiArray)
    sub_pos=message_filters.Subscriber("/isaac_go1/position", Float32MultiArray)
    sub_dt=message_filters.Subscriber("/isaac_go1/dt",Float32)
    rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #     rospy.loginfo("start")
    ts = message_filters.ApproximateTimeSynchronizer([sub_IMU_ang, sub_IMU_vel, sub_foot_force, sub_joint_angle, sub_joint_vel,sub_dt,sub_pos], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)

    rospy.spin()



