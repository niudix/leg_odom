#!/usr/bin/env python3

import numpy as np
import os
import sys

sys.path.append('../Tools')
# sys.path.append('/home/marmot/dixiao/catkin_ws/src/odom/leg_odom/Tools')

print("Current Working Directory:", os.getcwd())

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
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


step_count=0
def callback(imu_data, joint_state):

    # print("joint_state", joint_state.position)
    # print("time_stamp", imu_data.Header)
    # # print("Joint angle:", joint_angle.data)
    # print("Joint vel:", joint_vel.data)
    global k  # 全局变量k
    global x_list
    global cov_list
    global step_count
    global time_before


    # # reorder the data to meet the requirement of algorithm
    if (step_count==0):
        time_before=imu_data.header.stamp
        step_count=step_count+1
        X0 = param_init.Init.init_x     #init_x=0
        P0 = param_init.Init.init_cov * np.eye(param_init.Param.state_size)
        P0[21:27, 21:27] = param_init.Init.init_bias_cov * np.eye(6)
        cov_list = np.zeros((param_init.Param.state_size, param_init.Param.state_size, N + 1))
        x_list = np.zeros((param_init.Param.state_size, N))
        x_list[:, 0] = np.squeeze(X0)
        k=0
        cov_list[:, :, 0] = P0
        return
    
    
    step_count=step_count+1
    time_now=imu_data.header.stamp
    
    dt = (time_now - time_before).to_sec()
    time_before=time_now
    hat_joint_ang = np.array([joint_state.position[:12]])
    hat_joint_vel =np.array([joint_state.velocity[:12]])


    # 提取角速度和线加速度的 x, y, z 分量
    angular_velocity_x = imu_data.angular_velocity.x
    angular_velocity_y = imu_data.angular_velocity.y
    angular_velocity_z = imu_data.angular_velocity.z

    linear_acceleration_x = imu_data.linear_acceleration.x
    linear_acceleration_y = imu_data.linear_acceleration.y
    linear_acceleration_z = imu_data.linear_acceleration.z

    # 创建一个包含所有这些值和时间差 dt 的7行1列的数组
    uk = np.array([[angular_velocity_x],
                [angular_velocity_y],
                [angular_velocity_z],
                [linear_acceleration_x],
                [linear_acceleration_y],
                [linear_acceleration_z],
                [dt]])

    # contact mode（0 for unconatc, 1 for contact)
    contact_force=joint_state.effort[-4:]
    contact_mode = np.array([int(force > 100) for force in contact_force])
        
    x_list[:, k + 1], cov_list[:, :, k + 1] = LO.Lege_odom_func(x_list[:, k], uk, contact_mode, cov_list[:, :, k],
                                                                    hat_joint_ang, hat_joint_vel)
    



    print(x_list[0:3, k + 1])
    k=k+1



if __name__ == "__main__":
    rospy.init_node("main_node")
    rospy.loginfo("main_node init success")

    # sub_IMU_data = message_filters.Subscriber("/unitree_hardware/imu", Imu)
    # sub_joint=message_filters.Subscriber("/unitree_hardware/joint_foot", JointState)
    sub_IMU_data = message_filters.Subscriber("/trunk_imu", Imu)
    sub_joint=message_filters.Subscriber("/go1_gazebo/joint_states", JointState)

    ts = message_filters.ApproximateTimeSynchronizer([sub_IMU_data,sub_joint], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)

    rospy.spin()



