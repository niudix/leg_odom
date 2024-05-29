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

def save_data_to_csv(data_list, filename):
    # 构建完整的文件路径
    current_directory = os.getcwd()  
    save_path = os.path.join(current_directory, "../output", filename)  # 构建目标路径

    # 确保目标文件夹存在，如果不存在，则创建它
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 将数据转换为numpy数组
    data_array = np.array(data_list)
    # 保存到CSV文件
    np.savetxt(save_path, data_array, delimiter=",")



step_count=0
def callback(imu_data, joint_state):

    global k  # 全局变量k
    global step_count
    global time_before
    global x_list
    global cov_list


    # # reorder the data to meet the requirement of algorithm
    if (step_count==0):
        time_before=imu_data.header.stamp
        step_count=step_count+1
        X0 = param_init.Init.init_x     #init_x=0
        P0 = param_init.Init.init_cov * np.eye(param_init.Param.state_size)
        P0[21:27, 21:27] = param_init.Init.init_bias_cov * np.eye(6)
        x_list = []
        cov_list = []
        x_list.append(np.squeeze(X0))
        cov_list.append(P0)
        k=0

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
    reordered_contact_force = [contact_force[1], contact_force[0], contact_force[3], contact_force[2]]
    contact_mode = np.array([int(force > 150) for force in reordered_contact_force])

    x_last = x_list[-1]  # Access the last element of x_list
    cov_last = cov_list[-1]  # Access the last element of cov_list
        
    x_next, cov_next  = LO.Lege_odom_func(x_last, uk, contact_mode, cov_last,
                                                                    hat_joint_ang, hat_joint_vel)
    
    x_list.append(x_next)
    cov_list.append(cov_next)


    print(x_next[0:3])
    k=k+1



if __name__ == "__main__":
    rospy.init_node("main_node")
    rospy.loginfo("main_node init success")

    sub_IMU_data = message_filters.Subscriber("/Hardware_IMU", Imu)
    sub_joint=message_filters.Subscriber("/joint_foot", JointState)

    ts = message_filters.ApproximateTimeSynchronizer([sub_IMU_data,sub_joint], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)

    rospy.spin()

    save_data_to_csv(x_list, "output_x_list.csv")



