#!/usr/bin/env python3

import numpy as np
import os
import sys

sys.path.append('../Tools')
# sys.path.append('/home/marmot/dixiao/catkin_ws/src/odom/leg_odom/Tools')

print("Current Working Directory:", os.getcwd())

import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
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



step_count=0
def callback(imu_data, joint_state,FL_force_raw,FR_force_raw,RL_force_raw,RR_force_raw):

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
    new_order = [1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9]
    adjusted_joint_pos= [joint_state.position[i] for i in new_order]
    adjusted_joint_vel = [joint_state.velocity[i] for i in new_order]
    hat_joint_ang = np.array(adjusted_joint_pos)
    hat_joint_vel =np.array(adjusted_joint_vel)

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
    FL_force=FL_force_raw.wrench.force.z
    FR_force=FR_force_raw.wrench.force.z
    RL_force=RL_force_raw.wrench.force.z
    RR_force=RR_force_raw.wrench.force.z
    contact_force=np.array([FL_force,FR_force,RL_force,RR_force])
        
    contact_mode = np.array([int(abs(force >1) ) for force in contact_force])
    

    x_last = x_list[-1]  # Access the last element of x_list
    cov_last = cov_list[-1]  # Access the last element of cov_list
        
    x_next, cov_next  = LO.Lege_odom_func(x_last, uk, contact_mode, cov_last,
                                                                    hat_joint_ang, hat_joint_vel)
    
    x_list.append(x_next)
    cov_list.append(cov_next)


    print("position:",x_next[0:3])
    k=k+1



if __name__ == "__main__":
    rospy.init_node("main_node")
    rospy.loginfo("main_node init success")

    sub_IMU_data = message_filters.Subscriber("/trunk_imu", Imu)
    sub_joint=message_filters.Subscriber("/go1_gazebo/joint_states", JointState)
    sub_FL_force=message_filters.Subscriber("/visual/FL_foot_contact/the_force_timestamped", WrenchStamped)
    sub_FR_force=message_filters.Subscriber("/visual/FR_foot_contact/the_force_timestamped", WrenchStamped)
    sub_RL_force=message_filters.Subscriber("/visual/RL_foot_contact/the_force_timestamped", WrenchStamped)
    sub_RR_force=message_filters.Subscriber("/visual/RR_foot_contact/the_force_timestamped", WrenchStamped)
    
    
    ts = message_filters.ApproximateTimeSynchronizer([sub_IMU_data,sub_joint,sub_FL_force,sub_FR_force,sub_RL_force,sub_RR_force], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)

    rospy.spin()



