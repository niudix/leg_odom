import os
import numpy as np


# define robot motion data
Data_path = '../assets/SIPO_Data'
# Data_path = 'home/marmot/dixiao/catkin_legodom/src/legodom/assets/SIPO_Data'

file_Accel_body_IMU = os.path.join(Data_path, 'Accel_body_IMU.csv')
file_Gyro_body_IMU = os.path.join(Data_path, 'Gyro_body_IMU.csv')
file_Joint_ang = os.path.join(Data_path, 'Joint_ang.csv')
file_Joint_vel = os.path.join(Data_path, 'Joint_vel.csv')
file_Contact_mode = os.path.join(Data_path, 'Contact_mode.csv')
file_Orient_mocap_euler = os.path.join(Data_path, 'Orient_mocap_euler.csv')
file_Pos_mocap = os.path.join(Data_path, 'Pos_mocap.csv')

Accel_body_IMU = np.loadtxt(file_Accel_body_IMU, delimiter=',', skiprows=1)
Gyro_body_IMU = np.loadtxt(file_Gyro_body_IMU, delimiter=',', skiprows=1)
Joint_ang = np.loadtxt(file_Joint_ang, delimiter=',', skiprows=1)
Joint_vel = np.loadtxt(file_Joint_vel, delimiter=',', skiprows=1)
Contact_mode = np.loadtxt(file_Contact_mode, delimiter=',', skiprows=1)
Orient_mocap_euler = np.loadtxt(file_Orient_mocap_euler, delimiter=',', skiprows=1)
Pos_mocap = np.loadtxt(file_Pos_mocap, delimiter=',', skiprows=1)
# print(Pos_mocap)
