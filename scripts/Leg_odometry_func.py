import casadi as cas
import numpy as np

import os
import sys

sys.path.append('../Tools')
import param_init
import cas_init
import Data_Import

print("Current Working Directory:", os.getcwd())


def Lege_odom_func(x: np.ndarray, uk: np.ndarray, contact_mode: np.ndarray, cov_p: np.ndarray, joint_ang: np.ndarray,
                   joint_vel: np.ndarray) -> np.ndarray:
    dt = uk[-1, 0]
    X01 = np.array(cas_init.next_state_func(x, uk, dt)) # state of next time (from rk4)
    F = np.array(cas_init.process_jacobian_func(x, uk, dt))
    B = np.array(cas_init.control_jacobian_func(x, uk, dt))

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
    contact = contact_mode
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

    P01 = F @ cov_p @ F.T + Q1 + B @ Q2 @ B.T
    # print(P01)
    hat_IMU_gyro = uk[0:3, 0]
    hat_joint_ang = joint_ang
    hat_joint_vel = joint_vel
    yk = np.array(
        cas_init.yk_Func(X01, hat_IMU_gyro, hat_joint_ang, hat_joint_vel))  # measurement residual(Eq:46)
    H = np.array(cas_init.H_jac_Func(X01, hat_IMU_gyro, hat_joint_ang, hat_joint_vel))
    # print(yk)
    S = H @ P01 @ H.T + R
    Kk = P01 @ H.T @ np.linalg.inv(S)
    update = Kk @ yk
    x_ext = np.squeeze(X01 - update)
    cov_ext = (np.eye(param_init.Param.state_size) - Kk @ H) @ P01

    return x_ext, cov_ext
