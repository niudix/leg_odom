import casadi as ca
import param_init as parm
import Forward_kinematics_cas as fk_ca
import motion_jacobian_cas as mt_jac

s_xk = ca.MX.sym('X_k', parm.Param.state_size)  # 身体当前状态的状态矩阵
s_uk = ca.MX.sym('Uk', parm.Param.ctrl_size)  # IMU测得的值
s_dt = ca.MX.sym('dt', 1)
ca_hat_IMU_gyro = ca.MX.sym('ca_hat_IMU_gyro', 3)
ca_hat_joint_ang = ca.MX.sym('ca_hat_joint_ang', 12)
ca_hat_joint_vel = ca.MX.sym('ca_hat_joint_vel', 12)
# yawk = ca.MX.sym('yawk', 1)
euler_ang = ca.MX.sym('euler_ang', 3)


def euler_to_rot(euler: ca.MX) -> ca.MX:
    # listed by roll pitch yaw
    r = euler[0, 0]  # x axis
    p = euler[1, 0]  # y axis
    y = euler[2, 0]  # z axis

    row_r1 = ca.horzcat(1, 0, 0)
    row_r2 = ca.horzcat(0, ca.cos(r), -ca.sin(r))
    row_r3 = ca.horzcat(0, ca.sin(r), ca.cos(r))
    rot_r = ca.vertcat(row_r1, row_r2, row_r3)

    row_p1 = ca.horzcat(ca.cos(p), 0, ca.sin(p))
    row_p2 = ca.horzcat(0, 1, 0)
    row_p3 = ca.horzcat(-ca.sin(p), 0, ca.cos(p))
    rot_p = ca.vertcat(row_p1, row_p2, row_p3)

    row_y1 = ca.horzcat(ca.cos(y), -ca.sin(y), 0)
    row_y2 = ca.horzcat(ca.sin(y), ca.cos(y), 0)
    row_y3 = ca.horzcat(0, 0, 1)
    rot_y = ca.vertcat(row_y1, row_y2, row_y3)

    rot = rot_y @ rot_p @ rot_r
    return rot


def mtx_to_euler_dot(euler: ca.MX) -> ca.MX:
    """
        compute the matrix that can transfer euler to differentiation of euler(euler_dot)
        euler_dot = mtx * angular_velocity

        Parameters:
        euler (np.array)
        euler(1): roll
        euler(2): pitch
        euler(3): yaw
   """
    r = euler[0, 0]
    p = euler[1, 0]
    y = euler[2, 0]

    # 创建矩阵的每一行
    row1 = ca.horzcat(1, (ca.sin(p) * ca.sin(r)) / ca.cos(p), (ca.cos(r) * ca.sin(p)) / ca.cos(p))
    row2 = ca.horzcat(0, ca.cos(r), -ca.sin(r))
    row3 = ca.horzcat(0, ca.sin(r) / ca.cos(p), ca.cos(r) / ca.cos(p))

    # 垂直堆叠行以创建矩阵
    mtx = ca.vertcat(row1, row2, row3)

    return mtx


def x_dot_function(x: ca.MX, imu_data: ca.MX) -> ca.MX:
    """
        Computing first-order differentials for body postures

        Parameters:
        imu_data (np.array): Body acceleration and angular velocity measured by the IMU
        control imu_data
        - w      (1:3)      IMU angular velocity
        - a      (4:6)     IMU acceleration
        - hk     (7)

        x (np.array): param2 Body condition matrix
        x
        - position(1: 3)
        - velocity(4: 6)
        - euler angle(7: 9)
        - foot1(10: 12)
        - foot2(13: 15)
        - foot3(16: 18)
        - foot4(19: 21)
        - acc bias(22: 24)
        - gyro bias(25: 27)
        - time tk(28)

        Returns:
        x_dot: Differential matrix for body states
        x_dot
        - velocity
        - acc
        - deuler (Differentiation of Euler angles)
        - d foot
        - d acc bias
        - d gyro bias
        - hk

    """
    x = x.reshape((-1, 1))
    imu_data = imu_data.reshape((-1, 1))  # Make sure the arguments are column vectors

    # Obtaining Physical Condition
    pos = x[0:3, :]
    vel = x[3:6, :]
    euler = x[6:9, :]
    ba = x[21:24, :]  # acc bias
    bg = x[24:27, :]  # gyro bias
    w = imu_data[0:3, :] - bg
    a = imu_data[3:6, :] - ba

    euler_dot_mtx = mtx_to_euler_dot(euler)
    deuler = euler_dot_mtx @ w  # euler_dot = mtx * angular_velocity
    R = euler_to_rot(euler)  # rotation matrix
    acc = R @ a - ca.MX([0, 0, 9.8])
    x_dot = ca.vertcat(vel, acc, deuler, ca.MX.zeros(18, 1), ca.MX.ones(1, 1))

    return x_dot


def runge_kutta_4(x: ca.MX, imu_data: ca.MX, dt: ca.MX) -> ca.MX:
    k1 = x_dot_function(x, imu_data)
    k2 = x_dot_function(x + dt * k1 / 2, imu_data + dt / 2)
    k3 = x_dot_function(x + dt * k2 / 2, imu_data + dt / 2)
    k4 = x_dot_function(x + dt * k3, imu_data + dt)

    xn1 = x + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return xn1


pos = s_xk[0:3, :]
state_v = s_xk[3:6]
euler = s_xk[6:9]
R_er = euler_to_rot(euler)  # rotation mtx
foot_pos = s_xk[9:21]
bg = s_xk[24:27]

meas_per_leg = parm.Param.meas_per_leg  # 7( 3 3 1 )
meas_residual = ca.MX.zeros(parm.Param.meas_size, 1)

for i in range(parm.Param.num_leg):
    joint_ang = ca_hat_joint_ang[i * 3:i * 3 + 3, :]
    joint_vel = ca_hat_joint_vel[i * 3:i * 3 + 3, :]
    pos_f = fk_ca.fK_function(joint_ang, i)  # relative position of the feet by joint angles
    jac_motion = mt_jac.motion_jac(joint_ang, i)  # jacobian mtx (convert joint vel to foot vel)
    leg_v = jac_motion @ joint_vel + ca.skew(
        ca_hat_IMU_gyro - bg) @ pos_f  # foot_vel= speed( joint rotation)+speed(body_rotation)

    meas_residual[i * meas_per_leg: i * meas_per_leg + 3] = pos_f - R_er.T @ (
            foot_pos[i * 3:i * 3 + 3] - pos)
    meas_residual[i * meas_per_leg + 3: i * meas_per_leg + 6] = state_v + R_er @ leg_v
    meas_residual[i * meas_per_leg + 6] = foot_pos[i * 3 + 2]  # foot_height(foot position coordinate axis z)

    # meas_residual[-1]=注意：这里没有引用动作捕捉系统
    # meas_residual[-1] = yawk - euler[2]
# test--------------------------------------------------------


s_f = runge_kutta_4(s_xk, s_uk, s_dt)
process_jac = ca.jacobian(s_f, s_xk)
control_jac = ca.jacobian(s_f, s_uk)
H_jac = ca.jacobian(meas_residual, s_xk)
R_mtx = mtx_to_euler_dot(euler_ang)

next_state_func = ca.Function('next_state_func', [s_xk, s_uk, s_dt], [s_f])
process_jacobian_func = ca.Function('process_jacobian_func', [s_xk, s_uk, s_dt], [process_jac])
control_jacobian_func = ca.Function('control_jacobian_func', [s_xk, s_uk, s_dt], [control_jac])

yk_Func = ca.Function('yk_Func', [s_xk, ca_hat_IMU_gyro, ca_hat_joint_ang, ca_hat_joint_vel], [meas_residual])
H_jac_Func = ca.Function('H_jac_Func', [s_xk, ca_hat_IMU_gyro, ca_hat_joint_ang, ca_hat_joint_vel], [H_jac])
euler_to_rot_Func = ca.Function('euler_to_rot_Func', [euler_ang], [R_mtx])
