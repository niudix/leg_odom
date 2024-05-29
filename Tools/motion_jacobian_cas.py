import casadi as ca
import numpy as np

# auto generated funtion by matlab
ox = [0.1805, 0.1805, -0.1805, -0.1805]
oy = [0.047, -0.047, 0.047, -0.047]
d = [0.0838, -0.0838, 0.0838, -0.0838]
rho_fix_size = 4
rho_opt_size = 1

lc = 0.20
lt = 0.20

rho_fix = ca.MX.zeros((rho_fix_size, 4))
# rho_fix[:, 0] = [[ox[0]], [oy[0]], [d[0]], [lt]]
# rho_fix[:, 1] = [[ox[1]], [oy[1]], [d[1]], [lt]]
# rho_fix[:, 2] = [[ox[2]], [oy[2]], [d[2]], [lt]]
# rho_fix[:, 3] = [[ox[3]], [oy[3]], [d[3]], [lt]]
for i in range(rho_fix_size):
    rho_fix[:, i] = [ox[i], oy[i], d[i], lt]

rho_opt_true = ca.MX.zeros((rho_opt_size, 4))
rho_opt_true[:, 0] = [lc]
rho_opt_true[:, 1] = [lc]
rho_opt_true[:, 2] = [lc]
rho_opt_true[:, 3] = [lc]


def motion_jac(joint_ang: ca.MX, leg_idx: int) -> ca.MX:
    lc = rho_opt_true[:, leg_idx]
    in3 = rho_fix[:, leg_idx]
    d = in3[2, 0]
    lt = in3[3, 0]
    t1 = joint_ang[0, :]
    t2 = joint_ang[1, :]
    t3 = joint_ang[2, :]
    t5 = ca.MX.cos(t1)
    t6 = ca.MX.cos(t2)
    t7 = ca.MX.cos(t3)
    t8 = ca.MX.sin(t1)
    t9 = ca.MX.sin(t2)
    t10 = ca.MX.sin(t3)
    t11 = t2 + t3
    t12 = ca.MX.cos(t11)
    t13 = lt * t9
    t14 = ca.MX.sin(t11)
    t15 = lc * t12
    t16 = lc * t14
    t17 = -t15
    t18 = t13 + t16
    jacobian_mtx = ca.vertcat(0.0, -d * t8 + lt * t5 * t6 + lc * t5 * t6 * t7 - lc * t5 * t9 * t10,
                              d * t5 + lt * t6 * t8 + lc * t6 * t7 * t8 - lc * t8 * t9 * t10, t17 - lt * t6, -t8 * t18,
                              t5 * t18, t17, -t8 * t16, t5 * t16)
    jacobian_mtx = ca.MX.reshape(jacobian_mtx, (3, 3))
    return jacobian_mtx
