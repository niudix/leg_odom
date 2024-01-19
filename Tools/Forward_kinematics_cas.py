import numpy as np
import casadi as ca

# auto generated funtion by matlab
ox = [0.1805, 0.1805, -0.1805, -0.1805]
oy = [0.047, -0.047, 0.047, -0.047]
d = [0.0838, -0.0838, 0.0838, -0.0838]
rho_fix_size = 4
lc = 0.20
lt = 0.20

rho_fix = ca.MX.zeros((rho_fix_size, 4))
# rho_fix[:, 0] = [[ox[0]], [oy[0]], [d[0]], [lt]]
# rho_fix[:, 1] = [[ox[1]], [oy[1]], [d[1]], [lt]]
# rho_fix[:, 2] = [[ox[2]], [oy[2]], [d[2]], [lt]]
# rho_fix[:, 3] = [[ox[3]], [oy[3]], [d[3]], [lt]]
for i in range(rho_fix_size):
    rho_fix[:, i] = [ox[i], oy[i], d[i], lt]


def fK_function(angle: ca.MX, leg_idx: int) -> ca.MX:
    d = rho_fix[2, leg_idx]
    lt = rho_fix[3, leg_idx]
    ox = rho_fix[0, leg_idx]
    oy = rho_fix[1, leg_idx]
    t1 = angle[0]
    t2 = angle[1]
    t3 = angle[2]
    t5 = ca.cos(t1)
    t6 = ca.cos(t2)
    t7 = ca.cos(t3)
    t8 = ca.sin(t1)
    t9 = ca.sin(t2)
    t10 = ca.sin(t3)
    p_bf = ca.vertcat(ox - lt * t9 - lc * ca.sin(t2 + t3),
                      oy + d * t5 + lt * t6 * t8 + lc * t6 * t7 * t8 - lc * t8 * t9 * t10,
                      d * t8 - lt * t5 * t6 - lc * t5 * t6 * t7 + lc * t5 * t9 * t10)

    return p_bf

