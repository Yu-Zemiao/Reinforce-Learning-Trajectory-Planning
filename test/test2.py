# ================================
# 这个demo负责反向求解六轴DH参数
# ================================
import numpy as np

def inverse_d_a(end_pose, joint_angles_deg, alpha_deg):
    """
    简化demo: 反向求解六轴机械臂的 DH 参数 d 和 a

    参数:
    ----------
    end_pose : [X, Y, Z, Rx, Ry, Rz]  # 末端位姿，欧拉角 ZYX
    joint_angles_deg : [J1,J2,J3,J4,J5,J6]  # 关节角度，degree
    alpha_deg : [alpha1,...,alpha6]  # DH扭转角，degree

    返回:
    ----------
    d : (6,) numpy array
    a : (6,) numpy array
    """
    J = np.deg2rad(joint_angles_deg)
    alpha = np.deg2rad(alpha_deg)
    
    # 构建末端旋转矩阵
    rx, ry, rz = np.deg2rad(end_pose[3:6])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    R_end = np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx]
    ])
    P_end = np.array(end_pose[:3])
    
    # 初始化 d 和 a
    d = np.zeros(6)
    a = np.zeros(6)
    
    # 简化假设：每个关节只沿z平移(d)和x平移(a)
    # 从第一个关节开始迭代
    T = np.eye(4)
    
    for i in range(6):
        # Ti = [[R, p],[0,1]] = [[cosθ, -sinθ cosα, sinθ sinα, a cosθ],
        #                        [sinθ, cosθ cosα, -cosθ sinα, a sinθ],
        #                        [0, sinα, cosα, d],
        #                        [0,0,0,1]]
        # 利用 Ti @ ... = T_end 做线性方程求解 a,d
        # 对demo: 只计算 d[i] = z方向差, a[i] = xy平面距离投影
        
        # 当前总旋转矩阵
        R_prev = T[:3,:3]
        P_prev = T[:3,3]
        
        # 投影到前一关节坐标系
        P_local = np.linalg.inv(R_prev) @ (P_end - P_prev)
        
        d[i] = P_local[2]  # z方向
        a[i] = np.linalg.norm(P_local[:2])  # xy平面距离
        
        # 更新当前T，用简化Ti
        theta = J[i]
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,0,1]
        ])
        Rx = np.array([
            [1,0,0],
            [0,np.cos(alpha[i]), -np.sin(alpha[i])],
            [0,np.sin(alpha[i]), np.cos(alpha[i])]
        ])
        T_local = np.eye(4)
        T_local[:3,:3] = Rz @ Rx
        T_local[:3,3] = [a[i],0,d[i]]
        
        T = T @ T_local
    
    return d, a


# ======= DEMO 使用 =======
end_pose = [266.331, -886.461, 936.642, -90.917, -22.951, 179.988]  # X,Y,Z,Rx,Ry,Rz
joint_angles = [-80.339, 64.029, -49.259, 164.373, -280.029, -22.804]
alpha = [90, 0, 0, 90, -90, 0]

d, a = inverse_d_a(end_pose, joint_angles, alpha)

print("d =", d)
print("a =", a)