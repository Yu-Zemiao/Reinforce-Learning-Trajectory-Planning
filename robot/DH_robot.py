#引入模块----------------------------------
import numpy as np
#------------------------------------------
#自定义模块--------------------------------

#------------------------------------------

# 主体-------------------------------------

class DHRobot:
    
    def __init__(self, parameters):
        
        # DH参数
        # theta,   d,    a,       alpha
        self.DH_parameters = parameters


    def forward_kinematics(self, theta):

        posture = np.zeros((7, 6))
        # 构建包含基座位姿的初始齐次变换矩阵T
        T = np.eye(4)

        # DH参数
        d = self.DH_parameters[:, 1]
        a = self.DH_parameters[:, 2]
        alpha = np.deg2rad(self.DH_parameters[:, 3])

        # 依次计算后续6个关节
        for i in range(6):

            # 这是改进的DH模型
            Ti = np.array([
                [np.cos(theta[i]),                   -np.sin(theta[i]),                    0,                  a[i]                     ],
                [np.sin(theta[i])*np.cos(alpha[i]),  np.cos(theta[i])*np.cos(alpha[i]),    -np.sin(alpha[i]),  -d[i]*np.sin(alpha[i])   ],
                [np.sin(theta[i])*np.sin(alpha[i]),  np.cos(theta[i])*np.sin(alpha[i]),    np.cos(alpha[i]),   d[i]*np.cos(alpha[i])    ],
                [0,                                  0,                                    0,                  1                        ]
            ])

            # 矩阵累乘：将当前关节的局部变换叠加到总变换上
            T = T @ Ti

            # === 提取位置 ===
            pos = T[:3, 3]

            # === 提取姿态 (防死锁解算) ===
            R = T[:3, :3] 
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            singular = sy < 1e-6 

            if not singular:
                Rx_angle = np.arctan2(R[2, 1], R[2, 2])
                Ry_angle = np.arctan2(-R[2, 0], sy)
                Rz_angle = np.arctan2(R[1, 0], R[0, 0])
            else:
                Rx_angle = np.arctan2(-R[1, 2], R[1, 1])
                Ry_angle = np.arctan2(-R[2, 0], sy)
                Rz_angle = 0

            # === 更新自身属性 ===
            posture[i + 1, :3] = pos
            posture[i + 1, 3:6] = np.rad2deg([Rx_angle, Ry_angle, Rz_angle])

        return posture

        

    