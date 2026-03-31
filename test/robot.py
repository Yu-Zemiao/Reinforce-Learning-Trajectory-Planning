#引入模块----------------------------------
import numpy as np
#------------------------------------------
#自定义模块--------------------------------

#------------------------------------------

# 主体-------------------------------------

class Robot:

    def __init__(self):

        # DH参数
        # theta,   d,    a,       alpha
        self.parameters = np.array([# 这个参数目前测下来是最正确的
            [0, 141.32,  0,       0   ],
            [0, 0,       0,       90  ],
            [0, 0,       596.02,  0   ],
            [0, -131.97, 571.97,  0   ],
            [0, 115.23,  0,       90  ],
            [0, 104.39,  0,       -90 ]
        ], dtype=float)

        # 双向绑定，修改parameters会同步修改theta，修改theta会同步修改parameters
        self.theta = self.parameters[:, 0]
        self.d     = self.parameters[:, 1]
        self.a     = self.parameters[:, 2]
        self.alpha = self.parameters[:, 3]

        self.theta_limit = np.array([ # 限制范围
            [-360, 360],
            [-85 , 265],
            [-175, 175],
            [-85 , 265],
            [-360, 360],
            [-360, 360]
        ], dtype=float)

        self.posture = np.array([ # 7个点的坐标与姿态，必须调用前向传播后使用，第一个为机器人基坐标系原点位置与姿态
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=float)

        self.points = np.array([ # 用于表达7个点的坐标，不表示姿态，必须调用后使用
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=float)

        # 添加一个轴体宽度，用于后续碰撞检测
        self.axial_body_width = np.array([
            10,
            10,
            10,
            10,
            10,
            10
        ], dtype=float)

    # points和posture前三列绑定
    @property
    def points(self):
        return self.posture[:, :3]
    
    @points.setter
    def points(self, value):
        self.posture[:, :3] = value

    # theta 绑定
    @property
    def theta(self):
        return self.parameters[:, 0]

    @theta.setter
    def theta(self, value):
        self.parameters[:, 0] = value

    # d 绑定
    @property
    def d(self):
        return self.parameters[:, 1]

    @d.setter
    def d(self, value):
        self.parameters[:, 1] = value

    # a 绑定
    @property
    def a(self):
        return self.parameters[:, 2]

    @a.setter
    def a(self, value):
        self.parameters[:, 2] = value

    # alpha 绑定
    @property
    def alpha(self):
        return self.parameters[:, 3]

    @alpha.setter
    def alpha(self, value):
        self.parameters[:, 3] = value
    

    # 前向传播
    # 在这个函数中，本来想做到根据第0个点的位姿转变整个机器人的位姿
    # 但是考虑到暂时用不上，且实现起来有些问题，暂时先不做了
    def forward_kinematics(self):

        frames = []

        # # 2. 构建包含基座位姿的初始齐次变换矩阵 T
        T = np.eye(4)

        # DH参数
        theta = np.deg2rad(self.parameters[:, 0])
        d = self.parameters[:, 1]
        a = self.parameters[:, 2]
        alpha = np.deg2rad(self.parameters[:, 3])

        # 3. 依次计算后续 6 个关节
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
            self.points[i + 1] = pos
            self.posture[i + 1, :3] = pos
            self.posture[i + 1, 3:6] = np.rad2deg([Rx_angle, Ry_angle, Rz_angle])

            frames.append(T.copy())

        return frames # 这个frames可能有问题，输出的应该是弧度制，但是考虑到不用，不改了
    
    # 用于强化学习
    def set_joint_angles(self,theta):
        self.theta = theta
        self.forward_kinematics()
