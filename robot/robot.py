#引入模块----------------------------------
import numpy as np
#------------------------------------------
#自定义模块--------------------------------
from robot.collision_robot import CollisionRobot
from robot.DH_robot import DHRobot
#------------------------------------------

# 主体-------------------------------------

class Robot:

    def __init__(self):

        # DH参数
        # 采用改进版DH模型
        # theta,   d,    a,       alpha
        self.parameters = np.array([
            [0, 141.32,  0,       0   ],
            [0, 0,       0,       90  ],
            [0, 0,       596.02,  0   ],
            [0, -131.97, 571.97,  0   ],
            [0, 115.23,  0,       90  ],
            [0, 104.39,  0,       -90 ]
        ], dtype=float)
        
        # theta限制
        self.theta_limits = np.array([
            [-360, 360],
            [-85 , 265],
            [-175, 175],
            [-85 , 265],
            [-360, 360],
            [-360, 360]
        ])

        self.cr = CollisionRobot(self.parameters)
        self.dr = DHRobot(self.parameters)


    def parameters_set(self, parameters):
        self.parameters = parameters


    def forward_kinematics(self, theta):
        
        # for i in range(6): # 这一部分后续应该添加到惩罚函数中
        #     if theta[i] < self.theta_limits[i][0] or theta[i] > self.theta_limits[i][1]:
        #         print(f"第{i + 1}个关节角超出范围！")

        dr_posture = self.dr.forward_kinematics(theta)
        cr_posture = self.cr.forward_kinematics(theta)
        

        return dr_posture, cr_posture

