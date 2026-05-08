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

        self.arm_radius = np.array([10, 10, 10, 10, 10, 10])
        self.arm_length = np.linalg.norm(self.parameters[:, 1:3], axis=1)

        self.theta = self.parameters[:, 0]

        self.cr = CollisionRobot(self.parameters)
        self.dr = DHRobot(self.parameters)


    def parameters_set(self, parameters):
        self.parameters = parameters


    def forward_kinematics(self, theta = None):
        
        if theta is None:
            theta = self.theta

        dr_posture = self.dr.forward_kinematics(theta)
        cr_posture = self.cr.forward_kinematics(theta)
        
        return dr_posture, cr_posture

