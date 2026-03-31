#引入模块----------------------------------
import numpy as np
#------------------------------------------
#自定义模块--------------------------------

#------------------------------------------

# 主体-------------------------------------

class CollisionRobot:
    
    def __init__(self, parameters):
        # DH参数
        # theta,   d,    a,       alpha
        self.DH_parameters = parameters

    def forward_kinematics(self, theta):
        
        posture = [0, 0, 0, 0, 0, 0]

        return posture

