#引入模块----------------------------------
import numpy as np
from torch import norm
from utils.logger import logger
#------------------------------------------
#自定义模块--------------------------------
from robot.robot import Robot
#------------------------------------------
#注意事项----------------------------------

#------------------------------------------
# 主体-------------------------------------

class CollisionEnvironment:
    def __init__(self):
        self.robot = Robot()
        self.initial_angles = np.array([0, 0, 0, 0, 0, 0])
        self.target_angles = np.array([0, 0, 0, 0, 0, 0])
        self.step_size = 0.1
