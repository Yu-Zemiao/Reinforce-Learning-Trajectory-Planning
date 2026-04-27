#引入模块----------------------------------
import numpy as np
from torch import norm
from utils.logger import logger
#------------------------------------------
#自定义模块--------------------------------
from robot.robot import Robot
from environment.collision_environment import CollisionEnvironment
from environment.detect_environment import DetectEnvironment
#------------------------------------------
#注意事项----------------------------------
# 1.训练中的关节初始角度和目标角度用的是theta和target，实际使用时用的是initial_angles和target_angles
# 2.角度应该限制在360度范围以内
#------------------------------------------
# 主体-------------------------------------

class Environment:
    def __init__(self):
        self.robot = Robot()
        self.ce = CollisionEnvironment()
        self.de = DetectEnvironment()

        # 训练参数
        self.state_dim = 18
        self.action_dim = 6

        # 环境参数
        self.initial_angles = np.array([0, 0, 0, 0, 0, 0])
        self.target_angles = np.array([0, 0, 0, 0, 0, 0])
        self.step_size = 0.1

