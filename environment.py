#引入模块----------------------------------
import numpy as np
#------------------------------------------
#自定义模块--------------------------------
from robot.robot import Robot
#------------------------------------------
#注意事项----------------------------------
# 1.训练中的关节初始角度和目标角度用的是theta和target，实际使用时用的是initial_angles和target_angles
# 2.角度应该限制在360度范围以内
#------------------------------------------
# 主体-------------------------------------

class Environment:
    def __init__(self):

        self.robot = Robot()

        self.initial_angles = np.array([0, 0, 0, 0, 0, 0])

        self.target_angles = np.array([-80.339, 64.029, -49.259, 164.373, -280.029, -22.804])
        # self.target_angles = np.array([-121.464, 53.761, -33.861, 161.560, -237.948, -5.135])

        self.angles = self.initial_angles.copy()

        self.max_steps = 1000

        self.state_dim = 18
        self.action_dim = 6

        self.distance_error_threshold = 0.010 # 距离误差阈值
        self.angles_error_threshold = 0.10 # 各关节角度误差阈值
        
    def initial_set(self, initial_angles, target_angles):
        
        self.initial_angles = initial_angles
        self.target_angles = target_angles
        self.angles = self.initial_angles.copy()

        return 0

    def train_reset(self):

        self.theta = np.zeros(6)
        self.target = np.zeros(6)

        for i in range(6):
            theta_limit = self.robot.theta_limits[i]
            self.theta[i] = np.random.uniform(theta_limit[0], theta_limit[1])
            self.target[i] = np.random.uniform(theta_limit[0], theta_limit[1])

        theta_error = self.target - self.theta

        self.step_count = 0

        return self._get_state(theta_error)
        

    # def _get_state(self, theta_error):
    #     return np.concatenate([self.theta, self.target, theta_error])

    # 新一版的_get_state，添加了归一化
    def _get_state(self, angles_error):
        # 假设 self.theta, self.target 等数组已经存在
        # 提取所有关节的上下限
        lower_bounds = np.array([limit[0] for limit in self.robot.theta_limits])
        upper_bounds = np.array([limit[1] for limit in self.robot.theta_limits])
        ranges = upper_bounds - lower_bounds
        
        # 1. 对 theta 进行归一化到 [-1, 1]
        norm_theta = 2.0 * (self.theta - lower_bounds) / ranges - 1.0
        
        # 2. 对 target 进行归一化到 [-1, 1]
        norm_target = 2.0 * (self.target - lower_bounds) / ranges - 1.0
        
        # 3. 对 angles_error 进行归一化
        # 误差的最大范围是 [-ranges, ranges]，可以除以 ranges 映射到 [-1, 1]
        norm_angles_error = angles_error / ranges
        
        # 最后拼接作为网络的绝对纯净、归一化后的状态输入
        return np.concatenate([norm_theta, norm_target, norm_angles_error])


    def step(self, action):
        last_theta = self.theta.copy()
        last_posture_error, _ = self.error_calculate(last_theta, self.target)
        last_posture_error = np.linalg.norm(last_posture_error)
        self.theta += action
        now_posture_error, _ = self.error_calculate(self.theta, self.target)

        # 这里后续要添加一个关节角度范围是否超出的检测

        self.step_count += 1

        # 这里可能需要添加一个归一化
        angles_error, posture_error = self.error_calculate(self.theta, self.target)
        norm_angles_error = np.linalg.norm(angles_error)
        norm_posture_error = np.linalg.norm(posture_error)

        # reward = -norm_angles_error**2 - norm_posture_error**2

        # reward = -(0.1 * norm_angles_error + 0.5 * norm_posture_error)

        reward = - norm_angles_error

        # 添加一个新的reward计算方式，提高每一步中靠近最终结果时的奖励
        # 采用一种强激励，为每一次动作都加入奖励
        # 当姿态误差减少时，奖励增加；当姿态误差增加时，奖励减少
        # if norm_posture_error < last_posture_error:
        #     reward += 5
        # elif norm_posture_error > last_posture_error:
        #     reward -= 5

        done = False
        if self.arrive_detect(self.theta, self.target): # 到达目标位姿
            reward += 100
            done = True

        if self.step_count > self.max_steps: # 移动次数超出要求
            done = True

        return self._get_state(angles_error), reward, done

    # 计算姿态误差，分别输出为各关节角度误差，姿态误差
    def error_calculate(self, angles, target_angles):

        now_posture, _ = self.robot.forward_kinematics(angles)
        target_posture, _ = self.robot.forward_kinematics(target_angles)

        now_flange_posture = now_posture[6]
        target_flange_posture = target_posture[6]

        posture_error = target_flange_posture - now_flange_posture # 计算末端姿态误差
        
        angles_error = target_angles - angles # 计算角度误差

        return angles_error, posture_error
    
    def arrive_detect(self, angles, target_angles):

        angles_error, _ = self.error_calculate(angles, target_angles)

        if np.any(np.abs(angles_error) > self.angles_error_threshold):
            return 0
        return 1

    def collision_detect(self):
        return 0