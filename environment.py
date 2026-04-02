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
        self.action_bound = 5.0

        self.distance_error_threshold = 0.010 # 距离误差阈值
        self.angles_error_threshold = 1.0
        self._lo = self.robot.theta_limits[:, 0].astype(float)
        self._hi = self.robot.theta_limits[:, 1].astype(float)
        self._range = self._hi - self._lo
        self._prev_dist_norm = 0.0
        self.use_random_reset = False
        
    def initial_set(self, initial_angles, target_angles):
        
        self.initial_angles = initial_angles.astype(float)
        self.target_angles = target_angles.astype(float)
        self.angles = self.initial_angles.copy()

        return 0

    def train_reset(self):
        if self.use_random_reset:
            self.theta = np.zeros(6, dtype=float)
            self.target = np.zeros(6, dtype=float)
            for i in range(6):
                theta_limit = self.robot.theta_limits[i]
                self.theta[i] = np.random.uniform(theta_limit[0], theta_limit[1])
                self.target[i] = np.random.uniform(theta_limit[0], theta_limit[1])
        else:
            self.theta = self.initial_angles.copy().astype(float)
            self.target = self.target_angles.copy().astype(float)
        theta_error = self.target - self.theta

        self.step_count = 0
        self._prev_dist_norm = np.linalg.norm(theta_error / self._range)

        return self._get_state(theta_error)
        

    # def _get_state(self, theta_error):
    #     return np.concatenate([self.theta, self.target, theta_error])

    def _get_state(self, angles_error):
        norm_theta = 2.0 * (self.theta - self._lo) / self._range - 1.0
        norm_target = 2.0 * (self.target - self._lo) / self._range - 1.0
        norm_angles_error = angles_error / self._range
        return np.concatenate([norm_theta, norm_target, norm_angles_error]).astype(np.float32)


    def step(self, action):
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -self.action_bound, self.action_bound)
        self.theta = self.theta + action
        self.theta = np.clip(self.theta, self._lo, self._hi)

        self.step_count += 1


        # ====================================================
        # 重中之重，最需要调整的地方
        # ====================================================
        angles_error = self.target - self.theta
        norm_angles_error = np.linalg.norm(angles_error / self._range)
        shaping = 20.0 * (self._prev_dist_norm - norm_angles_error)
        self._prev_dist_norm = norm_angles_error
        action_penalty = 0.01 * np.linalg.norm(action)
        reward = shaping - action_penalty

        done = False
        success = False
        if self.arrive_detect(self.theta, self.target): # 到达目标位姿
            reward += 200.0
            done = True
            success = True

        if self.step_count >= self.max_steps: # 移动次数超出要求
            reward -= 5.0
            done = True
            success = False

        return self._get_state(angles_error), float(reward), done, success

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

        angles_error = target_angles - angles

        if np.any(np.abs(angles_error) > self.angles_error_threshold):
            return 0
        return 1

    def collision_detect(self):
        return 0
