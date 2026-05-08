#引入模块----------------------------------
import numpy as np
from torch import norm
from utils.logger import logger
#------------------------------------------
#自定义模块--------------------------------
from robot.robot import Robot
from environment.collision_environment import CollisionEnvironment
#------------------------------------------
#注意事项----------------------------------
# 1.训练中的关节初始角度和目标角度用的是theta和target，实际使用时用的是initial_angles和target_angles
# 2.角度应该限制在360度范围以内
# 3.训练环境应该分为：固定点、随机点、课程学习
#------------------------------------------
# 主体-------------------------------------

class Environment:
    def __init__(self):

        self.robot = Robot()
        self.ce = CollisionEnvironment()

        self.initial_angles = np.array([0, 0, 0, 0, 0, 0])
        self.target_angles = np.array([-80.339, 64.029, -49.259, 164.373, -280.029, -22.804])
        # self.target_angles = np.array([-121.464, 53.761, -33.861, 161.560, -237.948, -5.135])
        self.theta = self.initial_angles.copy()
        self.trajectory = np.array([]) # 用于储存关节角度轨迹

        # 步数与步长
        self.max_steps = 4000
        self.step_count = 0

        # 智能体维度
        # 状态维度随第一次初始化确定
        self.state_dim = self._get_state(self.theta, self.target_angles).shape[0]
        self.action_dim = 6

        # 误差阈值
        self.distance_error_threshold = 0.10
        self.angles_error_threshold = 1

        # 关节角度范围
        self._lo = self.robot.theta_limits[:, 0].astype(float)
        self._hi = self.robot.theta_limits[:, 1].astype(float)
        self._range = self._hi - self._lo

        # 课程学习参数
        self.use_random_reset = False
        self.curriculum_stage = 0  # 当前课程阶段
        self.curriculum_difficulty = 0.1  # 当前难度系数 [0, 1]
        self.success_count = 0  # 成功计数器
        self.curriculum_update_threshold = 10  # 成功多少次后增加难度
        
    # -------------------------------------------------------------------------------
    # 固定点训练
    # def stationary_point_train_reset(self, initial_angles: np.ndarray = None, target_angles: np.ndarray = None):

    #     if initial_angles is None:
    #         initial_angles = self.initial_angles
    #     if target_angles is None:
    #         target_angles = self.target_angles
        
    #     self.initial = initial_angles.astype(float)
    #     self.target = target_angles.astype(float)
    #     self.theta = self.initial.copy()
        
    #     angle_error = self.target - self.theta
    #     state = self._get_state(angle_error)

    #     return state
    
    # 采用笛卡尔为状态的版本
    def stationary_point_train_reset(self, initial_angles: np.ndarray = None, target_angles: np.ndarray = None):

        if initial_angles is None:
            initial_angles = self.initial_angles
        if target_angles is None:
            target_angles = self.target_angles
        
        self.initial = initial_angles.astype(float)
        self.target = target_angles.astype(float)
        self.theta = self.initial.copy()
        
        state = self._get_state(self.theta, self.target)

        return state
    
    def stationary_point_step(self, action):
        
        action = np.asarray(action, dtype=float)
        self.theta = self.theta + action
        self.step_count += 1
        self.theta = self.theta.astype(float)

        target_posture, _ = self.robot.forward_kinematics(self.target)
        current_posture, _ = self.robot.forward_kinematics(self.theta)
        initial_posture, _ = self.robot.forward_kinematics(self.initial_angles)
        target_point = target_posture[:3]
        current_point = current_posture[:3]
        initial_point = initial_posture[:3]
        distance_error = np.linalg.norm(target_point - current_point)
        distance_range = np.linalg.norm(target_point - initial_point)
        norm_distance_error = distance_error / distance_range

        self._prev_dist_norm = 0.0
        reward = 0.0

        done = False
        success = False

        # --------------------------------------------------------------------------------
        # 奖励函数

        # 关节角度限制，并加以惩罚
        # 如果超出范围，奖励减5并回退角度
        for i in range(6):
            if self.theta[i] < self._lo[i] or self.theta[i] > self._hi[i]:
                reward -= 5.0
                self.theta[i] -= action[i]

        # 改进奖励函数：使用连续的奖励塑形，而非二分类奖励
        angles_error = np.abs(self.target - self.theta)
        norm_angles_error = np.linalg.norm(angles_error / self._range) # 大角度

        # 1. 距离改善奖励（连续值，提供更稳定的梯度）
        # distance_improvement = self._prev_dist_norm - norm_angles_error
        # reward += 1 * distance_improvement # 放大奖励信号

        # 2. 潜在奖励塑形（基于距离的奖励）
        # reward += -1.0 * norm_angles_error # 距离越近奖励越高
        
        # # 3. 动作平滑性惩罚（避免剧烈抖动）
        # if hasattr(self, '_prev_action'):
        #     action_smoothness = -0.1 * np.linalg.norm(action - self._prev_action)
        #     reward += action_smoothness

        # 距离奖励
        reward += -1.0 * norm_distance_error # 距离越近奖励越高

        # 4. 碰撞惩罚
        wether_collision, min_dist = self.ce.collision_detect(self.theta)
        if wether_collision:
            reward -= 100.0
            done = True
            success = False
        else:
            # 如果没有障碍物，则输出没有碰撞且奖励dist为0，则奖励为0
            reward -= 0.01 * min_dist # 距离越近奖励越低

        self._prev_action = action.copy()
        
        self._prev_dist_norm = norm_angles_error

        if self.arrive_detect(self.theta, self.target): # 到达目标位姿
            reward += 200.0
            done = True
            success = True

        # --------------------------------------------------------------------------------

        if self.step_count >= self.max_steps: # 移动次数超出要求
            done = True
            success = False

        return self._get_state(self.theta, self.target), float(reward), done, success

    # -------------------------------------------------------------------------------
    # 随机点训练
    def random_point_train_reset(self, max_distance_threshold = 20):
        
        self.initial = np.random.uniform(self._lo, self._hi, size=6)

        low = np.maximum(self._lo, self.initial - max_distance_threshold)
        high = np.minimum(self._hi, self.initial + max_distance_threshold)

        self.target = np.random.uniform(low, high)

        self.theta = self.initial.copy()

        return 0
    
    def random_point_step(self, action):
        action = np.asarray(action, dtype=float)
        self.theta = self.theta + action
        reward = 0.0

        # ====================================================
        # 奖励函数
        # ====================================================

        # 关节角度限制，并加以惩罚
        # 如果超出范围，奖励减5并回退角度
        for i in range(6):
            if self.theta[i] < self._lo[i] or self.theta[i] > self._hi[i]:
                reward -= 5.0
                self.theta[i] -= action[i]

        self.theta = self.theta.astype(float)

        self.step_count += 1

        # 改进奖励函数：使用连续的奖励塑形，而非二分类奖励
        # angles_error = self.target - self.theta
        angles_error = np.abs(self.target - self.theta)

        norm_angles_error = np.linalg.norm(angles_error / self._range) # 大角度

        # 1. 距离改善奖励（连续值，提供更稳定的梯度）
        distance_improvement = self._prev_dist_norm - norm_angles_error
        reward += 1 * distance_improvement # 放大奖励信号

        # 2. 潜在奖励塑形（基于距离的奖励）
        # reward += -1.0 * norm_angles_error # 距离越近奖励越高
        
        # # 3. 动作平滑性惩罚（避免剧烈抖动）
        # if hasattr(self, '_prev_action'):
        #     action_smoothness = -0.1 * np.linalg.norm(action - self._prev_action)
        #     reward += action_smoothness
        self._prev_action = action.copy()
        
        self._prev_dist_norm = norm_angles_error

        done = False
        success = False

        # 新一种reward，依据最终与目标角度误差计算奖励
        if self.arrive_detect(self.theta, self.target): # 到达目标位姿
            reward += 20.0
            done = True
            success = True

        if self.step_count >= self.max_steps: # 移动次数超出要求
            done = True
            success = False

        return self._get_state(angles_error), float(reward), done, success

    # -------------------------------------------------------------------------------
    # 随机点训练
    def course_study_train_reset(self, max_angle_threshold = 20):
        if self.use_random_reset:
            self.theta = np.zeros(6, dtype=float)
            self.target = np.zeros(6, dtype=float)
            
            theta_distance = np.zeros(6, dtype=float)

            # 课程学习：根据难度系数生成初始点和目标点
            for i in range(6):
                theta_limit = self.robot.theta_limits[i]
                theta_range = theta_limit[1] - theta_limit[0]
                
                # 难度越高，初始点分布范围越大
                theta_center = (theta_limit[0] + theta_limit[1]) / 2
                theta_spread = theta_range * self.curriculum_difficulty * 0.5
                self.theta[i] = np.random.uniform(
                    theta_center - theta_spread,
                    theta_center + theta_spread
                )
                
                # 目标点距离初始点的距离也受难度控制
                max_distance = max_angle_threshold * self.curriculum_difficulty
                distance = np.random.uniform(0, max_distance)
                direction = np.random.choice([-1, 1])
                self.target[i] = np.clip(
                    self.theta[i] + direction * distance,
                    theta_limit[0],
                    theta_limit[1]
                )
                # theta_distance记录每个关节的目标角度与初始角度的距离
                theta_distance[i] = self.target[i] - self.theta[i]

            # 求二范数
            distance_l2 = np.linalg.norm(theta_distance)

        else:
            self.theta = self.initial_angles.copy().astype(float)
            self.target = self.target_angles.copy().astype(float)
        theta_error = self.target - self.theta

        self.step_count = 0
        self._prev_dist_norm = np.linalg.norm(theta_error / distance_l2)
        
        # 清除上一步的动作记录
        if hasattr(self, '_prev_action'):
            delattr(self, '_prev_action')

        return self._get_state(theta_error), distance_l2

    def course_study_step(self, action, distance_l2):
        action = np.asarray(action, dtype=float)
        self.theta = self.theta + action
        # self.theta = np.clip(self.theta, self._lo, self._hi)
        reward = 0.0

        # ====================================================
        # 重中之重，最需要调整的地方
        # ====================================================

        # 关节角度限制，并加以惩罚
        # 如果超出范围，奖励减5并回退角度
        for i in range(6):
            if self.theta[i] < self._lo[i] or self.theta[i] > self._hi[i]:
                reward -= 5.0
                self.theta[i] -= action[i]

        self.theta = self.theta.astype(float)

        self.step_count += 1

        # 改进奖励函数：使用连续的奖励塑形，而非二分类奖励
        # angles_error = self.target - self.theta
        angles_error = np.abs(self.target - self.theta)

        norm_angles_error = np.linalg.norm(angles_error / distance_l2) # 小角度
        # norm_angles_error = np.linalg.norm(angles_error / self._range) # 大角度

        # 1. 距离改善奖励（连续值，提供更稳定的梯度）
        distance_improvement = self._prev_dist_norm - norm_angles_error
        reward += 1 * distance_improvement # 放大奖励信号

        # 2. 潜在奖励塑形（基于距离的奖励）
        # reward += -1.0 * norm_angles_error # 距离越近奖励越高
        
        # # 3. 动作平滑性惩罚（避免剧烈抖动）
        # if hasattr(self, '_prev_action'):
        #     action_smoothness = -0.1 * np.linalg.norm(action - self._prev_action)
        #     reward += action_smoothness
        self._prev_action = action.copy()
        
        self._prev_dist_norm = norm_angles_error

        done = False
        success = False

        # 新一种reward，依据最终与目标角度误差计算奖励
        if self.arrive_detect(self.theta, self.target): # 到达目标位姿
            reward += 200.0
            done = True
            success = True

        if self.step_count >= self.max_steps: # 移动次数超出要求
            done = True
            success = False
        
        # 更新课程学习进度
        if success:
            self.success_count += 1
            if self.success_count >= self.curriculum_update_threshold:
                self.curriculum_difficulty = min(1.0, self.curriculum_difficulty + 0.1)
                self.curriculum_stage += 1
                self.success_count = 0
                logger.info(f"课程难度提升至: {self.curriculum_difficulty:.2f}")

        return self._get_state(angles_error), float(reward), done, success

    # -------------------------------------------------------------------------------
    # 其他函数

    # def _get_state(self):
    #     norm_theta = 2.0 * (self.theta - self._lo) / self._range - 1.0
    #     norm_target = 2.0 * (self.target - self._lo) / self._range - 1.0
    #     norm_angles_error = angles_error / self._range
    #     return np.concatenate([norm_theta, norm_target, norm_angles_error]).astype(np.float32)

    def _get_state(self, theta, target):
        
        target_posture, _ = self.robot.forward_kinematics(target)
        now_posture, _ = self.robot.forward_kinematics(theta)
        wether_collision, min_dist = self.ce.collision_detect(theta)

        flange_posture = now_posture[6]
        target_flange_posture = target_posture[6]

        return np.concatenate([flange_posture, target_flange_posture, [wether_collision, min_dist]]).astype(np.float32)

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
            return False
        return True

