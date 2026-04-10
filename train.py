#引入模块----------------------------------
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import copy
import os
#------------------------------------------

#自定义模块--------------------------------
from robot.robot import Robot
from visiualization import Visiualization
from environment import Environment
from agent.PPO_agent import PPOAgent
from agent.SAC_agent import SACAgent
from read_and_write_file import ReadAndWritefile
from utils.logger import logger
#------------------------------------------

#注意事项--------------------------------
# 1.现在的输入：初始角度、目标角度、角度偏差
#------------------------------------------

# 主体-------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fileio = ReadAndWritefile()

current_dir = os.path.dirname(os.path.abspath(__file__))

reward_path = os.path.join(current_dir, "log", "data", "reward")
reward_path = os.path.join(reward_path, "reward.txt")
os.makedirs(os.path.dirname(reward_path), exist_ok=True)
write_reward_file_path = reward_path # type: ignore

training_path = os.path.join(current_dir, "log", "train")
best_training_parameters_path = os.path.join(training_path, "best_training_parameters.pt")
os.makedirs(os.path.dirname(best_training_parameters_path), exist_ok=True)
last_training_parameters_path = os.path.join(training_path, "last_training_parameters.pt")
os.makedirs(os.path.dirname(last_training_parameters_path), exist_ok=True)



class Train:
    
    def __init__(self, environment:Environment):
        self.agent = PPOAgent(
            state_dim=environment.state_dim,
            action_dim=environment.action_dim,
        )

        self.robot = Robot()
        self.max_episodes = 10_0000
        self.batch_size = 100
        self.environment = environment

        self.training_parameter_saving_path = None
        self.model_snapshots = []
        self.final_reward_memory = []

        self.loss_history = []

        self.the_smallest_lr_threshold = 1.0e-6
        
        # 定义测试集：10个代表性场景
        self.test_cases = [
            # 场景1：近距离
            (np.array([0, 0, 0, 0, 0, 0]), np.array([30, 30, 30, 30, 30, 30])),
            # 场景2：中距离
            (np.array([0, 0, 0, 0, 0, 0]), np.array([90, 60, -60, 90, -90, 60])),
            # 场景3：远距离
            (np.array([0, 0, 0, 0, 0, 0]), np.array([180, 120, -120, 180, -180, 120])),
            # 场景4：逆向
            (np.array([120, 90, -90, 120, -120, 90]), np.array([0, 0, 0, 0, 0, 0])),
            # 场景5：非零起点
            (np.array([60, 45, 30, 60, -45, 30]), np.array([-60, 90, -30, 120, -90, 60])),
            # 场景6：侧向
            (np.array([90, 0, 0, 0, 0, 90]), np.array([-90, 0, 0, 0, 0, -90])),
            # 场景7：极端角度
            (np.array([180, 150, -150, 180, -180, 150]), np.array([0, 0, 0, 0, 0, 0])),
            # 场景8：大角度变换
            (np.array([-90, -60, 60, -90, 90, -60]), np.array([90, 60, -60, 90, -90, 60])),
            # 场景9：多关节协同
            (np.array([45, 45, 45, 45, 45, 45]), np.array([-45, -45, -45, -45, -45, -45])),
            # 场景10：混合场景
            (np.array([120, 60, -30, 90, -60, 120]), np.array([-60, 120, 60, -90, 120, -60])),
        ]
        
        self.best_test_error = float('inf')  # 记录最优测试误差
        self.test_frequency = 20  # 每20个episode测试一次
        self.wether_test = 0 # 是否测试
    
    def evaluate_on_test_set(self):
        """
        在测试集上评估当前模型性能
        使用确定性策略，关闭梯度计算
        返回：平均角度误差
        """
        total_angle_error = 0.0
        test_env = Environment()  # 创建独立的环境用于测试
        
        with torch.no_grad():  # 关闭梯度计算
            for idx, (initial_angles, target_angles) in enumerate(self.test_cases):
                # 设置测试场景
                test_env.initial_set(initial_angles, target_angles)
                state = torch.FloatTensor(test_env.train_reset()).to(device)
                
                # 运行测试场景
                for step in range(test_env.max_steps):
                    # 使用确定性策略：直接用actor输出的均值，不采样
                    mean = torch.tanh(self.agent.policy.actor(state)) * self.agent.policy.action_bound
                    action = mean  # 确定性动作
                    
                    next_state, _, done, _ = test_env.step(action.detach().cpu().numpy())
                    state = torch.FloatTensor(next_state).to(device)
                    
                    if done:
                        break
                
                # 计算该场景的角度误差
                angle_error = np.linalg.norm(test_env.target - test_env.theta)
                total_angle_error += angle_error
                
                logger.info(f"  测试场景 {idx+1}: 初始={np.round(initial_angles, 1)}, 目标={np.round(target_angles, 1)}, 误差={angle_error:.3f}°")
        
        avg_angle_error = total_angle_error / len(self.test_cases)
        return avg_angle_error
        

    def train(self):
        rewards_history = []
        steps_history = []
        success_history = []

        last_best_loss = 1.0e+8
        now_loss = 0.0
        wether_update_lr_count = 0 # 如果episode达到一定次数没有更新，减小lr
        biggest_loss = 0.0

        for episode in range(self.max_episodes):
            logger.info(f"Episode {episode + 1}--------------------------------------------------------------------------------")
            # state = torch.FloatTensor(self.environment.train_reset()).to(device)

            state, max_distance = self.environment.small_angle_train_reset()
            state = torch.FloatTensor(state).to(device)

            initial_theta = self.environment.theta
            episode_reward = 0.0
            episode_steps = 0
            success = False

            update_count = 0

            
            for step in range(self.environment.max_steps):

                action, log_prob = self.agent.policy.act(state)

                # next_state, reward, done, success = self.environment.step(
                #     action.detach().cpu().numpy(),
                # )

                next_state, reward, done, success = self.environment.small_step(
                    action.detach().cpu().numpy(),
                    max_distance
                )
                
                # 训练用
                self.agent.memory.states.append(state)
                self.agent.memory.actions.append(action)
                self.agent.memory.logprobs.append(log_prob)
                self.agent.memory.rewards.append(reward)
                self.agent.memory.dones.append(done)

                # 自用
                self.final_reward_memory.append(reward)

                state = torch.FloatTensor(next_state).to(device)
                episode_reward += reward
                episode_steps += 1

                if done:
                    break

            # 这里应该是每个episode最多训练一次
            if len(self.agent.memory.states) >= self.batch_size: # 这个值是不是应该调大一点，不让其每次都训练
                self.agent.update()
                update_count += 1
                now_loss = self.agent.loss
                self.agent.memory.clear()
                self.loss_history.append(self.agent.loss)

            # 更新lr
            if abs(now_loss) > abs(biggest_loss):
                self.agent.lr = self.agent.origin_lr # 如果loss为史上最大，重置lr
                wether_update_lr_count = 0
            else:
                wether_update_lr_count += 1
                if wether_update_lr_count >= 100 and self.agent.lr > self.the_smallest_lr_threshold:
                    self.agent.lr *= 0.8
                    logger.info(f"lr减小，当前lr: {self.agent.lr:.6f}")
                    wether_update_lr_count = 0

            # 记录loss变化（仅用于监控训练过程，不再用于保存最优模型）
            if abs(now_loss) < abs(last_best_loss):
                logger.info(f"Loss改善: {last_best_loss:.3f} → {now_loss:.3f}")
                last_best_loss = now_loss
            if abs(now_loss) > abs(biggest_loss):
                biggest_loss = now_loss

            # 保存最近一次训练参数
            fileio.write_training_parameters_file(self.agent,last_training_parameters_path)
            
            # 每test_frequency个episode进行一次测试评估
            if (episode + 1) % self.test_frequency == 0 and self.wether_test:
                logger.info(f"{'='*80}")
                logger.info(f"开始测试集评估 (Episode {episode + 1})")
                logger.info(f"{'='*80}")
                
                # 评估当前模型
                current_test_error = self.evaluate_on_test_set()
                
                logger.info(f"平均测试误差: {current_test_error:.3f}°")
                logger.info(f"历史最优误差: {self.best_test_error:.3f}°")
                
                # 如果误差减小，保存最优模型
                if current_test_error < self.best_test_error:
                    self.best_test_error = current_test_error
                    fileio.write_training_parameters_file(self.agent, best_training_parameters_path)
                    logger.info(f"{GREEN}✓ 测试误差减小，保存最优模型！{RESET}")
                else:
                    logger.info(f"{RED}✗ 测试误差未改善，不更新最优模型{RESET}")
                
                logger.info(f"{'='*80}\n")
            
            angles_error = self.environment.target - self.environment.theta
            angles_error_l2 = np.linalg.norm(angles_error)
            initial_error = self.environment.target - initial_theta

            # 构造每个关节的误差字符串，方便判断
            error_parts = []
            for i in range(len(angles_error)):
                arrow = "↑" if abs(angles_error[i]) > abs(initial_error[i]) else "↓"
                error_parts.append(f"{abs(angles_error[i]):.3f}{arrow}")
            error_str = f"[{', '.join(error_parts)}]"
            # 整体L2箭头：更靠近目标为↑，远离为↓
            global_arrow = "↑" if (np.linalg.norm(initial_error) < angles_error_l2) else "↓"
            error_str_L2 = f"from {np.linalg.norm(initial_error):.3f} to {angles_error_l2:.3f} {global_arrow}"
            

            rewards_history.append(episode_reward)
            steps_history.append(episode_steps)
            # 如果成功的话，在success_history中记录下episode
            if success:
                success_history.append(episode)

            # 输出奖励数据
            # 实际想记录的是每个episode的最终奖励
            if (episode + 1) % 10 == 0:
                fileio.write_reward_file(reward_container = rewards_history, write_reward_file_path = write_reward_file_path)

            # 打印输出------------------------------------------------------------------------------------------
            logger.info(f"last_best_loss: {last_best_loss:.3f}   now_loss: {now_loss:.3f}   final_reward: {episode_reward:.3f}   lr: {self.agent.lr:.6f}   step: {episode_steps}")
            logger.info(f"initial_angles: {np.round(initial_theta, 3)}")
            logger.info(f"final_angles:   {np.round(self.environment.theta, 3)}")
            logger.info(f"target_angles:  {np.round(self.environment.target, 3)}")
            logger.info(f"angles_error:   {error_str},   {error_str_L2}")
            
            if success:
                logger.info(f"{GREEN}成功{RESET}")
            else:
                logger.info(f"{RED}失败{RESET}")

            logger.info(f"-----------------------------------------------------------------------------------------")
            logger.info(f" ")

    
