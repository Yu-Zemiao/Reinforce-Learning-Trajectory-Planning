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
        self.batch_size = 512
        self.environment = environment

        self.training_parameter_saving_path = None
        self.model_snapshots = []
        self.final_reward_memory = []

        self.loss_history = []
        

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
            state = torch.FloatTensor(self.environment.train_reset()).to(device)

            initial_theta = self.environment.theta
            episode_reward = 0.0
            episode_steps = 0
            success = False

            update_count = 0

            
            for step in range(self.environment.max_steps):

                action, log_prob = self.agent.policy.act(state)

                next_state, reward, done, success = self.environment.step(
                    action.detach().cpu().numpy(),
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
                if wether_update_lr_count >= 100:
                    self.agent.lr *= 0.8
                    logger.info(f"lr减小，当前lr: {self.agent.lr:.6f}")
                    wether_update_lr_count = 0

            # 保存最优训练参数
            # loss越靠近0越好
            if abs(now_loss) < abs(last_best_loss):
                fileio.write_training_parameters_file(self.agent,best_training_parameters_path)
                logger.info("网络最优参数更新")
                last_best_loss = now_loss
            # 记录最大loss
            if abs(now_loss) > abs(biggest_loss):
                biggest_loss = now_loss

            # 保存最近一次训练参数
            fileio.write_training_parameters_file(self.agent,last_training_parameters_path)
            
            angles_error = self.environment.target - self.environment.theta
            angles_error_l2 = np.linalg.norm(angles_error)
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
            logger.info(f"last_best_loss: {last_best_loss:.3f}   now_loss: {now_loss:.3f}   final_reward: {episode_reward:.3f}   lr: {self.agent.lr:.6f}")
            logger.info(f"initial_angles: {np.round(initial_theta, 3)}")
            logger.info(f"final_angles:   {np.round(self.environment.theta, 3)}")
            logger.info(f"target_angles:  {np.round(self.environment.target, 3)}")
            logger.info(f"angles_error:   {np.round(angles_error, 3)}  angles_error_l2: {angles_error_l2:.3f}")
            
            if success:
                logger.info(f"{GREEN}成功{RESET}")
            else:
                logger.info(f"{RED}失败{RESET}")

            logger.info(f"-----------------------------------------------------------------------------------------")
            logger.info(f" ")

    
