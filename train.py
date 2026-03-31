#引入模块----------------------------------
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
#------------------------------------------

#自定义模块--------------------------------
from robot.robot import Robot
from visiualization import Visiualization
from environment import Environment
from agent.PPO_agent import PPOAgent
from agent.SAC_agent import SACAgent
#------------------------------------------

#注意事项--------------------------------
# 1.现在的输入：初始角度、目标角度、角度偏差
#------------------------------------------

# 主体-------------------------------------

GREEN = "\033[92m"
RESET = "\033[0m"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Train:
    
    def __init__(self, environment:Environment):
        self.agent = PPOAgent(state_dim = environment.state_dim, action_dim = environment.action_dim)
        # self.agent = SACAgent(state_dim = 12, action_dim = 6)
        self.robot = Robot()
        self.max_episodes = 1000
        self.environment = environment

        self.training_parameter_saving_path = None
        

    def train(self):

        update_count = 0
        reward_count = 0
        average_reward = 0

        for episode in range(self.max_episodes):
            state = torch.FloatTensor(self.environment.train_reset()).to(device)

            theta0 = self.environment.theta.copy()

            done_or_not = 0
            
            for step in range(self.environment.max_steps):

                action, log_prob = self.agent.policy.act(state)

                next_state, reward, done = self.environment.step(
                    action.detach().cpu().numpy(),
                )

                self.agent.memory.states.append(state)
                self.agent.memory.actions.append(action)
                self.agent.memory.logprobs.append(log_prob)
                self.agent.memory.rewards.append(reward)
                self.agent.memory.dones.append(done)

                state = torch.FloatTensor(next_state).to(device)

                if done:
                    done_or_not = done
                    break

            
            # 输出最终误差
            angles_error, posture_error = self.environment.error_calculate(angles = self.environment.theta, target_angles = self.environment.target)
            print(f"初始角度为：{np.round(theta0, 3)}")
            print(f"最终角度为：{np.round(self.environment.theta, 3)}")
            print(f"目标角度为：{np.round(self.environment.target, 3)}")
            print(f"角度误差为：{np.round(angles_error, 3)}")
            # print(f"位置误差为：{np.round(posture_error[:3], 3)}")
            print(f"当前奖励为：{np.round(self.agent.memory.rewards[-1], 3)}")
            average_reward += self.agent.memory.rewards[-1]
            reward_count += 1
            
            # 检测抵达
            if done_or_not:
                print(f"{GREEN}第{episode + 1}次成功抵达！{RESET}")
                done_or_not = 0

            if len(self.agent.memory.states) >= 1900: # 更新参数
                self.agent.update()
                self.agent.memory.clear()
                update_count += 1
                print(f"-------------------------------------------")
                print(f"第{update_count}次更新")
                print(f"当前损失为：{np.round(self.agent.loss, 3)}")
                print(f"平均奖励为：{np.round(average_reward / reward_count, 3)}")
                print(f"-------------------------------------------")
                reward_count = 0
                average_reward = 0

            print(f"Episode {episode + 1} finished")



    