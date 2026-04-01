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
        self.agent = PPOAgent(
            state_dim=environment.state_dim,
            action_dim=environment.action_dim,
            action_bound=environment.action_bound
        )
        # self.agent = SACAgent(state_dim = 12, action_dim = 6)
        self.robot = Robot()
        self.max_episodes = 1000
        self.batch_size = 512
        self.environment = environment

        self.training_parameter_saving_path = None
        

    def train(self):
        rewards_history = []
        steps_history = []
        success_history = []

        for episode in range(self.max_episodes):
            state = torch.FloatTensor(self.environment.train_reset()).to(device)

            episode_reward = 0.0
            episode_steps = 0
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
                episode_reward += reward
                episode_steps += 1

                if done:
                    done_or_not = done
                    break

                if len(self.agent.memory.states) >= self.batch_size:
                    self.agent.update()
                    self.agent.memory.clear()
            
            if len(self.agent.memory.states) > 0:
                self.agent.update()
                self.agent.memory.clear()
            
            angles_error = self.environment.target - self.environment.theta
            angles_error_l2 = np.linalg.norm(angles_error)
            success = bool(done_or_not and self.environment.arrive_detect(self.environment.theta, self.environment.target))
            rewards_history.append(episode_reward)
            steps_history.append(episode_steps)
            success_history.append(float(success))

            if (episode + 1) % 100 == 0:
                recent_success = np.mean(success_history[-100:]) * 100
                recent_reward = np.mean(rewards_history[-100:])
                recent_steps = np.mean(steps_history[-100:])
                print(f"[Ep {episode + 1}] SuccessRate(100ep): {recent_success:.1f}%  AvgReward: {recent_reward:.2f}  AvgSteps: {recent_steps:.1f}  angles_error: {np.round(angles_error, 3)}  angles_error_l2: {angles_error_l2:.3f}")
            else:
                print(f"Episode {episode + 1} angles_error: {np.round(angles_error, 3)}  angles_error_l2: {angles_error_l2:.3f}")

    
