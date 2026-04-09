#引入模块----------------------------------
from pickletools import optimize
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
#------------------------------------------
#自定义模块--------------------------------

#------------------------------------------
#注意事项--------------------------------
# 1.这个算法实际是PPO-clip，即PPO的一种改版
#------------------------------------------
# 主体-------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_bound = 0.1

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # log_std是可学习参数（关键）
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state):
        state = state.to(device)
        mean = torch.tanh(self.actor(state)) * self.action_bound
        log_std = torch.clamp(self.log_std, -2.0, 1.0)
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -self.action_bound, self.action_bound)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate(self, state, action):
        mean = torch.tanh(self.actor(state)) * self.action_bound
        log_std = torch.clamp(self.log_std, -2.0, 1.0)
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1) # 计算熵，将其作为一个奖励加入至Loss函数中，用于防止网络过早陷入局部最优
        value = self.critic(state)

        return log_prob, value.squeeze(), entropy

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        
    def clear(self):
        self.__init__()  

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.lr = 3e-4
        self.origin_lr = self.lr
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.memory = Memory()

        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.value_coef = 0.5
        self.entropy_coef = 0.02  # 增加熵系数，鼓励探索
        self.gae_lambda = 0.95  # 添加GAE参数
        self.max_grad_norm = 0.5
        self.loss = 0
        self.loss_history = []

    def update(self):
        states = torch.stack(self.memory.states).to(device)
        actions = torch.stack(self.memory.actions).to(device)
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        rewards = self.memory.rewards
        dones = self.memory.dones

        # 使用GAE计算优势函数和回报
        with torch.no_grad():
            values = self.policy.critic(states).squeeze()
            
        advantages = []
        returns = []
        gae = 0
        
        # 从后向前计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, entropy = self.policy.evaluate(states, actions)

            ratios = torch.exp(logprobs - old_logprobs)
            
            # 使用GAE优势函数
            advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratios * advantages_normalized
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_normalized

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(state_values, returns)
            entropy_loss = entropy.mean()
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy_loss
                )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            self.loss_history.append(loss.item())

        self.loss = np.mean(self.loss_history)

        


    
