#引入模块----------------------------------
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

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # log_std是可学习参数（关键）
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state):
        state = state.to(device)
        mean = self.actor(state)
        std = torch.exp(self.log_std)

        dist = Normal(mean, std) # 创建一个分布库，用于构建一个正态分布对象，均值和标准差由上一步计算出
        action = dist.sample()

        # action = 0.2 * torch.tanh(action)
        # action = torch.clamp(action, -0.02, 0.02)

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate(self, state, action):
        mean = self.actor(state)
        std = torch.exp(self.log_std)

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
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-5)
        self.memory = Memory()

        self.gamma = 0.99
        self.eps_clip = 0.2 # 截断参数
        self.K_epochs = 10
        self.loss = 0

    def update(self):
        states = torch.stack(self.memory.states).to(device)
        actions = torch.stack(self.memory.actions).to(device)
        old_logprobs = torch.stack(self.memory.logprobs).to(device)
        rewards = self.memory.rewards
        dones = self.memory.dones

        # 计算回报
        returns = []
        discounted = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted = 0
            discounted = r + self.gamma * discounted
            returns.insert(0, discounted) # 将每次新的值插到列表最前面，即倒序排序

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.K_epochs):
            logprobs, state_values, entropy = self.policy.evaluate(states, actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = returns - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * (returns - state_values).pow(2)
                # - 0.01 * entropy
            )
            
            # loss = -torch.min(surr1, surr2)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            self.loss = np.array(loss.mean().item())


    
