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
    def __init__(self, state_dim, action_dim, action_bound=5.0):
        super().__init__()
        self.action_bound = float(action_bound)

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
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

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
    def __init__(self, state_dim, action_dim, action_bound=5.0):
        self.policy = ActorCritic(state_dim, action_dim, action_bound=action_bound).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        # 最简单版本学习率衰减，有待优化
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,   # 每100次episode降低一次学习率
            gamma=0.8
        )

        self.memory = Memory()

        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.loss = 0
        self.loss_history = []

    def update(self):
        states = torch.stack(self.memory.states).to(device)
        actions = torch.stack(self.memory.actions).to(device)
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
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

        for _ in range(self.K_epochs):
            logprobs, state_values, entropy = self.policy.evaluate(states, actions)

            ratios = torch.exp(logprobs - old_logprobs)
            advantages = returns - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

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
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.scheduler.step()

            self.loss_history.append(loss.item())

        self.loss = np.mean(self.loss_history)

        


    
