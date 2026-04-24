#引入模块----------------------------------
from pickletools import optimize
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
#------------------------------------------
#自定义模块--------------------------------
from config import device
#------------------------------------------
#注意事项--------------------------------
# 1.这个算法实际是PPO-clip，即PPO的一种改版
#------------------------------------------
# 主体-------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_bound = 0.2

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
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
            nn.Linear(256, 1)
        )

    def act(self, state):
        state = state.to(device)

        # 不对 mean 做 tanh（关键修改）
        mean = self.actor(state)

        # 更稳定的 std 处理
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        # 使用 rsample（可重参数化，梯度更稳定）
        raw_action = dist.rsample()

        # tanh squash 到 [-1, 1]
        action = torch.tanh(raw_action)

        # 计算 log_prob（关键：用 raw_action + Jacobian 修正）
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        # 最后 scale 到环境范围
        action = action * self.action_bound

        return action.detach(), log_prob.detach()

    def evaluate(self, state, action):
        # actor 输出
        mean = self.actor(state)

        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        # ⚠️ 关键：把 action 还原回 raw_action
        # 因为 action 是经过 tanh + scale 的
        action_scaled = action / self.action_bound

        # 防止数值溢出
        action_scaled = torch.clamp(action_scaled, -0.999, 0.999)

        # 反 tanh（atanh）
        raw_action = 0.5 * torch.log((1 + action_scaled) / (1 - action_scaled))

        # 正确 log_prob（含修正项）
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action_scaled.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        # entropy（注意：这里仍是高斯 entropy，近似用即可）
        entropy = dist.entropy().sum(dim=-1)

        # critic
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
        self.lr = 1e-4
        self.origin_lr = self.lr
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.memory = Memory()

        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4 # 一批数据重复训练K次
        self.value_coef = 0.5
        self.entropy_coef = 0.01  # 增加熵系数，鼓励探索
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
        self.loss_history = []

