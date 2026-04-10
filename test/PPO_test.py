#引入模块----------------------------------
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
#------------------------------------------
#自定义模块--------------------------------

#------------------------------------------

# 主体-------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Actor 网络：输出动作的高斯分布的均值 (Mean)
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Tanh 将输出限制在 [-1, 1] 之间，后续可以根据机械臂实际角度范围进行缩放
        )
        
        # 定义动作的标准差 (Log Std)，设为可训练参数或固定值
        # 这里为了简便，使用可训练的独立参数
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic 网络：评估当前状态的价值 (Value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor_mean(state)
        action_var = self.actor_log_std.exp().expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        # 创建多维高斯分布
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # 采样得到动作
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor_mean(state)
        action_var = self.actor_log_std.exp().expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPO_Agent:
    def __init__(self, state_dim=12, action_dim=6, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=40, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': [self.policy.actor_log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        # 旧策略，用于计算 PPO 的 ratio
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, prev_angles, target_pos, obs_pos):
        """
        组合输入信息并选择动作
        """
        # 拼接状态向量 (假设输入都是一维 numpy array 或 list)
        state_array = np.concatenate([prev_angles, target_pos, obs_pos])
        
        with torch.no_grad():
            state = torch.FloatTensor(state_array).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        
        # 将数据存入 Buffer，用于后续更新
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        # 返回的是归一化的动作 [-1, 1]，需要在环境中将其映射到机械臂的实际角度范围
        return action.detach().cpu().numpy().flatten()

    def update(self):
        """
        使用收集到的数据更新网络参数
        """
        # 计算 Returns (Monte Carlo 估计)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # 归一化 rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 转换 Buffer 中的数据为 Tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # 计算 Advantages
        advantages = rewards.detach() - old_state_values.detach()

        # 优化 PPO 策略 K 个 epoch
        for _ in range(self.K_epochs):
            # 评估旧的动作和状态
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # 计算 Ratio: pi_theta / pi_theta__old
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算 Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # PPO 损失函数 = Actor Loss + Critic Loss - Entropy Bonus
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # 梯度下降
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # 更新 policy_old
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 清空 Buffer 准备下一轮收集
        self.buffer.clear()
