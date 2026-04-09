#引入模块----------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
#------------------------------------------

#注意事项--------------------------------
# 1. SAC使用最大熵框架，自动平衡探索与利用
# 2. 使用双Q网络减少过估计
# 3. 使用自动温度调节
#------------------------------------------

# device将由外部传入
device = None


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储一个transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批数据"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class SACActor(nn.Module):
    """SAC的策略网络（Actor）- 5层深度网络（与PPO一致）"""
    def __init__(self, state_dim, action_dim, action_bound=5.0):
        super().__init__()
        self.action_bound = action_bound
        
        # 5层深度网络，与PPO保持一致
        self.network = nn.Sequential(
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
        )
        
        # 输出均值和对数标准差
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        
    def forward(self, state):
        x = self.network(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # 提高log_std下限以增加探索（-2而非-20）
        log_std = torch.clamp(log_std, -2, 2)  # 限制标准差范围，增强探索
        
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        """采样动作，使用重参数化技巧"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = torch.tanh(mean) * self.action_bound
            return action, None
        
        # 重参数化技巧
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 可导的采样
        action = torch.tanh(x_t) * self.action_bound
        
        # 计算log_prob，考虑tanh变换
        log_prob = normal.log_prob(x_t)
        # 修正tanh变换的log_prob
        log_prob -= torch.log(self.action_bound * (1 - torch.tanh(x_t)**2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class SACCritic(nn.Module):
    """SAC的Q网络（Critic）- 5层深度网络（与PPO一致）"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # 5层深度网络，与PPO保持一致
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class SACAgent:
    def __init__(self, state_dim, action_dim, device=None):
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # 更新全局变量
        import agent.SAC_agent as sac_module
        sac_module.device = self.device
        
        self.action_bound = 5.0
        
        # 初始化网络
        self.actor = SACActor(state_dim, action_dim, self.action_bound).to(self.device)
        self.critic1 = SACCritic(state_dim, action_dim).to(self.device)
        self.critic2 = SACCritic(state_dim, action_dim).to(self.device)
        
        # 目标网络
        self.target_critic1 = SACCritic(state_dim, action_dim).to(self.device)
        self.target_critic2 = SACCritic(state_dim, action_dim).to(self.device)
        
        # 复制参数到目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.lr = 3e-4
        self.origin_lr = self.lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.lr)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(capacity=1000000)
        
        # SAC超参数
        self.gamma = 0.99
        self.tau = 0.005  # 软更新系数
        self.batch_size = 256
        
        # 自动温度调节
        self.target_entropy = -action_dim  # 目标熵
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        
        # 记录loss
        self.loss = 0
        self.loss_history = []
        
        # 奖励归一化参数（使用running mean/std）
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        self.reward_clip = 10.0  # 奖励裁剪范围
        
        # 用于兼容PPO接口的临时存储
        self.temp_states = []
        self.temp_actions = []
        self.temp_logprobs = []
        self.temp_rewards = []
        self.temp_dones = []
        self.temp_next_states = []
    
    def act(self, state):
        """与PPO接口兼容的act方法"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.sample(state, deterministic=False)
        
        # 兼容PPO的返回格式
        return action, log_prob.squeeze() if log_prob is not None else torch.tensor(0.0)
    
    def normalize_reward(self, reward):
        """归一化奖励（使用running statistics）"""
        # 更新统计信息
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std = np.sqrt(self.reward_std**2 + delta * delta2)
        self.reward_std = max(self.reward_std, 1e-8)  # 避免除零
        
        # 归一化并裁剪
        normalized_reward = (reward - self.reward_mean) / self.reward_std
        normalized_reward = np.clip(normalized_reward, -self.reward_clip, self.reward_clip)
        
        return normalized_reward
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储transition到经验回放缓冲区"""
        # 归一化奖励
        normalized_reward = self.normalize_reward(reward)
        
        self.memory.push(
            state.cpu().numpy() if isinstance(state, torch.Tensor) else state,
            action.cpu().numpy() if isinstance(action, torch.Tensor) else action,
            normalized_reward,  # 使用归一化后的奖励
            next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state,
            done
        )
    
    def update(self):
        """更新网络参数"""
        if len(self.memory) < self.batch_size:
            return
        
        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # 当前alpha值
        alpha = self.log_alpha.exp()
        
        # ---------------------------
        # 更新Critic
        # ---------------------------
        with torch.no_grad():
            # 从当前策略采样下一个动作
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_log_probs = next_log_probs
            
            # 计算目标Q值
            q1_target = self.target_critic1(next_states, next_actions)
            q2_target = self.target_critic2(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_target
        
        # 当前Q值
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        # Critic loss
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        
        # 更新Critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # ---------------------------
        # 更新Actor
        # ---------------------------
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ---------------------------
        # 更新温度参数
        # ---------------------------
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ---------------------------
        # 软更新目标网络
        # ---------------------------
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # 记录loss（使用actor loss）
        total_loss = actor_loss.item() + critic1_loss.item() + critic2_loss.item()
        self.loss_history.append(total_loss)
        self.loss = np.mean(self.loss_history[-100:])  # 最近100次的平均
