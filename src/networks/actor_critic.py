"""
Actor-Critic网络实现

- ActorNetwork: obs -> hidden -> action_dist (支持离散/连续)
- CriticNetwork: state -> hidden -> value (全局状态输入)
- 支持RNN（GRU隐藏状态）
- 支持通信接口（comm_input维度，用于CommNet/TarMAC）
- 正交初始化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """
    正交初始化
    
    对线性层和GRU层进行正交初始化，偏置初始化为0
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=gain)
            elif "bias" in name:
                nn.init.zeros_(param)


class ActorNetwork(nn.Module):
    """
    Actor网络：obs -> hidden -> action_dist
    
    支持离散动作（Categorical分布）和连续动作（Gaussian分布）
    支持RNN（GRU隐藏状态）
    支持通信接口（comm_input维度）
    
    Args:
        obs_dim: 观测维度
        action_dim: 动作维度
        action_type: 动作类型 "discrete" 或 "continuous"
        hidden_dim: 隐藏层维度
        use_rnn: 是否使用RNN
        comm_dim: 通信输入维度（0表示无通信）
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str = "discrete",
        hidden_dim: int = 64,
        use_rnn: bool = True,
        comm_dim: int = 0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.hidden_dim = hidden_dim
        self.use_rnn = use_rnn
        self.comm_dim = comm_dim

        # 输入维度：观测 + 通信输入
        input_dim = obs_dim + comm_dim

        # 特征提取层
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # RNN层
        if self.use_rnn:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 动作输出层
        if action_type == "discrete":
            self.action_out = nn.Linear(hidden_dim, action_dim)
        else:
            # 连续动作：输出均值和对数标准差
            self.mean_out = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        # 正交初始化
        self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))
        # 输出层使用较小的gain
        if action_type == "discrete":
            orthogonal_init(self.action_out, gain=0.01)
        else:
            orthogonal_init(self.mean_out, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        comm_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测, shape=(batch, obs_dim) 或 (batch, seq, obs_dim)
            hidden_state: GRU隐藏状态, shape=(1, batch, hidden_dim)
            comm_input: 通信输入, shape=(batch, comm_dim)
            
        Returns:
            action_dist: 动作分布
            hidden_state: 更新后的隐藏状态
        """
        # 拼接通信输入
        if comm_input is not None:
            x = torch.cat([obs, comm_input], dim=-1)
        else:
            x = obs

        # 特征提取
        x = F.relu(self.fc1(x))

        # RNN处理
        if self.use_rnn:
            # 确保输入是3D: (batch, seq, hidden_dim)
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
                if hidden_state is None:
                    hidden_state = torch.zeros(
                        1, x.size(0), self.hidden_dim, device=x.device
                    )
                x, hidden_state = self.gru(x, hidden_state)
                x = x.squeeze(1)  # (batch, hidden_dim)
            else:
                # (batch, seq, hidden_dim)
                if hidden_state is None:
                    hidden_state = torch.zeros(
                        1, x.size(0), self.hidden_dim, device=x.device
                    )
                x, hidden_state = self.gru(x, hidden_state)

        # 构建动作分布
        if self.action_type == "discrete":
            logits = self.action_out(x)
            action_dist = torch.distributions.Categorical(logits=logits)
        else:
            mean = self.mean_out(x)
            std = torch.exp(torch.clamp(self.log_std, -5, 2))
            action_dist = torch.distributions.Normal(mean, std)

        return action_dist, hidden_state

    def get_action(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        comm_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作
        
        Args:
            obs: 观测
            hidden_state: 隐藏状态
            deterministic: 是否确定性选择
            comm_input: 通信输入
            
        Returns:
            action: 选择的动作
            action_log_prob: 动作对数概率
            hidden_state: 更新后的隐藏状态
        """
        action_dist, hidden_state = self.forward(obs, hidden_state, comm_input)

        if deterministic:
            if self.action_type == "discrete":
                action = action_dist.probs.argmax(dim=-1)
            else:
                action = action_dist.mean
        else:
            action = action_dist.sample()

        action_log_prob = action_dist.log_prob(action)

        # 连续动作需要对每个维度求和
        if self.action_type == "continuous" and action_log_prob.dim() > 1:
            action_log_prob = action_log_prob.sum(dim=-1)

        return action, action_log_prob, hidden_state

    def evaluate_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        comm_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估已选动作的对数概率和熵
        
        Args:
            obs: 观测
            action: 已选动作
            hidden_state: 隐藏状态
            comm_input: 通信输入
            
        Returns:
            action_log_prob: 动作对数概率
            entropy: 策略熵
            hidden_state: 更新后的隐藏状态
        """
        action_dist, hidden_state = self.forward(obs, hidden_state, comm_input)

        action_log_prob = action_dist.log_prob(action)
        if self.action_type == "continuous" and action_log_prob.dim() > 1:
            action_log_prob = action_log_prob.sum(dim=-1)

        entropy = action_dist.entropy()
        if self.action_type == "continuous" and entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)

        return action_log_prob, entropy, hidden_state


class CriticNetwork(nn.Module):
    """
    Critic网络：state -> hidden -> value (全局状态输入)
    
    使用CTDE范式，Critic接收全局状态。
    
    Args:
        state_dim: 全局状态维度
        hidden_dim: 隐藏层维度
        use_rnn: 是否使用RNN
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        use_rnn: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_rnn = use_rnn

        # 特征提取层
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # RNN层
        if self.use_rnn:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 价值输出层
        self.value_out = nn.Linear(hidden_dim, 1)

        # 正交初始化
        self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))
        orthogonal_init(self.value_out, gain=1.0)

    def forward(
        self,
        state: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 全局状态, shape=(batch, state_dim) 或 (batch, seq, state_dim)
            hidden_state: GRU隐藏状态, shape=(1, batch, hidden_dim)
            
        Returns:
            value: 价值估计, shape=(batch, 1)
            hidden_state: 更新后的隐藏状态
        """
        x = F.relu(self.fc1(state))

        if self.use_rnn:
            if x.dim() == 2:
                x = x.unsqueeze(1)
                if hidden_state is None:
                    hidden_state = torch.zeros(
                        1, x.size(0), self.hidden_dim, device=x.device
                    )
                x, hidden_state = self.gru(x, hidden_state)
                x = x.squeeze(1)
            else:
                if hidden_state is None:
                    hidden_state = torch.zeros(
                        1, x.size(0), self.hidden_dim, device=x.device
                    )
                x, hidden_state = self.gru(x, hidden_state)

        value = self.value_out(x)
        return value, hidden_state
