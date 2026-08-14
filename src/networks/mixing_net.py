"""
QMIX Mixing Network

- 输入：各Agent的Q值
- 输出：Q_tot
- 超网络生成权重，保证单调性（权重非负）
- 二级混合网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class HyperNetwork(nn.Module):
    """
    超网络：根据全局状态生成Mixing Network的权重
    
    Args:
        state_dim: 全局状态维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度（权重数量）
    """

    def __init__(self, state_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 全局状态, shape=(batch, state_dim)
            
        Returns:
            weights: 生成的权重, shape=(batch, output_dim)
        """
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x


class MixingNetwork(nn.Module):
    """
    QMIX的Mixing Network
    
    通过超网络生成权重，保证Q_tot对每个Agent的Q_i是单调的（权重非负）。
    使用二级混合网络结构。
    
    Args:
        n_agents: Agent数量
        state_dim: 全局状态维度
        mixing_embed_dim: 混合网络嵌入维度
        hyper_hidden_dim: 超网络隐藏维度
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hyper_hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_embed_dim = mixing_embed_dim

        # 第一级混合网络
        self.hyper_w1 = HyperNetwork(
            state_dim, hyper_hidden_dim, n_agents * mixing_embed_dim
        )
        self.hyper_b1 = HyperNetwork(state_dim, hyper_hidden_dim, mixing_embed_dim)

        # 第二级混合网络
        self.hyper_w2 = HyperNetwork(
            state_dim, hyper_hidden_dim, mixing_embed_dim
        )
        self.hyper_b2 = HyperNetwork(state_dim, hyper_hidden_dim, 1)

    def forward(
        self, agent_qs: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            agent_qs: 各Agent的Q值, shape=(batch, n_agents)
            state: 全局状态, shape=(batch, state_dim)
            
        Returns:
            q_tot: 混合后的Q_tot, shape=(batch, 1)
        """
        batch_size = agent_qs.size(0)

        # 第一级：生成权重和偏置
        w1 = torch.abs(self.hyper_w1(state))  # 绝对值保证非负
        b1 = self.hyper_b1(state)
        w1 = w1.view(batch_size, self.n_agents, self.mixing_embed_dim)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)

        # 第一级混合：Q_i * W1 + b1
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # 第二级：生成权重和偏置
        w2 = torch.abs(self.hyper_w2(state))  # 绝对值保证非负
        b2 = self.hyper_b2(state)
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        b2 = b2.view(batch_size, 1, 1)

        # 第二级混合：hidden * W2 + b2
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot
