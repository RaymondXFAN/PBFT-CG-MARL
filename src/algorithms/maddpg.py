"""
MADDPG: Multi-Agent Deep Deterministic Policy Gradient

Centralized Critic + Decentralized Actor
- DDPG风格的确定性策略梯度
- 支持连续动作空间
- Off-policy训练
- Target network + soft update
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import copy

from src.algorithms.base import BaseAlgorithm


class MADDPGActor(nn.Module):
    """
    MADDPG的Actor网络（确定性策略）
    
    Args:
        obs_dim: 观测维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        action_limit: 动作上限（用于连续动作）
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        action_limit: float = 1.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.action_limit = action_limit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            obs: 观测, shape=(batch, obs_dim)
            
        Returns:
            action: 动作, shape=(batch, action_dim)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.action_limit
        return action


class MADDPGCritic(nn.Module):
    """
    MADDPG的Critic网络（集中式）
    
    输入：全局状态 + 所有Agent的动作
    输出：Q值
    
    Args:
        state_dim: 全局状态维度
        n_agents: Agent数量
        action_dim: 单Agent动作维度
        hidden_dim: 隐藏层维度
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        action_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        # 输入：全局状态 + 所有Agent的动作
        input_dim = state_dim + n_agents * action_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 全局状态, shape=(batch, state_dim)
            actions: 所有Agent的动作, shape=(batch, n_agents * action_dim)
            
        Returns:
            q_value: Q值, shape=(batch, 1)
        """
        x = torch.cat([state, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class MADDPG(BaseAlgorithm):
    """
    MADDPG算法
    
    Centralized Critic + Decentralized Actor
    DDPG风格的确定性策略梯度，支持连续动作空间。
    
    Args:
        config: 算法超参数字典
        env_info: 环境信息字典
    """

    def __init__(self, config: dict, env_info: dict):
        super().__init__(config, env_info)

        # 超参数
        self.lr_actor = config.get("lr_actor", 0.0001)
        self.lr_critic = config.get("lr_critic", 0.001)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.batch_size = config.get("batch_size", 256)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.action_limit = config.get("action_limit", 1.0)
        self.noise_std = config.get("noise_std", 0.1)  # 探索噪声

        # 构建网络
        self._build_networks()

        # 优化器
        self.actor_optimizers = [
            torch.optim.Adam(self.actors[i].parameters(), lr=self.lr_actor)
            for i in range(self.n_agents)
        ]
        self.critic_optimizers = [
            torch.optim.Adam(self.critics[i].parameters(), lr=self.lr_critic)
            for i in range(self.n_agents)
        ]

        # 训练步数
        self._training_step = 0

    def _build_networks(self) -> None:
        """构建Actor和Critic网络"""
        # Actor网络
        self.actors = nn.ModuleList([
            MADDPGActor(
                obs_dim=self.obs_shape,
                action_dim=self.action_shape,
                hidden_dim=self.hidden_dim,
                action_limit=self.action_limit,
            )
            for _ in range(self.n_agents)
        ])

        # Critic网络
        self.critics = nn.ModuleList([
            MADDPGCritic(
                state_dim=self.state_shape,
                n_agents=self.n_agents,
                action_dim=self.action_shape,
                hidden_dim=self.hidden_dim,
            )
            for _ in range(self.n_agents)
        ])

        # Target网络
        self.target_actors = nn.ModuleList([
            MADDPGActor(
                obs_dim=self.obs_shape,
                action_dim=self.action_shape,
                hidden_dim=self.hidden_dim,
                action_limit=self.action_limit,
            )
            for _ in range(self.n_agents)
        ])
        self.target_critics = nn.ModuleList([
            MADDPGCritic(
                state_dim=self.state_shape,
                n_agents=self.n_agents,
                action_dim=self.action_shape,
                hidden_dim=self.hidden_dim,
            )
            for _ in range(self.n_agents)
        ])

        # 复制参数到target网络
        for i in range(self.n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

    def _soft_update_target(self) -> None:
        """软更新target网络"""
        for i in range(self.n_agents):
            for param, target_param in zip(
                self.actors[i].parameters(), self.target_actors[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.critics[i].parameters(), self.target_critics[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def act(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量动作选择
        
        Args:
            obs: 观测, shape=(n_agents, obs_dim)
            hidden_state: 隐藏状态（MADDPG不使用）
            deterministic: 是否确定性选择
            
        Returns:
            actions: 动作, shape=(n_agents, action_dim)
            hidden_state: 隐藏状态（未使用）
            action_log_probs: 动作对数概率（未使用）
        """
        actions = []
        for agent_id in range(self.n_agents):
            agent_obs = obs[agent_id].unsqueeze(0)
            action = self.actors[agent_id](agent_obs)

            if not deterministic:
                # 添加探索噪声
                noise = torch.randn_like(action) * self.noise_std
                action = torch.clamp(action + noise, -self.action_limit, self.action_limit)

            actions.append(action.squeeze(0))

        actions = torch.stack(actions, dim=0)
        action_log_probs = torch.zeros(self.n_agents, device=self.device)

        return actions, hidden_state, action_log_probs

    def get_action(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单Agent动作选择
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        action = self.actors[0](obs)

        if not deterministic:
            noise = torch.randn_like(action) * self.noise_std
            action = torch.clamp(action + noise, -self.action_limit, self.action_limit)

        return action.squeeze(0), torch.tensor(0.0, device=self.device), hidden_state

    def get_value(
        self,
        state: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        集中式Critic价值估计
        """
        return torch.tensor(0.0, device=self.device)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        MADDPG更新
        
        使用确定性策略梯度更新Actor和Critic。
        
        Args:
            batch: 数据batch字典
            
        Returns:
            loss_dict: 损失字典
        """
        obs = batch["obs"].to(self.device)  # (batch, n_agents, obs_dim)
        state = batch["state"].to(self.device)  # (batch, state_dim)
        actions = batch["actions"].to(self.device)  # (batch, n_agents, action_dim)
        rewards = batch["rewards"].to(self.device)  # (batch, 1)
        next_obs = batch["next_obs"].to(self.device)  # (batch, n_agents, obs_dim)
        next_state = batch["next_state"].to(self.device)  # (batch, state_dim)
        dones = batch["dones"].to(self.device)  # (batch, 1)

        batch_size = obs.size(0)

        total_critic_loss = 0.0
        total_actor_loss = 0.0

        # 展平actions: (batch, n_agents * action_dim)
        if actions.dim() == 2 and actions.size(-1) == self.n_agents:
            # 离散动作空间，one-hot编码
            actions_flat = F.one_hot(actions.long(), self.action_shape).float()
            actions_flat = actions_flat.view(batch_size, -1)
        else:
            actions_flat = actions.view(batch_size, -1) if actions.dim() > 2 else actions

        # 计算target actions
        with torch.no_grad():
            target_actions = []
            for agent_id in range(self.n_agents):
                next_agent_obs = next_obs[:, agent_id, :]
                target_action = self.target_actors[agent_id](next_agent_obs)
                target_actions.append(target_action)
            target_actions = torch.cat(target_actions, dim=-1)  # (batch, n_agents * action_dim)

        # 更新每个Agent的Critic和Actor
        for agent_id in range(self.n_agents):
            # ===== Critic更新 =====
            # 当前Q值
            current_q = self.critics[agent_id](state, actions_flat)

            # Target Q值
            with torch.no_grad():
                target_q = self.target_critics[agent_id](next_state, target_actions)
                target_q_value = rewards + self.gamma * (1 - dones) * target_q

            critic_loss = F.mse_loss(current_q, target_q_value)

            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critics[agent_id].parameters(), self.max_grad_norm
            )
            self.critic_optimizers[agent_id].step()

            # ===== Actor更新 =====
            # 计算当前策略的动作
            current_actions = []
            for i in range(self.n_agents):
                if i == agent_id:
                    current_actions.append(self.actors[i](obs[:, i, :]))
                else:
                    with torch.no_grad():
                        current_actions.append(self.actors[i](obs[:, i, :]))
            current_actions = torch.cat(current_actions, dim=-1)

            # Actor loss: 最大化Q值
            actor_loss = -self.critics[agent_id](state, current_actions).mean()

            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actors[agent_id].parameters(), self.max_grad_norm
            )
            self.actor_optimizers[agent_id].step()

            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()

        # 软更新target网络
        self._soft_update_target()

        self._training_step += 1

        return {
            "critic_loss": total_critic_loss / self.n_agents,
            "actor_loss": total_actor_loss / self.n_agents,
            "lr_actor": self.actor_optimizers[0].param_groups[0]["lr"],
            "lr_critic": self.critic_optimizers[0].param_groups[0]["lr"],
        }

    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            "model_state_dict": self.state_dict(),
            "actor_optimizer_states": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizer_states": [opt.state_dict() for opt in self.critic_optimizers],
            "training_step": self._training_step,
            "config": self.config,
            "env_info": self.env_info,
        }, path)

    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        for i, opt_state in enumerate(checkpoint["actor_optimizer_states"]):
            self.actor_optimizers[i].load_state_dict(opt_state)
        for i, opt_state in enumerate(checkpoint["critic_optimizer_states"]):
            self.critic_optimizers[i].load_state_dict(opt_state)
        self._training_step = checkpoint["training_step"]
