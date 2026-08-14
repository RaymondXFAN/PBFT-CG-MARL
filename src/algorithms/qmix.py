"""
QMIX: Value Function Factorisation for Deep Multi-Agent Reinforcement Learning

QMIX值分解算法：
- 每个Agent有独立Q网络
- Mixing Network用超网络保证单调性
- EPS-greedy探索
- Off-policy训练，使用Replay Buffer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import copy

from src.algorithms.base import BaseAlgorithm
from src.networks.mixing_net import MixingNetwork


class AgentQNetwork(nn.Module):
    """
    单Agent的Q网络
    
    Args:
        obs_dim: 观测维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            obs: 观测, shape=(batch, obs_dim)
            
        Returns:
            q_values: Q值, shape=(batch, action_dim)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class QMIX(BaseAlgorithm):
    """
    QMIX值分解算法
    
    通过Mixing Network保证混合后的Q_tot对每个Agent的Q_i是单调的。
    
    Args:
        config: 算法超参数字典
        env_info: 环境信息字典
    """

    def __init__(self, config: dict, env_info: dict):
        super().__init__(config, env_info)

        # QMIX只支持离散动作
        assert self.action_type == "discrete", "QMIX只支持离散动作空间"

        # 超参数
        self.lr = config.get("lr", 0.0005)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_finish = config.get("epsilon_finish", 0.05)
        self.epsilon_anneal_time = config.get("epsilon_anneal_time", 50000)
        self.batch_size = config.get("batch_size", 32)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.mixing_embed_dim = config.get("mixing_embed_dim", 32)
        self.hyper_hidden_dim = config.get("hyper_hidden_dim", 64)
        self.max_grad_norm = config.get("max_grad_norm", 10.0)
        self.target_update_interval = config.get("target_update_interval", 200)
        self.tau = config.get("tau", 0.005)  # 软更新系数

        # 构建网络
        self._build_networks()

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # 训练步数
        self._training_step = 0

    def _build_networks(self) -> None:
        """构建Q网络和Mixing Network"""
        # 各Agent的Q网络
        self.agent_q_networks = nn.ModuleList([
            AgentQNetwork(self.obs_shape, self.action_shape, self.hidden_dim)
            for _ in range(self.n_agents)
        ])

        # Mixing Network
        self.mixing_network = MixingNetwork(
            n_agents=self.n_agents,
            state_dim=self.state_shape,
            mixing_embed_dim=self.mixing_embed_dim,
            hyper_hidden_dim=self.hyper_hidden_dim,
        )

        # Target网络
        self.target_agent_q_networks = nn.ModuleList([
            AgentQNetwork(self.obs_shape, self.action_shape, self.hidden_dim)
            for _ in range(self.n_agents)
        ])
        self.target_mixing_network = MixingNetwork(
            n_agents=self.n_agents,
            state_dim=self.state_shape,
            mixing_embed_dim=self.mixing_embed_dim,
            hyper_hidden_dim=self.hyper_hidden_dim,
        )

        # 复制参数到target网络
        self._update_target_hard()

    def _update_target_hard(self) -> None:
        """硬更新target网络"""
        for agent_id in range(self.n_agents):
            self.target_agent_q_networks[agent_id].load_state_dict(
                self.agent_q_networks[agent_id].state_dict()
            )
        self.target_mixing_network.load_state_dict(
            self.mixing_network.state_dict()
        )

    def _update_target_soft(self) -> None:
        """软更新target网络"""
        for agent_id in range(self.n_agents):
            for param, target_param in zip(
                self.agent_q_networks[agent_id].parameters(),
                self.target_agent_q_networks[agent_id].parameters(),
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        for param, target_param in zip(
            self.mixing_network.parameters(),
            self.target_mixing_network.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _get_epsilon(self) -> float:
        """获取当前epsilon值（线性衰减）"""
        if self._training_step >= self.epsilon_anneal_time:
            return self.epsilon_finish
        else:
            return (
                self.epsilon_start
                - (self.epsilon_start - self.epsilon_finish)
                * self._training_step
                / self.epsilon_anneal_time
            )

    def act(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量动作选择（EPS-greedy）
        
        Args:
            obs: 观测, shape=(n_agents, obs_dim)
            hidden_state: 隐藏状态（QMIX不使用，保留接口兼容）
            deterministic: 是否确定性选择
            
        Returns:
            actions: 动作, shape=(n_agents,)
            hidden_state: 隐藏状态（未使用）
            action_log_probs: 动作对数概率（未使用）
        """
        actions = []
        epsilon = 0.0 if deterministic else self._get_epsilon()

        for agent_id in range(self.n_agents):
            agent_obs = obs[agent_id].unsqueeze(0)
            q_values = self.agent_q_networks[agent_id](agent_obs)

            if np.random.random() < epsilon:
                # 随机探索
                action = torch.randint(0, self.action_shape, (1,), device=self.device)
            else:
                # 贪心选择
                action = q_values.argmax(dim=-1)

            actions.append(action.squeeze(0))

        actions = torch.stack(actions, dim=0)
        # QMIX不使用log_prob，返回0
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
        q_values = self.agent_q_networks[0](obs)
        epsilon = 0.0 if deterministic else self._get_epsilon()

        if np.random.random() < epsilon:
            action = torch.randint(0, self.action_shape, (1,), device=self.device)
        else:
            action = q_values.argmax(dim=-1)

        return action.squeeze(0), torch.tensor(0.0, device=self.device), hidden_state

    def get_value(
        self,
        state: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取Q_tot值
        """
        # QMIX的get_value需要obs，这里简化处理
        return torch.tensor(0.0, device=self.device)

    def get_q_values(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取Q_tot值
        
        Args:
            obs: 各Agent的观测, shape=(batch, n_agents, obs_dim)
            state: 全局状态, shape=(batch, state_dim)
            
        Returns:
            q_tot: Q_tot, shape=(batch, 1)
            agent_qs: 各Agent的Q值, shape=(batch, n_agents)
        """
        batch_size = obs.size(0)
        agent_qs = []

        for agent_id in range(self.n_agents):
            agent_obs = obs[:, agent_id, :]  # (batch, obs_dim)
            q_values = self.agent_q_networks[agent_id](agent_obs)  # (batch, action_dim)
            agent_qs.append(q_values)

        # 选择当前动作的Q值
        agent_q_values = torch.stack(agent_qs, dim=1)  # (batch, n_agents, action_dim)

        return agent_q_values, state

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        QMIX更新
        
        使用TD-loss和Mixing Network更新参数。
        
        Args:
            batch: 数据batch字典，包含obs, state, actions, rewards, next_obs, next_state, dones
            
        Returns:
            loss_dict: 损失字典
        """
        obs = batch["obs"].to(self.device)  # (batch, n_agents, obs_dim)
        state = batch["state"].to(self.device)  # (batch, state_dim)
        actions = batch["actions"].to(self.device)  # (batch, n_agents)
        rewards = batch["rewards"].to(self.device)  # (batch, 1)
        next_obs = batch["next_obs"].to(self.device)  # (batch, n_agents, obs_dim)
        next_state = batch["next_state"].to(self.device)  # (batch, state_dim)
        dones = batch["dones"].to(self.device)  # (batch, 1)

        batch_size = obs.size(0)

        # 计算当前Q值
        current_agent_qs = []
        for agent_id in range(self.n_agents):
            agent_obs = obs[:, agent_id, :]
            q_values = self.agent_q_networks[agent_id](agent_obs)
            # 选择执行动作的Q值
            action = actions[:, agent_id].unsqueeze(-1)
            q_value = q_values.gather(1, action)  # (batch, 1)
            current_agent_qs.append(q_value)

        current_agent_qs = torch.cat(current_agent_qs, dim=-1)  # (batch, n_agents)

        # Mixing Network计算Q_tot
        q_tot = self.mixing_network(current_agent_qs, state)  # (batch, 1, 1)
        q_tot = q_tot.squeeze(-1)  # (batch, 1)

        # 计算目标Q值（不计算梯度）
        with torch.no_grad():
            next_agent_qs = []
            for agent_id in range(self.n_agents):
                next_agent_obs = next_obs[:, agent_id, :]
                q_values = self.target_agent_q_networks[agent_id](next_agent_obs)
                # 贪心选择
                q_value = q_values.max(dim=-1, keepdim=True)[0]  # (batch, 1)
                next_agent_qs.append(q_value)

            next_agent_qs = torch.cat(next_agent_qs, dim=-1)  # (batch, n_agents)
            target_q_tot = self.target_mixing_network(next_agent_qs, next_state)  # (batch, 1, 1)
            target_q_tot = target_q_tot.squeeze(-1)  # (batch, 1)

            # TD目标
            target_q = rewards + self.gamma * (1 - dones) * target_q_tot

        # TD-loss
        loss = F.mse_loss(q_tot, target_q.detach())

        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # 更新target网络
        self._training_step += 1
        if self._training_step % self.target_update_interval == 0:
            self._update_target_hard()

        return {
            "q_loss": loss.item(),
            "q_tot_mean": q_tot.mean().item(),
            "epsilon": self._get_epsilon(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self._training_step,
            "config": self.config,
            "env_info": self.env_info,
        }, path)

    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._training_step = checkpoint["training_step"]
