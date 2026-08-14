"""
CommNet: 隐式通信多Agent强化学习

在Actor网络中增加通信步骤：
h_i^{k+1} = f(h_i^k, 1/(N-1) * sum_{j!=i} h_j^k)

多步通信（comm_steps=1~3）
On-policy训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any, List

from src.algorithms.base import BaseAlgorithm
from src.networks.actor_critic import CriticNetwork


class CommNetActor(nn.Module):
    """
    CommNet的Actor网络（带隐式通信）
    
    多步通信：每步所有Agent的隐藏状态通过平均池化进行通信
    
    Args:
        obs_dim: 观测维度
        action_dim: 动作维度
        action_type: 动作类型
        hidden_dim: 隐藏层维度
        comm_steps: 通信步数
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str = "discrete",
        hidden_dim: int = 64,
        comm_steps: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.hidden_dim = hidden_dim
        self.comm_steps = comm_steps

        # 编码层
        self.encoder = nn.Linear(obs_dim, hidden_dim)

        # 通信步：每步有独立的通信层
        self.comm_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(comm_steps)
        ])

        # 动作输出层
        if action_type == "discrete":
            self.action_out = nn.Linear(hidden_dim, action_dim)
        else:
            self.mean_out = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        obs: torch.Tensor,
        all_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测, shape=(batch, obs_dim)
            all_hidden: 所有Agent的隐藏状态, shape=(batch, n_agents, hidden_dim)
            
        Returns:
            action_dist: 动作分布
            hidden_state: 通信后的隐藏状态
        """
        # 编码观测
        h = F.relu(self.encoder(obs))  # (batch, hidden_dim)

        # ---- 归一化 batch 维 (兼容调用方传入 2D/3D/4D 的 all_hidden) ----
        # act/get_action 对 hidden_state 做了 unsqueeze(0)，而 trainer 传入时若已带
        # batch 维会产生 4D；单步推理时可能为 2D。统一收敛到 3D (B, n_agents, hidden_dim)
        # 以避免 all_hidden.mean(dim=1) 后维度错位导致 cat 时报 "2 and 3"。
        if all_hidden is not None:
            if all_hidden.dim() == 2:
                all_hidden = all_hidden.unsqueeze(0)   # (n_agents, hidden) -> (1, n_agents, hidden)
            elif all_hidden.dim() == 4:
                all_hidden = all_hidden.squeeze(0)      # 去掉多余 batch 维 -> (B, n_agents, hidden)
        if h.dim() == 1:
            h = h.unsqueeze(0)                          # (hidden) -> (1, hidden)

        # 多步通信
        for step in range(self.comm_steps):
            if all_hidden is not None:
                # 计算通信消息：其他Agent隐藏状态的平均
                # all_hidden: (batch, n_agents, hidden_dim)
                # 需要知道当前Agent的ID，这里简化为使用所有Agent的平均
                comm_message = all_hidden.mean(dim=1)  # (batch, hidden_dim)
            else:
                # 第一步没有通信消息
                comm_message = torch.zeros_like(h)

            # 拼接自身隐藏状态和通信消息
            h_input = torch.cat([h, comm_message], dim=-1)  # (batch, hidden_dim * 2)
            h = F.relu(self.comm_layers[step](h_input))  # (batch, hidden_dim)

        # 构建动作分布
        if self.action_type == "discrete":
            logits = self.action_out(h)
            action_dist = torch.distributions.Categorical(logits=logits)
        else:
            mean = self.mean_out(h)
            std = torch.exp(torch.clamp(self.log_std, -5, 2))
            action_dist = torch.distributions.Normal(mean, std)

        return action_dist, h

    def get_action(
        self,
        obs: torch.Tensor,
        all_hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作
        """
        action_dist, hidden_state = self.forward(obs, all_hidden)

        if deterministic:
            if self.action_type == "discrete":
                action = action_dist.probs.argmax(dim=-1)
            else:
                action = action_dist.mean
        else:
            action = action_dist.sample()

        action_log_prob = action_dist.log_prob(action)
        if self.action_type == "continuous" and action_log_prob.dim() > 1:
            action_log_prob = action_log_prob.sum(dim=-1)

        return action, action_log_prob, hidden_state

    def evaluate_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        all_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估已选动作的对数概率和熵
        """
        action_dist, hidden_state = self.forward(obs, all_hidden)

        action_log_prob = action_dist.log_prob(action)
        if self.action_type == "continuous" and action_log_prob.dim() > 1:
            action_log_prob = action_log_prob.sum(dim=-1)

        entropy = action_dist.entropy()
        if self.action_type == "continuous" and entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)

        return action_log_prob, entropy, hidden_state


class CommNet(BaseAlgorithm):
    """
    CommNet算法：隐式通信多Agent强化学习
    
    在Actor网络中增加通信步骤，实现Agent间的隐式通信。
    On-policy训练。
    
    Args:
        config: 算法超参数字典
        env_info: 环境信息字典
    """

    def __init__(self, config: dict, env_info: dict):
        super().__init__(config, env_info)

        # 超参数
        self.lr = config.get("lr", 0.0005)
        self.clip_ratio = config.get("clip_ratio", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.ppo_epoch = config.get("ppo_epoch", 5)
        self.num_mini_batch = config.get("num_mini_batch", 1)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.comm_steps = config.get("comm_steps", 1)
        self.share_param = config.get("share_param", True)

        # 构建网络
        self._build_networks()

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, eps=1e-5
        )

        # 训练步数
        self._training_step = 0

    def _build_networks(self) -> None:
        """构建CommNet Actor和Critic网络"""
        if self.share_param:
            self.actor = CommNetActor(
                obs_dim=self.obs_shape,
                action_dim=self.action_shape,
                action_type=self.action_type,
                hidden_dim=self.hidden_dim,
                comm_steps=self.comm_steps,
            )
        else:
            self.actors = nn.ModuleList([
                CommNetActor(
                    obs_dim=self.obs_shape,
                    action_dim=self.action_shape,
                    action_type=self.action_type,
                    hidden_dim=self.hidden_dim,
                    comm_steps=self.comm_steps,
                )
                for _ in range(self.n_agents)
            ])

        self.critic = CriticNetwork(
            state_dim=self.state_shape,
            hidden_dim=self.hidden_dim,
            use_rnn=False,  # CommNet的Critic不使用RNN
        )

    def act(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量动作选择（带通信）
        
        Args:
            obs: 观测, shape=(n_agents, obs_dim)
            hidden_state: 隐藏状态, shape=(n_agents, hidden_dim)
            deterministic: 是否确定性选择
            
        Returns:
            actions: 动作
            hidden_state: 更新后的隐藏状态
            action_log_probs: 动作对数概率
        """
        # 使用隐藏状态作为通信的初始状态
        all_hidden = hidden_state.unsqueeze(0)  # (1, n_agents, hidden_dim)

        actions_list = []
        log_probs_list = []
        new_hidden_states = []

        for agent_id in range(self.n_agents):
            agent_obs = obs[agent_id].unsqueeze(0)  # (1, obs_dim)

            if self.share_param:
                action, log_prob, new_hidden = self.actor.get_action(
                    agent_obs, all_hidden, deterministic=deterministic
                )
            else:
                action, log_prob, new_hidden = self.actors[agent_id].get_action(
                    agent_obs, all_hidden, deterministic=deterministic
                )

            actions_list.append(action.squeeze(0))
            log_probs_list.append(log_prob.squeeze(0))
            new_hidden_states.append(new_hidden.squeeze(0))

        hidden_state = torch.stack(new_hidden_states, dim=0)
        actions = torch.stack(actions_list, dim=0)
        action_log_probs = torch.stack(log_probs_list, dim=0)

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
        obs = obs.unsqueeze(0)
        hidden_state = hidden_state.unsqueeze(0)

        if self.share_param:
            action, log_prob, new_hidden = self.actor.get_action(
                obs, hidden_state, deterministic=deterministic
            )
        else:
            action, log_prob, new_hidden = self.actors[0].get_action(
                obs, hidden_state, deterministic=deterministic
            )

        return action.squeeze(0), log_prob.squeeze(0), new_hidden.squeeze(0)

    def get_value(
        self,
        state: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        集中式Critic价值估计
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        value, _ = self.critic(state)
        return value.squeeze(-1)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        PPO clip更新
        """
        obs = batch["obs"].to(self.device)
        state = batch["state"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["action_log_probs"].to(self.device)
        returns = batch["returns"].to(self.device)
        advantages = batch["advantages"].to(self.device)

        # 优势归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epoch):
            new_log_probs = []
            entropies = []

            for agent_id in range(self.n_agents):
                agent_obs = obs[:, agent_id, :]
                if self.share_param:
                    log_prob, entropy, _ = self.actor.evaluate_action(
                        agent_obs, actions[:, agent_id]
                    )
                else:
                    log_prob, entropy, _ = self.actors[agent_id].evaluate_action(
                        agent_obs, actions[:, agent_id]
                    )
                new_log_probs.append(log_prob)
                entropies.append(entropy)

            new_log_probs = torch.stack(new_log_probs, dim=-1)
            mean_entropy = torch.stack(entropies, dim=-1).mean()

            if old_log_probs.dim() == 2:
                old_log_probs_sum = old_log_probs.sum(dim=-1, keepdim=True)
            else:
                old_log_probs_sum = old_log_probs
            new_log_probs_sum = new_log_probs.sum(dim=-1, keepdim=True)

            ratio = torch.exp(new_log_probs_sum - old_log_probs_sum)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value, _ = self.critic(state)
            value = value.squeeze(-1)
            value_loss = F.mse_loss(returns.squeeze(-1), value)

            loss = (
                policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * mean_entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += mean_entropy.item()

        self._training_step += 1

        return {
            "policy_loss": total_policy_loss / self.ppo_epoch,
            "value_loss": total_value_loss / self.ppo_epoch,
            "entropy": total_entropy / self.ppo_epoch,
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
