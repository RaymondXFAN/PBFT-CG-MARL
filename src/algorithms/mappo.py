"""
MAPPO: Multi-Agent Proximal Policy Optimization

标准MAPPO实现（无共识层），作为消融实验的基线。
与PBFT-CG-MAPPO相同的架构，但没有PBFT共识层。
直接使用Actor输出的动作。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any, List

from src.algorithms.base import BaseAlgorithm
from src.networks.actor_critic import ActorNetwork, CriticNetwork


class MAPPO(BaseAlgorithm):
    """
    标准MAPPO算法（无共识层）
    
    Centralized Training Decentralized Execution范式：
    - Actor: 各Agent独立观测，独立决策
    - Critic: 接收全局状态，估计价值
    
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
        self.use_rnn = config.get("use_rnn", True)
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
        """构建Actor和Critic网络"""
        if self.share_param:
            self.actor = ActorNetwork(
                obs_dim=self.obs_shape,
                action_dim=self.action_shape,
                action_type=self.action_type,
                hidden_dim=self.hidden_dim,
                use_rnn=self.use_rnn,
            )
            self.critic = CriticNetwork(
                state_dim=self.state_shape,
                hidden_dim=self.hidden_dim,
                use_rnn=self.use_rnn,
            )
        else:
            self.actors = nn.ModuleList([
                ActorNetwork(
                    obs_dim=self.obs_shape,
                    action_dim=self.action_shape,
                    action_type=self.action_type,
                    hidden_dim=self.hidden_dim,
                    use_rnn=self.use_rnn,
                )
                for _ in range(self.n_agents)
            ])
            self.critic = CriticNetwork(
                state_dim=self.state_shape,
                hidden_dim=self.hidden_dim,
                use_rnn=self.use_rnn,
            )

    def act(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量动作选择（无共识）
        
        Args:
            obs: 观测, shape=(n_agents, obs_dim)
            hidden_state: 隐藏状态, shape=(n_agents, hidden_dim)
            deterministic: 是否确定性选择
            
        Returns:
            actions: 动作
            hidden_state: 更新后的隐藏状态
            action_log_probs: 动作对数概率
        """
        actions_list = []
        log_probs_list = []
        new_hidden_states = []

        for agent_id in range(self.n_agents):
            agent_obs = obs[agent_id].unsqueeze(0)
            # hidden_state 形状: (num_layers=1, batch=n_agents, hidden_size)
            agent_hidden = hidden_state[:, agent_id, :].unsqueeze(0)  # (1, 1, hidden_dim)

            if self.share_param:
                action, log_prob, new_hidden = self.actor.get_action(
                    agent_obs, agent_hidden, deterministic=deterministic
                )
            else:
                action, log_prob, new_hidden = self.actors[agent_id].get_action(
                    agent_obs, agent_hidden, deterministic=deterministic
                )

            actions_list.append(action.squeeze(0))
            log_probs_list.append(log_prob.squeeze(0))
            new_hidden_states.append(new_hidden.squeeze(0).squeeze(0))

        # 重新组装为 (num_layers=1, batch=n_agents, hidden_size)
        hidden_state = torch.stack(new_hidden_states, dim=0).unsqueeze(0)
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
        
        Args:
            obs: 单Agent观测, shape=(obs_dim,)
            hidden_state: 隐藏状态, shape=(hidden_dim,)
            deterministic: 是否确定性选择
            
        Returns:
            action: 动作
            action_log_prob: 动作对数概率
            hidden_state: 更新后的隐藏状态
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
        if hidden_state is None:
            hidden_state = torch.zeros(
                1, state.size(0), self.hidden_dim, device=self.device
            )
        value, _ = self.critic(state, hidden_state)
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

            value_preds = batch["value_preds"].to(self.device)
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
