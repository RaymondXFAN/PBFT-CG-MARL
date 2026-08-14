"""
经验回放缓冲区

- OnPolicyReplayBuffer: 存储轨迹，支持GAE计算
- OffPolicyReplayBuffer: 存储transitions，支持随机采样
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


class OnPolicyReplayBuffer:
    """
    On-policy经验回放缓冲区
    
    存储完整轨迹，支持GAE（Generalized Advantage Estimation）计算。
    用于MAPPO系列算法。
    
    Args:
        n_agents: Agent数量
        obs_shape: 观测维度
        state_shape: 全局状态维度
        action_shape: 动作维度
        buffer_size: 缓冲区大小（时间步数）
        n_envs: 并行环境数
        action_type: 动作类型
    """

    def __init__(
        self,
        n_agents: int,
        obs_shape: int,
        state_shape: int,
        action_shape: int,
        buffer_size: int = 1000,
        n_envs: int = 1,
        action_type: str = "discrete",
        hidden_dim: int = 64,
    ):
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.action_type = action_type
        self.hidden_dim = hidden_dim

        # 初始化存储
        self.reset()

    def reset(self) -> None:
        """重置缓冲区"""
        self.obs = np.zeros(
            (self.buffer_size, self.n_envs, self.n_agents, self.obs_shape),
            dtype=np.float32,
        )
        self.state = np.zeros(
            (self.buffer_size, self.n_envs, self.state_shape), dtype=np.float32
        )
        # 连续动作: (buffer_size, n_envs, n_agents, action_dim)
        # 离散动作: (buffer_size, n_envs, n_agents)
        if self.action_type == "continuous":
            self.actions = np.zeros(
                (self.buffer_size, self.n_envs, self.n_agents, self.action_shape), dtype=np.float32
            )
        else:
            self.actions = np.zeros(
                (self.buffer_size, self.n_envs, self.n_agents), dtype=np.int64
            )
        self.rewards = np.zeros(
            (self.buffer_size, self.n_envs, 1), dtype=np.float32
        )
        self.dones = np.zeros(
            (self.buffer_size, self.n_envs, 1), dtype=np.float32
        )
        # 连续动作: (buffer_size, n_envs, n_agents)
        # 离散动作: (buffer_size, n_envs, n_agents)
        self.action_log_probs = np.zeros(
            (self.buffer_size, self.n_envs, self.n_agents), dtype=np.float32
        )
        self.hidden_states = np.zeros(
            (self.buffer_size, self.n_envs, self.n_agents, self.hidden_dim), dtype=np.float32
        )
        self.value_preds = np.zeros(
            (self.buffer_size, self.n_envs, 1), dtype=np.float32
        )
        self.returns = np.zeros(
            (self.buffer_size, self.n_envs, 1), dtype=np.float32
        )
        self.advantages = np.zeros(
            (self.buffer_size, self.n_envs, 1), dtype=np.float32
        )

        # 共识信息
        self.consensus_info: List[Dict[str, Any]] = []

        self._ptr = 0
        self._size = 0

    def insert(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        action_log_probs: np.ndarray,
        hidden_states: np.ndarray,
        value_preds: np.ndarray,
        consensus_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        插入一条经验
        
        Args:
            obs: 观测, shape=(n_envs, n_agents, obs_dim)
            state: 全局状态, shape=(n_envs, state_dim)
            actions: 动作, shape=(n_envs, n_agents)
            rewards: 奖励, shape=(n_envs, 1)
            dones: 终止标志, shape=(n_envs, 1)
            action_log_probs: 动作对数概率, shape=(n_envs, n_agents)
            hidden_states: 隐藏状态, shape=(n_envs, n_agents, hidden_dim)
            value_preds: 价值预测, shape=(n_envs, 1)
            consensus_info: 共识信息字典
        """
        self.obs[self._ptr] = obs
        self.state[self._ptr] = state
        self.actions[self._ptr] = actions
        self.rewards[self._ptr] = rewards
        self.dones[self._ptr] = dones
        self.action_log_probs[self._ptr] = action_log_probs
        self.hidden_states[self._ptr] = hidden_states
        self.value_preds[self._ptr] = value_preds

        if consensus_info is not None:
            self.consensus_info.append(consensus_info)

        self._ptr = (self._ptr + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def compute_returns(
        self,
        next_value: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        计算GAE优势和回报
        
        Args:
            next_value: 下一个状态的价值, shape=(n_envs, 1)
            gamma: 折扣因子
            gae_lambda: GAE参数
        """
        gae = 0
        for step in reversed(range(self._size)):
            if step == self._size - 1:
                next_value_step = next_value
                next_non_terminal = 1.0 - self.dones[step]
            else:
                next_value_step = self.value_preds[step + 1]
                next_non_terminal = 1.0 - self.dones[step]

            # TD误差
            delta = (
                self.rewards[step]
                + gamma * next_value_step * next_non_terminal
                - self.value_preds[step]
            )
            # GAE
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.returns[step] = gae + self.value_preds[step]
            self.advantages[step] = gae

    def get_mini_batch(
        self, batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        获取mini-batch数据
        
        Args:
            batch_size: mini-batch大小
            
        Returns:
            batch: 数据字典
        """
        # 随机采样
        indices = np.random.randint(0, self._size, size=batch_size)

        batch = {
            "obs": torch.tensor(self.obs[indices], dtype=torch.float32),
            "state": torch.tensor(self.state[indices], dtype=torch.float32),
            "actions": torch.tensor(
                self.actions[indices],
                dtype=torch.long if self.action_type == "discrete" else torch.float32,
            ),
            "rewards": torch.tensor(self.rewards[indices], dtype=torch.float32),
            "dones": torch.tensor(self.dones[indices], dtype=torch.float32),
            "action_log_probs": torch.tensor(
                self.action_log_probs[indices], dtype=torch.float32
            ),
            "hidden_states": torch.tensor(
                self.hidden_states[indices], dtype=torch.float32
            ),
            "value_preds": torch.tensor(
                self.value_preds[indices], dtype=torch.float32
            ),
            "returns": torch.tensor(self.returns[indices], dtype=torch.float32),
            "advantages": torch.tensor(
                self.advantages[indices], dtype=torch.float32
            ),
        }

        return batch

    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        获取所有数据，将buffer_size和n_envs维度合并为batch维度
        
        Returns:
            batch: 数据字典
        """
        batch = {
            "obs": torch.tensor(self.obs[:self._size], dtype=torch.float32),
            "state": torch.tensor(self.state[:self._size], dtype=torch.float32),
            "actions": torch.tensor(
                self.actions[:self._size],
                dtype=torch.long if self.action_type == "discrete" else torch.float32,
            ),
            "rewards": torch.tensor(self.rewards[:self._size], dtype=torch.float32),
            "dones": torch.tensor(self.dones[:self._size], dtype=torch.float32),
            "action_log_probs": torch.tensor(
                self.action_log_probs[:self._size], dtype=torch.float32
            ),
            "hidden_states": torch.tensor(
                self.hidden_states[:self._size], dtype=torch.float32
            ),
            "value_preds": torch.tensor(
                self.value_preds[:self._size], dtype=torch.float32
            ),
            "returns": torch.tensor(self.returns[:self._size], dtype=torch.float32),
            "advantages": torch.tensor(
                self.advantages[:self._size], dtype=torch.float32
            ),
        }

        # 将buffer_size和n_envs维度合并为batch维度
        # obs: (buffer_size, n_envs, n_agents, obs_dim) -> (batch, n_agents, obs_dim)
        if batch["obs"].dim() == 4:
            batch["obs"] = batch["obs"].reshape(-1, self.n_agents, self.obs_shape)
        # state: (buffer_size, n_envs, state_dim) -> (batch, state_dim)
        if batch["state"].dim() == 3:
            batch["state"] = batch["state"].reshape(-1, self.state_shape)
        # actions: 连续(buffer_size, n_envs, n_agents, action_dim) -> (batch, n_agents, action_dim)
        #          离散(buffer_size, n_envs, n_agents) -> (batch, n_agents)
        if self.action_type == "continuous" and batch["actions"].dim() == 4:
            batch["actions"] = batch["actions"].reshape(-1, self.n_agents, self.action_shape)
        elif batch["actions"].dim() == 3:
            batch["actions"] = batch["actions"].reshape(-1, self.n_agents)
        # rewards: (buffer_size, n_envs, 1) -> (batch, 1)
        if batch["rewards"].dim() == 3:
            batch["rewards"] = batch["rewards"].reshape(-1, 1)
        # dones: (buffer_size, n_envs, 1) -> (batch, 1)
        if batch["dones"].dim() == 3:
            batch["dones"] = batch["dones"].reshape(-1, 1)
        # action_log_probs: (buffer_size, n_envs, n_agents) -> (batch, n_agents)
        if batch["action_log_probs"].dim() == 3:
            batch["action_log_probs"] = batch["action_log_probs"].reshape(-1, self.n_agents)
        # hidden_states: (buffer_size, n_envs, n_agents, hidden_dim) -> (batch, n_agents, hidden_dim)
        if batch["hidden_states"].dim() == 4:
            batch["hidden_states"] = batch["hidden_states"].reshape(-1, self.n_agents, self.hidden_dim)
        # value_preds: (buffer_size, n_envs, 1) -> (batch, 1)
        if batch["value_preds"].dim() == 3:
            batch["value_preds"] = batch["value_preds"].reshape(-1, 1)
        # returns: (buffer_size, n_envs, 1) -> (batch, 1)
        if batch["returns"].dim() == 3:
            batch["returns"] = batch["returns"].reshape(-1, 1)
        # advantages: (buffer_size, n_envs, 1) -> (batch, 1)
        if batch["advantages"].dim() == 3:
            batch["advantages"] = batch["advantages"].reshape(-1, 1)

        return batch

    def after_update(self) -> None:
        """清空缓冲区（在参数更新后调用）"""
        self.reset()

    @property
    def size(self) -> int:
        return self._size


class OffPolicyReplayBuffer:
    """
    Off-policy经验回放缓冲区
    
    存储transitions，支持随机采样。
    用于QMIX、MADDPG等算法。
    
    Args:
        n_agents: Agent数量
        obs_shape: 观测维度
        state_shape: 全局状态维度
        action_shape: 动作维度
        buffer_size: 缓冲区大小
        action_type: 动作类型
    """

    def __init__(
        self,
        n_agents: int,
        obs_shape: int,
        state_shape: int,
        action_shape: int,
        buffer_size: int = 10000,
        action_type: str = "discrete",
    ):
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.action_type = action_type

        # 初始化存储
        self.obs = np.zeros(
            (buffer_size, n_agents, obs_shape), dtype=np.float32
        )
        self.state = np.zeros(
            (buffer_size, state_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (buffer_size, n_agents, action_shape) if action_type == "continuous"
            else (buffer_size, n_agents),
            dtype=np.float32 if action_type == "continuous" else np.int64
        )
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_obs = np.zeros(
            (buffer_size, n_agents, obs_shape), dtype=np.float32
        )
        self.next_state = np.zeros(
            (buffer_size, state_shape), dtype=np.float32
        )
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def insert(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        插入一条transition
        
        Args:
            obs: 观测, shape=(n_agents, obs_dim)
            state: 全局状态, shape=(state_dim,)
            actions: 动作, shape=(n_agents,)
            rewards: 奖励, shape=(1,)
            next_obs: 下一步观测, shape=(n_agents, obs_dim)
            next_state: 下一步全局状态, shape=(state_dim,)
            dones: 终止标志, shape=(1,)
        """
        self.obs[self._ptr] = obs
        self.state[self._ptr] = state
        self.actions[self._ptr] = actions
        self.rewards[self._ptr] = rewards
        self.next_obs[self._ptr] = next_obs
        self.next_state[self._ptr] = next_state
        self.dones[self._ptr] = dones

        self._ptr = (self._ptr + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        随机采样一个batch
        
        Args:
            batch_size: 采样大小
            
        Returns:
            batch: 数据字典
        """
        indices = np.random.randint(0, self._size, size=batch_size)

        batch = {
            "obs": torch.tensor(self.obs[indices], dtype=torch.float32),
            "state": torch.tensor(self.state[indices], dtype=torch.float32),
            "actions": torch.tensor(
                self.actions[indices],
                dtype=torch.long if self.action_type == "discrete" else torch.float32,
            ),
            "rewards": torch.tensor(self.rewards[indices], dtype=torch.float32),
            "next_obs": torch.tensor(self.next_obs[indices], dtype=torch.float32),
            "next_state": torch.tensor(
                self.next_state[indices], dtype=torch.float32
            ),
            "dones": torch.tensor(self.dones[indices], dtype=torch.float32),
        }

        return batch

    @property
    def size(self) -> int:
        return self._size
