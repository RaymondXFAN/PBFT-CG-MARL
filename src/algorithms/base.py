"""
算法基类 - 定义统一接口

所有MARL算法必须继承BaseAlgorithm并实现以下接口：
- act: 批量动作选择
- update: 从batch更新参数
- save/load: 模型保存/加载
- get_action: 单Agent动作选择
- get_value: 集中式Critic价值估计
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod


class BaseAlgorithm(nn.Module, ABC):
    """
    MARL算法基类

    Args:
        config: 算法超参数字典
        env_info: 环境信息字典，包含：
            - n_agents: Agent数量
            - obs_shape: 单Agent观测空间维度
            - state_shape: 全局状态空间维度
            - action_shape: 动作空间维度
            - action_type: 动作类型 "discrete" 或 "continuous"
    """

    def __init__(self, config: dict, env_info: dict):
        super().__init__()
        self.config = config
        self.env_info = env_info

        # 环境信息
        self.n_agents = env_info["n_agents"]
        # 统一 shape 类型：env 可能返回 tuple/list（如 (18,)），算法/网络需要 int
        self.obs_shape = self._shape_to_int(env_info["obs_shape"], "obs_shape")
        self.state_shape = self._shape_to_int(env_info["state_shape"], "state_shape")
        self.action_shape = self._shape_to_int(env_info["action_shape"], "action_shape")
        self.action_type = env_info.get("action_type", "discrete")

        # 设备
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # 训练统计
        self._training_step = 0

    def _shape_to_int(self, shape, name: str = "shape") -> int:
        """
        统一shape类型为int

        环境适配器可能返回 tuple/list（如 (18,)，numpy.shape），算法/网络需要 int
        支持多维shape（如图像(3, 84, 84) → 21168）

        Args:
            shape: tuple / list / int
            name: 字段名（用于调试）

        Returns:
            int: 维度总量
        """
        if isinstance(shape, (list, tuple)):
            result = 1
            for s in shape:
                result *= int(s)
            return result
        if hasattr(shape, '__int__'):  # numpy.int64等
            return int(shape)
        return int(shape)

    @abstractmethod
    def act(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量动作选择

        Args:
            obs: 观测张量, shape=(n_agents, obs_dim) 或 (batch, n_agents, obs_dim)
            hidden_state: 隐藏状态, shape=(n_agents, hidden_dim) 或 (batch, n_agents, hidden_dim)
            deterministic: 是否确定性选择

        Returns:
            actions: 动作张量, shape=(n_agents, ...) 或 (batch, n_agents, ...)
            hidden_state: 更新后的隐藏状态
            action_log_probs: 动作对数概率
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        从batch更新参数

        Args:
            batch: 数据batch字典，包含obs, state, actions, rewards, dones等

        Returns:
            loss_dict: 损失字典
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """加载模型"""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_value(
        self,
        state: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        集中式Critic价值估计

        Args:
            state: 全局状态, shape=(state_dim,) 或 (batch, state_dim)
            hidden_state: 隐藏状态

        Returns:
            value: 价值估计
        """
        pass

    def to_device(self, device: Optional[str] = None) -> None:
        """将模型移动到指定设备"""
        if device is not None:
            self.device = torch.device(device)
        self.to(self.device)