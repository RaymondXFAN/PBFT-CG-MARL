"""
PBFT-CG-MARL 基础环境类
定义所有环境适配器必须实现的统一接口
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any


class BaseEnv(ABC):
    """
    多智能体强化学习环境基类

    所有环境适配器必须继承此类并实现其抽象方法。
    统一接口设计，支持集中式训练分布式执行（CTDE）范式。

    数据格式约定:
    - obs_dict: {agent_id: np.array} 每个Agent的局部观测
    - state: np.array 全局状态（用于Centralized Critic）
    - rewards_dict: {agent_id: float} 每个Agent的奖励
    - dones_dict: {agent_id: bool} 每个Agent的终止标志
    - infos_dict: {agent_id: dict} 每个Agent的额外信息
    """

    def __init__(self, config: dict):
        """
        初始化环境

        Args:
            config: 环境配置字典，包含环境特定参数
        """
        self.config = config
        self._n_agents = config.get("n_agents", 3)
        self._max_steps = config.get("max_steps", 100)
        self._step_count = 0
        self._agent_ids = [f"agent_{i}" for i in range(self._n_agents)]

    @abstractmethod
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境，返回初始观测和信息

        Returns:
            obs_dict: {agent_id: np.array} 每个Agent的初始局部观测
            info_dict: {agent_id: dict} 每个Agent的初始信息
                info_dict 中应包含 "state" 键，对应全局状态 np.array
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions_dict: Dict[str, Any]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """
        执行一步环境交互

        Args:
            actions_dict: {agent_id: action} 每个Agent的动作
                离散动作: int
                连续动作: np.array

        Returns:
            obs_dict: {agent_id: np.array} 每个Agent的新局部观测
            rewards_dict: {agent_id: float} 每个Agent的奖励
            dones_dict: {agent_id: bool} 每个Agent的终止标志
            infos_dict: {agent_id: dict} 每个Agent的额外信息
                infos_dict[agent_id] 应包含 "state" 键，对应全局状态
        """
        raise NotImplementedError

    @abstractmethod
    def get_env_info(self) -> Dict[str, Any]:
        """
        获取环境信息，用于算法初始化

        Returns:
            dict: 包含以下键值
                - n_agents: int, 智能体数量
                - obs_shape: tuple, 单个智能体观测空间形状
                - state_shape: tuple, 全局状态空间形状
                - action_shape: int或tuple, 动作空间形状
                - action_type: str, "discrete"或"continuous"
        """
        raise NotImplementedError

    def close(self):
        """关闭环境，释放资源"""
        pass

    @property
    def agent_ids(self) -> list:
        """获取所有智能体ID列表"""
        return self._agent_ids

    @property
    def n_agents(self) -> int:
        """获取智能体数量"""
        return self._n_agents

    @property
    def max_steps(self) -> int:
        """获取最大步数"""
        return self._max_steps

    def _check_truncation(self) -> bool:
        """检查是否达到最大步数（截断）"""
        return self._step_count >= self._max_steps
