"""
PBFT-CG-MARL 环境模块
注册所有环境适配器，提供统一的 ENV_REGISTRY 字典
"""

from src.envs.base import BaseEnv
from src.envs.mpe_wrapper import MPEEnv
from src.envs.smaclite_wrapper import SMACliteEnv
from src.envs.vmas_wrapper import VMASEnv
from src.envs.lbf_wrapper import LBFEnv
from src.envs.permafrost_env import PermafrostMonitoringEnv

# 环境注册表：环境名称 -> 环境类
ENV_REGISTRY = {
    "mpe_spread": MPEEnv,
    "mpe_reference": MPEEnv,
    "smaclite_5m_vs_6m": SMACliteEnv,
    "smaclite_3s5z": SMACliteEnv,
    "vmas_uav_coverage": VMASEnv,
    "vmas_formation": VMASEnv,
    "lbf_2s3f": LBFEnv,
    # 真实数据环境：青藏高原冻土监测网络
    "permafrost_monitoring": PermafrostMonitoringEnv,
}


def make_env(env_name: str, config: dict = None) -> BaseEnv:
    """
    根据环境名称创建环境实例

    Args:
        env_name: 环境名称，必须在 ENV_REGISTRY 中
        config: 环境配置字典

    Returns:
        BaseEnv 实例

    Raises:
        ValueError: 如果环境名称不在注册表中
    """
    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"未知环境: {env_name}. 可用环境: {list(ENV_REGISTRY.keys())}"
        )
    env_cls = ENV_REGISTRY[env_name]
    if config is None:
        config = {}
    return env_cls(config)
