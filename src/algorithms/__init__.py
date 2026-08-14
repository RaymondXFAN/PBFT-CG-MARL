"""
算法模块 - 注册所有算法，提供统一注册表
"""
from src.algorithms.pbft_cg_mappo import PBFTCGMAPPO
from src.algorithms.mappo import MAPPO
from src.algorithms.qmix import QMIX
from src.algorithms.maddpg import MADDPG
from src.algorithms.commnet import CommNet
from src.algorithms.tarmac import TarMAC

ALGORITHM_REGISTRY = {
    "pbft_cg_mappo": PBFTCGMAPPO,
    "mappo": MAPPO,
    "qmix": QMIX,
    "maddpg": MADDPG,
    "commnet": CommNet,
    "tarmac": TarMAC,
}

__all__ = ["ALGORITHM_REGISTRY", "PBFTCGMAPPO", "MAPPO", "QMIX", "MADDPG", "CommNet", "TarMAC"]
