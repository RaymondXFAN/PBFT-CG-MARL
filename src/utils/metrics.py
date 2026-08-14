"""
评估指标工具

- compute_consensus_rate: 共识达成率
- compute_avg_consensus_rounds: 平均共识轮数
- compute_message_overhead: 消息开销
- compute_byzantine_tolerance: 拜占庭容错度
- compute_performance_drop_rate: 性能下降率
- compute_convergence_speed: 收敛速度
- smooth_curve: 平滑曲线
"""

import numpy as np
from typing import List, Dict, Any, Optional


def compute_consensus_rate(consensus_info_list: List[Dict[str, Any]]) -> float:
    """
    计算共识达成率
    
    在所有时间步中，成功达成共识的比例。
    
    Args:
        consensus_info_list: 共识信息列表，每个元素是consensus_info字典
        
    Returns:
        consensus_rate: 共识达成率，范围[0, 1]
    """
    if not consensus_info_list:
        return 0.0
    n_consensus = sum(
        1 for info in consensus_info_list
        if info.get("consensus_achieved", False)
    )
    return n_consensus / len(consensus_info_list)


def compute_avg_consensus_rounds(consensus_info_list: List[Dict[str, Any]]) -> float:
    """
    计算平均共识轮数
    
    达成共识所需的平均通信轮数。
    
    Args:
        consensus_info_list: 共识信息列表
        
    Returns:
        avg_rounds: 平均共识轮数
    """
    if not consensus_info_list:
        return 0.0
    total_rounds = sum(
        info.get("consensus_rounds", 1) for info in consensus_info_list
    )
    return total_rounds / len(consensus_info_list)


def compute_message_overhead(
    consensus_info_list: List[Dict[str, Any]], n_agents: int
) -> Dict[str, float]:
    """
    计算消息开销
    
    PBFT共识需要额外的通信开销，计算每步的消息数量。
    
    Args:
        consensus_info_list: 共识信息列表
        n_agents: Agent数量
        
    Returns:
        overhead: 消息开销字典
            - total_messages: 总消息数
            - avg_messages_per_step: 每步平均消息数
            - messages_per_step_theoretical: 理论每步消息数（PBFT为O(n^2)）
    """
    n_steps = len(consensus_info_list)
    if n_steps == 0:
        return {
            "total_messages": 0.0,
            "avg_messages_per_step": 0.0,
            "messages_per_step_theoretical": 0.0,
        }

    # PBFT消息数：Pre-prepare(1) + Prepare(n-1) + Commit(n) ≈ 2n
    # 实际每步消息数取决于共识是否达成
    total_messages = 0
    for info in consensus_info_list:
        rounds = info.get("consensus_rounds", 1)
        # 每轮：Pre-prepare(1) + Prepare(n-1) + Commit(n)
        messages_per_round = 1 + (n_agents - 1) + n_agents
        total_messages += messages_per_round * rounds

    avg_messages = total_messages / n_steps
    theoretical = 2 * n_agents  # PBFT理论消息复杂度

    return {
        "total_messages": float(total_messages),
        "avg_messages_per_step": float(avg_messages),
        "messages_per_step_theoretical": float(theoretical),
    }


def compute_byzantine_tolerance(
    rewards_clean: List[float], rewards_byzantine: List[float]
) -> float:
    """
    计算拜占庭容错度
    
    在有拜占庭Agent和无拜占庭Agent时的奖励比值。
    值越接近1.0，说明容错能力越强。
    
    Args:
        rewards_clean: 无拜占庭Agent时的奖励列表
        rewards_byzantine: 有拜占庭Agent时的奖励列表
        
    Returns:
        tolerance: 拜占庭容错度，范围[0, 1+]
    """
    if not rewards_clean or not rewards_byzantine:
        return 0.0
    avg_clean = np.mean(rewards_clean)
    avg_byzantine = np.mean(rewards_byzantine)
    if abs(avg_clean) < 1e-8:
        return 0.0
    return float(np.clip(avg_byzantine / avg_clean, 0.0, 1.0 + 1e-6))


def compute_performance_drop_rate(
    rewards_clean: List[float], rewards_byzantine: List[float]
) -> float:
    """
    计算性能下降率
    
    拜占庭Agent导致的性能下降比例。
    值越接近0，说明容错能力越强。
    
    Args:
        rewards_clean: 无拜占庭Agent时的奖励列表
        rewards_byzantine: 有拜占庭Agent时的奖励列表
        
    Returns:
        drop_rate: 性能下降率，范围[0, 1]
    """
    if not rewards_clean or not rewards_byzantine:
        return 1.0
    avg_clean = np.mean(rewards_clean)
    avg_byzantine = np.mean(rewards_byzantine)
    if abs(avg_clean) < 1e-8:
        return 0.0
    return float(np.clip(1.0 - avg_byzantine / avg_clean, 0.0, 1.0))


def compute_convergence_speed(
    rewards: List[float], threshold: float = 0.9
) -> Dict[str, Any]:
    """
    计算收敛速度
    
    奖励达到最大奖励的threshold比例时所需的episode数。
    
    Args:
        rewards: 奖励曲线
        threshold: 收敛阈值比例
        
    Returns:
        convergence_info: 收敛信息字典
            - convergence_episode: 收敛episode数（-1表示未收敛）
            - max_reward: 最大奖励
            - threshold_reward: 收敛阈值奖励
            - is_converged: 是否收敛
    """
    if not rewards:
        return {
            "convergence_episode": -1,
            "max_reward": 0.0,
            "threshold_reward": 0.0,
            "is_converged": False,
        }

    max_reward = max(rewards)
    threshold_reward = max_reward * threshold

    # 使用滑动窗口判断收敛
    window_size = max(10, len(rewards) // 20)
    convergence_episode = -1

    for i in range(window_size, len(rewards)):
        window_avg = np.mean(rewards[i - window_size : i])
        if window_avg >= threshold_reward:
            convergence_episode = i
            break

    return {
        "convergence_episode": convergence_episode,
        "max_reward": float(max_reward),
        "threshold_reward": float(threshold_reward),
        "is_converged": convergence_episode > 0,
    }


def smooth_curve(values: List[float], window: int = 10) -> List[float]:
    """
    平滑曲线
    
    使用滑动窗口平均进行曲线平滑。
    
    Args:
        values: 原始值列表
        window: 窗口大小
        
    Returns:
        smoothed: 平滑后的值列表
    """
    if not values or window <= 1:
        return values.copy()

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(float(np.mean(values[start:end])))

    return smoothed
