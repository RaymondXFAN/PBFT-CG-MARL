"""
评估脚本

- 加载训练好的模型
- 在环境中运行N个episode
- 计算评估指标
- 保存结果到JSON
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.algorithms import ALGORITHM_REGISTRY
from src.utils.metrics import (
    compute_consensus_rate,
    compute_avg_consensus_rounds,
    compute_message_overhead,
    compute_byzantine_tolerance,
    compute_performance_drop_rate,
    compute_convergence_speed,
    smooth_curve,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PBFT-CG-MARL 评估脚本"
    )

    parser.add_argument("--algo", type=str, default="pbft_cg_mappo",
                        choices=list(ALGORITHM_REGISTRY.keys()),
                        help="算法名称")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型文件路径")
    parser.add_argument("--env", type=str, default="simple_spread",
                        help="环境名称")
    parser.add_argument("--n_episodes", type=int, default=50,
                        help="评估episode数")
    parser.add_argument("--seed", type=int, default=1,
                        help="随机种子")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU设备ID")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="结果输出目录")
    parser.add_argument("--byzantine_n", type=int, default=0,
                        help="拜占庭Agent数量（用于评估容错性）")
    parser.add_argument("--byzantine_type", type=str, default="random",
                        choices=["random", "adversarial", "silence"],
                        help="拜占庭类型")
    parser.add_argument("--deterministic", action="store_true",
                        help="是否确定性策略")
    parser.add_argument("--render", action="store_true",
                        help="是否渲染环境")

    return parser.parse_args()


def create_env(env_name: str):
    """
    创建环境（与训练脚本相同的接口）
    """
    try:
        import pettingzoo
        from pettingzoo.mpe import simple_spread_v3, simple_reference_v3, simple_speaker_listener_v4
        from gymnasium import spaces
    except ImportError:
        print("警告：PettingZoo未安装，使用模拟环境")
        from src.train import MockEnv
        return MockEnv(n_agents=4, obs_dim=18, state_dim=72, action_dim=5, action_type="discrete")

    env_map = {
        "simple_spread": simple_spread_v3,
        "simple_reference": simple_reference_v3,
        "simple_speaker_listener": simple_speaker_listener_v4,
    }

    if env_name not in env_map:
        print(f"警告：环境 {env_name} 不在支持列表中，使用模拟环境")
        from src.train import MockEnv
        return MockEnv(n_agents=4, obs_dim=18, state_dim=72, action_dim=5, action_type="discrete")

    from src.train import PettingZooWrapper
    env_creator = env_map[env_name]
    env = env_creator.parallel_env(render_mode="human" if args.render else None)

    env.reset()
    agents = list(env.possible_agents)
    n_agents = len(agents)

    sample_obs = env.observation_space(agents[0])
    sample_action = env.action_space(agents[0])

    if isinstance(sample_obs, spaces.Box):
        obs_shape = sample_obs.shape[0]
    else:
        obs_shape = sample_obs.n

    if isinstance(sample_action, spaces.Discrete):
        action_shape = sample_action.n
        action_type = "discrete"
    elif isinstance(sample_action, spaces.Box):
        action_shape = sample_action.shape[0]
        action_type = "continuous"
    else:
        action_shape = sample_action.n
        action_type = "discrete"

    state_shape = obs_shape * n_agents

    return PettingZooWrapper(
        env=env, n_agents=n_agents, obs_shape=obs_shape,
        state_shape=state_shape, action_shape=action_shape,
        action_type=action_type,
    )


def evaluate_algorithm(
    algorithm,
    env,
    n_episodes: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    评估算法性能
    
    Args:
        algorithm: 算法实例
        env: 环境实例
        n_episodes: 评估episode数
        deterministic: 是否确定性策略
        
    Returns:
        results: 评估结果字典
    """
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    hidden_dim = getattr(algorithm, "hidden_dim", 64)

    episode_rewards = []
    episode_lengths = []
    consensus_info_list = []
    all_actions = []

    algorithm.eval()

    for ep in range(n_episodes):
        obs, state = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algorithm.device)
        hidden_state = torch.zeros(n_agents, hidden_dim, device=algorithm.device)

        episode_reward = 0.0
        episode_length = 0
        episode_actions = []
        done = False

        while not done:
            with torch.no_grad():
                actions, hidden_state, action_log_probs = algorithm.act(
                    obs_tensor, hidden_state, deterministic=deterministic
                )

            if env_info["action_type"] == "discrete":
                actions_np = actions.cpu().numpy()
            else:
                actions_np = actions.cpu().numpy()

            episode_actions.append(actions_np.copy())

            # 获取共识信息
            if hasattr(algorithm, "_last_consensus_info") and algorithm._last_consensus_info is not None:
                consensus_info_list.append(algorithm._last_consensus_info)

            obs, state, rewards, dones, infos = env.step(actions_np)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algorithm.device)

            episode_reward += rewards[0]
            episode_length += 1
            done = dones[0] > 0.5

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        all_actions.append(episode_actions)

    algorithm.train()

    # 计算评估指标
    results = {
        "n_episodes": n_episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
    }

    # 计算共识相关指标
    if consensus_info_list:
        results["consensus_rate"] = compute_consensus_rate(consensus_info_list)
        results["avg_consensus_rounds"] = compute_avg_consensus_rounds(consensus_info_list)
        results["message_overhead"] = compute_message_overhead(
            consensus_info_list, n_agents
        )

    # 如果有PBFT共识统计
    if hasattr(algorithm, "get_consensus_stats"):
        results["consensus_stats"] = algorithm.get_consensus_stats()

    return results


def evaluate_byzantine_tolerance(
    algorithm,
    env,
    n_episodes: int,
    byzantine_n: int,
    byzantine_type: str,
) -> Dict[str, Any]:
    """
    评估拜占庭容错性
    
    Args:
        algorithm: 算法实例
        env: 环境实例
        n_episodes: 评估episode数
        byzantine_n: 拜占庭Agent数量
        byzantine_type: 拜占庭类型
        
    Returns:
        results: 拜占庭容错评估结果
    """
    # 先评估无拜占庭时的性能
    clean_results = evaluate_algorithm(algorithm, env, n_episodes, deterministic=True)

    # 注入拜占庭Agent
    if hasattr(algorithm, "inject_byzantine"):
        algorithm.inject_byzantine(byzantine_n, byzantine_type)

    # 评估有拜占庭时的性能
    byzantine_results = evaluate_algorithm(algorithm, env, n_episodes, deterministic=True)

    # 计算容错指标
    tolerance_results = {
        "byzantine_n": byzantine_n,
        "byzantine_type": byzantine_type,
        "clean_mean_reward": clean_results["mean_reward"],
        "byzantine_mean_reward": byzantine_results["mean_reward"],
        "byzantine_tolerance": compute_byzantine_tolerance(
            clean_results["episode_rewards"],
            byzantine_results["episode_rewards"],
        ),
        "performance_drop_rate": compute_performance_drop_rate(
            clean_results["episode_rewards"],
            byzantine_results["episode_rewards"],
        ),
    }

    return tolerance_results


def main():
    """主函数"""
    global args
    args = parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设置设备
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )

    # 创建环境
    env = create_env(args.env)
    env_info = env.get_env_info()

    print(f"环境信息: {env_info}")

    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get("config", {})
    saved_env_info = checkpoint.get("env_info", env_info)

    # 确保设备一致
    config["device"] = str(device)

    # 创建算法
    algorithm_cls = ALGORITHM_REGISTRY[args.algo]
    algorithm = algorithm_cls(config, saved_env_info)
    algorithm.load(args.model_path)
    algorithm.to(device)
    algorithm.eval()

    print(f"已加载模型: {args.model_path}")

    # 评估
    print(f"开始评估 {args.n_episodes} 个episodes...")
    results = evaluate_algorithm(
        algorithm, env, args.n_episodes, deterministic=args.deterministic
    )

    print(f"\n===== 评估结果 =====")
    print(f"平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"最大奖励: {results['max_reward']:.2f}")
    print(f"最小奖励: {results['min_reward']:.2f}")
    print(f"平均长度: {results['mean_length']:.1f}")

    if "consensus_rate" in results:
        print(f"共识率: {results['consensus_rate']:.4f}")
        print(f"平均共识轮数: {results['avg_consensus_rounds']:.2f}")

    # 拜占庭容错评估
    if args.byzantine_n > 0:
        print(f"\n===== 拜占庭容错评估 =====")
        tolerance_results = evaluate_byzantine_tolerance(
            algorithm, env, args.n_episodes,
            args.byzantine_n, args.byzantine_type,
        )
        results["byzantine_tolerance"] = tolerance_results

        print(f"拜占庭Agent数量: {args.byzantine_n}")
        print(f"拜占庭类型: {args.byzantine_type}")
        print(f"无拜占庭平均奖励: {tolerance_results['clean_mean_reward']:.2f}")
        print(f"有拜占庭平均奖励: {tolerance_results['byzantine_mean_reward']:.2f}")
        print(f"拜占庭容错度: {tolerance_results['byzantine_tolerance']:.4f}")
        print(f"性能下降率: {tolerance_results['performance_drop_rate']:.4f}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"eval_{args.algo}_{args.env}_seed{args.seed}.json"
    )

    # 将numpy数组转为列表
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, list):
            serializable_results[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
        elif isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.floating, np.integer)):
                    serializable_results[key][k] = float(v)
                elif isinstance(v, np.ndarray):
                    serializable_results[key][k] = v.tolist()
                else:
                    serializable_results[key][k] = v
        else:
            serializable_results[key] = value

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n评估结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
