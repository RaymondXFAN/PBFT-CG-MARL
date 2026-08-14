"""
主训练脚本

支持所有6种算法：PBFT-CG-MAPPO, MAPPO, QMIX, MADDPG, CommNet, TarMAC
支持on-policy和off-policy两种训练循环
支持拜占庭Agent注入
支持WandB日志
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch
import yaml
from typing import Dict, Any, Optional, Tuple

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class NumpyEncoder(json.JSONEncoder):
    """支持 numpy/torch 类型的 JSON encoder"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)


from src.algorithms import ALGORITHM_REGISTRY
from src.utils.buffer import OnPolicyReplayBuffer, OffPolicyReplayBuffer
from src.utils.logger import Logger
from src.utils.metrics import (
    compute_consensus_rate,
    compute_avg_consensus_rounds,
    smooth_curve,
)


class TrainEnvWrapper:
    """
    将BaseEnv的dict接口转换为train.py需要的array接口
    
    BaseEnv.reset() -> (obs_dict, info_dict)
    TrainEnvWrapper.reset() -> (obs_array, state_array)
    
    BaseEnv.step(actions_dict) -> (obs_dict, rewards_dict, dones_dict, infos_dict)
    TrainEnvWrapper.step(actions_np) -> (obs_array, state_array, reward_float, done_bool, info_dict)
    """
    
    def __init__(self, env):
        self._env = env
        self._agent_ids = env.agent_ids
        self._n_agents = env.n_agents
        # 获取env_info并标准化
        self._env_info = _normalize_env_info(env.get_env_info())
    
    def get_env_info(self):
        return self._env_info
    
    def reset(self):
        """返回 (obs_array, state_array)"""
        obs_dict, info_dict = self._env.reset()
        obs_array = np.stack([obs_dict[aid] for aid in self._agent_ids], axis=0)  # (n_agents, obs_dim)
        state = info_dict[self._agent_ids[0]]["state"]  # (state_dim,)
        return obs_array, state
    
    def step(self, actions_np):
        """
        Args:
            actions_np: (n_agents,) for discrete or (n_agents, action_dim) for continuous
        
        Returns:
            obs_array: (n_agents, obs_dim)
            state: (state_dim,)
            reward: float (团队平均奖励)
            done: bool
            info: dict
        """
        # 转为dict格式
        actions_dict = {}
        for i, aid in enumerate(self._agent_ids):
            if self._env_info["action_type"] == "discrete":
                # 离散动作：转为int（兼容各种shape：scalar, (1,), (action_dim,)）
                act = actions_np[i]
                if isinstance(act, (np.ndarray, list)):
                    # 如果是one-hot或logits，取argmax；如果是单元素数组，取其值
                    act_arr = np.asarray(act).flatten()
                    if len(act_arr) > 1:
                        act = int(np.argmax(act_arr))
                    else:
                        act = int(act_arr[0])
                elif hasattr(act, 'item'):
                    act = act.item()
                else:
                    act = int(act)
                actions_dict[aid] = act
            else:
                # 连续动作：转为1D numpy array
                actions_dict[aid] = np.array(actions_np[i], dtype=np.float32).flatten()
        
        obs_dict, rewards_dict, dones_dict, infos_dict = self._env.step(actions_dict)
        
        # 转回array格式
        obs_array = np.stack([obs_dict[aid] for aid in self._agent_ids], axis=0)
        state = infos_dict[self._agent_ids[0]]["state"]
        reward = np.mean([rewards_dict[aid] for aid in self._agent_ids])  # 团队平均奖励
        done = any(dones_dict.values())
        
        # 转为训练循环期望的格式
        reward = np.array([reward], dtype=np.float32)  # (1,)
        done = np.array([float(done)], dtype=np.float32)  # (1,)
        
        return obs_array, state, reward, done, infos_dict
    
    @property
    def agent_ids(self):
        return self._agent_ids
    
    @property
    def n_agents(self):
        return self._n_agents
    
    def close(self):
        self._env.close()


def _normalize_env_info(env_info: dict) -> dict:
    """
    统一env_info格式：确保obs_shape/state_shape/action_shape为int
    
    环境适配器可能返回tuple如(18,)，算法和Buffer需要int如18
    """
    def _to_int(shape):
        if isinstance(shape, (list, tuple)):
            result = 1
            for s in shape:
                result *= s
            return result
        return int(shape)
    
    env_info = dict(env_info)  # 浅拷贝，不修改原始
    env_info["obs_shape"] = _to_int(env_info["obs_shape"])
    env_info["state_shape"] = _to_int(env_info["state_shape"])
    env_info["action_shape"] = _to_int(env_info["action_shape"])
    return env_info


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PBFT-CG-MARL 训练脚本"
    )

    # 算法和环境
    parser.add_argument(
        "--algo", type=str, default="pbft_cg_mappo",
        choices=list(ALGORITHM_REGISTRY.keys()),
        help="算法名称"
    )
    parser.add_argument(
        "--env", type=str, default="simple_spread",
        help="环境名称"
    )
    parser.add_argument(
        "--env_config", type=str, default=None,
        help="环境配置YAML路径"
    )
    parser.add_argument(
        "--algo_config", type=str, default=None,
        help="算法配置YAML路径"
    )

    # 训练参数
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--n_timesteps", type=int, default=1000000, help="总训练步数")
    parser.add_argument("--eval_interval", type=int, default=5000, help="评估间隔")
    parser.add_argument("--eval_episodes", type=int, default=10, help="评估episode数")
    parser.add_argument("--save_interval", type=int, default=20000, help="保存间隔")
    parser.add_argument("--log_dir", type=str, default="./results", help="日志目录")

    # WandB
    parser.add_argument("--use_wandb", action="store_true", help="是否使用WandB")

    # 拜占庭Agent
    parser.add_argument("--byzantine_n", type=int, default=0, help="拜占庭Agent数量")
    parser.add_argument(
        "--byzantine_type", type=str, default="random",
        choices=["random", "adversarial", "silence"],
        help="拜占庭类型"
    )

    # 消融实验
    parser.add_argument(
        "--ablation", type=str, default=None,
        choices=[None, "no_consensus", "no_leader_rotation", "no_fallback",
                 "consensus_threshold_0.3", "consensus_threshold_0.7",
                 "temperature_0.5", "temperature_2.0"],
        help="消融实验设置"
    )

    # 通信频率
    parser.add_argument("--comm_freq", type=int, default=1, help="通信频率（每N步通信一次）")

    # 其他
    parser.add_argument("--gpu", type=int, default=0, help="GPU设备ID")
    parser.add_argument("--share_param", action="store_true", help="是否参数共享")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Optional[str], default_config: dict) -> dict:
    """加载配置文件"""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # 合并默认配置
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        return config
    return default_config


def create_env(env_name: str, env_config: Optional[dict] = None, action_type: str = "discrete"):
    """
    创建环境（PettingZoo风格的parallel API）
    
    返回一个封装了PettingZoo环境的wrapper，提供统一接口：
    - reset() -> obs, state
    - step(actions) -> obs, state, rewards, dones, infos
    - get_env_info() -> dict with n_agents, obs_shape, state_shape, action_shape, action_type
    """
    try:
        import pettingzoo
        from pettingzoo.mpe import simple_spread_v3, simple_reference_v3, simple_speaker_listener_v4
        from gymnasium import spaces
    except ImportError:
        print("警告：PettingZoo未安装，使用模拟环境进行测试")
        if action_type == "continuous":
            return MockEnv(n_agents=4, obs_dim=18, state_dim=72, action_dim=2, action_type="continuous")
        return MockEnv(n_agents=4, obs_dim=18, state_dim=72, action_dim=5, action_type="discrete")

    env_map = {
        "simple_spread": simple_spread_v3,
        "simple_reference": simple_reference_v3,
        "simple_speaker_listener": simple_speaker_listener_v4,
    }

    if env_name not in env_map:
        print(f"警告：环境 {env_name} 不在支持列表中，使用模拟环境")
        return MockEnv(n_agents=4, obs_dim=18, state_dim=72, action_dim=5, action_type="discrete")

    env_creator = env_map[env_name]
    env = env_creator.parallel_env(render_mode=None)

    # 获取环境信息
    env.reset()
    agents = list(env.possible_agents)
    n_agents = len(agents)

    # 获取观测和动作空间
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

    # 全局状态 = 所有Agent观测拼接
    state_shape = obs_shape * n_agents

    return PettingZooWrapper(
        env=env,
        n_agents=n_agents,
        obs_shape=obs_shape,
        state_shape=state_shape,
        action_shape=action_shape,
        action_type=action_type,
    )


class MockEnv:
    """
    模拟环境，用于测试（无需安装PettingZoo）
    """

    def __init__(self, n_agents=3, obs_dim=18, state_dim=54, action_dim=5, action_type="discrete"):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.max_steps = 100
        self._step = 0

    def reset(self):
        self._step = 0
        obs = np.random.randn(self.n_agents, self.obs_dim).astype(np.float32)
        state = np.random.randn(self.state_dim).astype(np.float32)
        return obs, state

    def step(self, actions):
        self._step += 1
        obs = np.random.randn(self.n_agents, self.obs_dim).astype(np.float32)
        state = np.random.randn(self.state_dim).astype(np.float32)
        rewards = np.array([np.random.randn() * 0.1], dtype=np.float32)
        dones = np.array([1.0 if self._step >= self.max_steps else 0.0], dtype=np.float32)
        infos = {}
        return obs, state, rewards, dones, infos

    def get_env_info(self):
        return {
            "n_agents": self.n_agents,
            "obs_shape": self.obs_dim,
            "state_shape": self.state_dim,
            "action_shape": self.action_dim,
            "action_type": self.action_type,
        }


class PettingZooWrapper:
    """
    PettingZoo环境封装器，提供统一接口
    """

    def __init__(self, env, n_agents, obs_shape, state_shape, action_shape, action_type):
        self.env = env
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_type = action_type
        self.agents = list(env.possible_agents)

    def reset(self):
        obs_dict, infos = self.env.reset()
        obs = np.stack([obs_dict[agent] for agent in self.agents], axis=0).astype(np.float32)
        if obs.ndim > 2:
            obs = obs.reshape(self.n_agents, -1)
        state = obs.flatten().astype(np.float32)
        return obs, state

    def step(self, actions):
        # 构建动作字典
        if self.action_type == "discrete":
            action_dict = {
                agent: int(actions[i]) for i, agent in enumerate(self.agents)
            }
        else:
            action_dict = {
                agent: actions[i] for i, agent in enumerate(self.agents)
            }

        obs_dict, reward_dict, terminated_dict, truncated_dict, infos = self.env.step(action_dict)

        # 转换为统一格式
        obs = np.stack([obs_dict.get(agent, np.zeros(self.obs_shape)) for agent in self.agents], axis=0).astype(np.float32)
        if obs.ndim > 2:
            obs = obs.reshape(self.n_agents, -1)
        state = obs.flatten().astype(np.float32)
        rewards = np.array([sum(reward_dict.values())], dtype=np.float32)
        dones = np.array([1.0 if all(terminated_dict.values()) or all(truncated_dict.values()) else 0.0], dtype=np.float32)

        return obs, state, rewards, dones, infos

    def get_env_info(self):
        return {
            "n_agents": self.n_agents,
            "obs_shape": self.obs_shape,
            "state_shape": self.state_shape,
            "action_shape": self.action_shape,
            "action_type": self.action_type,
        }


def train_on_policy(
    algorithm,
    env,
    logger: Logger,
    args,
    config: dict,
) -> None:
    """
    On-policy训练循环
    
    适用于MAPPO系列算法（MAPPO, PBFT-CG-MAPPO, CommNet, TarMAC）
    """
    env_info = _normalize_env_info(env.get_env_info())
    n_agents = env_info["n_agents"]
    obs_shape = env_info["obs_shape"]
    state_shape = env_info["state_shape"]
    action_shape = env_info["action_shape"]
    action_type = env_info["action_type"]

    # 超参数
    gamma = config.get("gamma", 0.99)
    gae_lambda = config.get("gae_lambda", 0.95)
    n_steps = config.get("n_steps", 128)  # 每次采集的步数
    ppo_epoch = config.get("ppo_epoch", 5)
    num_mini_batch = config.get("num_mini_batch", 1)

    # 创建buffer
    buffer = OnPolicyReplayBuffer(
        n_agents=n_agents,
        obs_shape=obs_shape,
        state_shape=state_shape,
        action_shape=action_shape,
        buffer_size=n_steps,
        n_envs=1,
        action_type=action_type,
        hidden_dim=config.get("hidden_dim", 64),
    )

    # 训练统计
    episode_rewards = []
    episode_lengths = []
    consensus_info_list = []
    current_episode_reward = 0.0
    current_episode_length = 0
    episode_count = 0

    # 初始化环境
    obs, state = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algorithm.device)
    state_tensor = torch.tensor(state, dtype=torch.float32, device=algorithm.device)
    # hidden_state 形状: (num_layers=1, batch=n_agents, hidden_size) - GRU 期望格式
    hidden_state = torch.zeros(
        1, n_agents, config.get("hidden_dim", 64), device=algorithm.device
    )

    total_steps = 0
    start_time = time.time()

    logger.info(f"开始On-policy训练: {args.algo}, 环境: {args.env}")
    logger.info(f"环境信息: {env_info}")

    while total_steps < args.n_timesteps:
        # 采集数据
        for step in range(n_steps):
            # 选择动作
            with torch.no_grad():
                actions, new_hidden_state, action_log_probs = algorithm.act(
                    obs_tensor, hidden_state, deterministic=False
                )

            # 获取价值预测
            with torch.no_grad():
                value = algorithm.get_value(state_tensor, None)

            # 执行动作
            if action_type == "discrete":
                actions_np = actions.cpu().numpy()
            else:
                actions_np = actions.cpu().numpy()

            next_obs, next_state, rewards, dones, infos = env.step(actions_np)

            # 获取共识信息
            consensus_info = None
            if hasattr(algorithm, "_last_consensus_info"):
                consensus_info = algorithm._last_consensus_info
                consensus_info_list.append(consensus_info)

            # 存入buffer
            buffer.insert(
                obs=obs[np.newaxis, :, :],
                state=state[np.newaxis, :],
                actions=actions_np[np.newaxis, :],
                rewards=rewards[np.newaxis, :],
                dones=dones[np.newaxis, :],
                action_log_probs=action_log_probs.cpu().numpy()[np.newaxis, :],
                hidden_states=hidden_state.cpu().numpy()[np.newaxis, :, :],
                value_preds=value.cpu().numpy()[np.newaxis, :],
                consensus_info=consensus_info,
            )

            # 更新状态
            obs = next_obs
            state = next_state
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algorithm.device)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=algorithm.device)
            hidden_state = new_hidden_state

            current_episode_reward += rewards[0]
            current_episode_length += 1
            total_steps += 1

            # Episode结束
            if dones[0] > 0.5:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                episode_count += 1

                # 记录日志
                consensus_rate = compute_consensus_rate(consensus_info_list[-100:]) if consensus_info_list else 0.0
                logger.log_episode(
                    episode=episode_count,
                    episode_reward=current_episode_reward,
                    episode_length=current_episode_length,
                    consensus_rate=consensus_rate,
                )

                # 重置环境
                obs, state = env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algorithm.device)
                state_tensor = torch.tensor(state, dtype=torch.float32, device=algorithm.device)
                hidden_state = torch.zeros(1, n_agents, config.get("hidden_dim", 64), device=algorithm.device)
                current_episode_reward = 0.0
                current_episode_length = 0

        # 计算GAE
        with torch.no_grad():
            next_value = algorithm.get_value(state_tensor, None)
        buffer.compute_returns(next_value.cpu().numpy(), gamma, gae_lambda)

        # 更新算法
        batch = buffer.get_all_data()
        loss_dict = algorithm.update(batch)

        # 记录训练指标
        consensus_rate = compute_consensus_rate(consensus_info_list[-100:]) if consensus_info_list else 0.0
        logger.log_training(
            step=total_steps,
            value_loss=loss_dict.get("value_loss", 0.0),
            policy_loss=loss_dict.get("policy_loss", 0.0),
            entropy=loss_dict.get("entropy", 0.0),
            lr=loss_dict.get("lr", 0.0),
            consensus_rate=consensus_rate,
            extra_metrics=loss_dict,
        )

        # 清空buffer
        buffer.after_update()

        # 定期评估
        if total_steps % args.eval_interval == 0:
            eval_reward = evaluate(algorithm, env, args.eval_episodes, config)
            logger.info(
                f"Step {total_steps} | Eval Reward: {eval_reward:.2f} | "
                f"Episode Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}"
            )

        # 定期保存模型
        if total_steps % args.save_interval == 0:
            save_path = os.path.join(args.log_dir, args.algo, f"model_{total_steps}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            algorithm.save(save_path)
            logger.info(f"模型已保存到 {save_path}")

    # 训练结束
    elapsed_time = time.time() - start_time
    logger.info(f"训练完成！总步数: {total_steps}, 耗时: {elapsed_time:.1f}s")

    # 保存最终模型
    save_path = os.path.join(args.log_dir, args.algo, "model_final.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    algorithm.save(save_path)

    # 保存训练曲线
    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "smoothed_rewards": smooth_curve(episode_rewards),
    }
    results_path = os.path.join(args.log_dir, args.algo, "training_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.close()


def train_off_policy(
    algorithm,
    env,
    logger: Logger,
    args,
    config: dict,
) -> None:
    """
    Off-policy训练循环
    
    适用于QMIX、MADDPG等算法
    """
    env_info = _normalize_env_info(env.get_env_info())
    n_agents = env_info["n_agents"]
    obs_shape = env_info["obs_shape"]
    state_shape = env_info["state_shape"]
    action_shape = env_info["action_shape"]
    action_type = env_info["action_type"]

    # 超参数
    gamma = config.get("gamma", 0.99)
    batch_size = config.get("batch_size", 256)
    buffer_size = config.get("buffer_size", 10000)
    learning_starts = config.get("learning_starts", 1000)
    update_interval = config.get("update_interval", 4)

    # 创建buffer
    buffer = OffPolicyReplayBuffer(
        n_agents=n_agents,
        obs_shape=obs_shape,
        state_shape=state_shape,
        action_shape=action_shape,
        buffer_size=buffer_size,
        action_type=action_type,
    )

    # 训练统计
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0.0
    current_episode_length = 0
    episode_count = 0

    # 初始化环境
    obs, state = env.reset()
    hidden_state = torch.zeros(1, n_agents, config.get("hidden_dim", 64), device=algorithm.device)

    total_steps = 0
    start_time = time.time()

    logger.info(f"开始Off-policy训练: {args.algo}, 环境: {args.env}")
    logger.info(f"环境信息: {env_info}")

    while total_steps < args.n_timesteps:
        # 选择动作
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algorithm.device)
        with torch.no_grad():
            actions, hidden_state, _ = algorithm.act(
                obs_tensor, hidden_state, deterministic=False
            )

        # 执行动作
        if action_type == "discrete":
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions.cpu().numpy()

        next_obs, next_state, rewards, dones, infos = env.step(actions_np)

        # 存入buffer
        buffer.insert(
            obs=obs,
            state=state,
            actions=actions_np,
            rewards=rewards,
            next_obs=next_obs,
            next_state=next_state,
            dones=dones,
        )

        # 更新状态
        obs = next_obs
        state = next_state
        current_episode_reward += rewards[0]
        current_episode_length += 1
        total_steps += 1

        # Episode结束
        if dones[0] > 0.5:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            episode_count += 1

            logger.log_episode(
                episode=episode_count,
                episode_reward=current_episode_reward,
                episode_length=current_episode_length,
            )

            # 重置环境
            obs, state = env.reset()
            hidden_state = torch.zeros(1, n_agents, config.get("hidden_dim", 64), device=algorithm.device)
            current_episode_reward = 0.0
            current_episode_length = 0

        # 更新算法
        if buffer.size >= learning_starts and total_steps % update_interval == 0:
            batch = buffer.sample(batch_size)
            loss_dict = algorithm.update(batch)

            if total_steps % 1000 == 0:
                logger.log_training(
                    step=total_steps,
                    extra_metrics=loss_dict,
                )

        # 定期评估
        if total_steps % args.eval_interval == 0:
            eval_reward = evaluate(algorithm, env, args.eval_episodes, config)
            logger.info(
                f"Step {total_steps} | Eval Reward: {eval_reward:.2f} | "
                f"Episode Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}"
            )

        # 定期保存模型
        if total_steps % args.save_interval == 0:
            save_path = os.path.join(args.log_dir, args.algo, f"model_{total_steps}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            algorithm.save(save_path)
            logger.info(f"模型已保存到 {save_path}")

    # 训练结束
    elapsed_time = time.time() - start_time
    logger.info(f"训练完成！总步数: {total_steps}, 耗时: {elapsed_time:.1f}s")

    # 保存最终模型
    save_path = os.path.join(args.log_dir, args.algo, "model_final.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    algorithm.save(save_path)

    # 保存训练曲线
    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "smoothed_rewards": smooth_curve(episode_rewards),
    }
    results_path = os.path.join(args.log_dir, args.algo, "training_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.close()

    # 保存最终指标摘要（便于论文 §5 实验数据提取）
    final_metrics = {
        "episode_reward_final": float(episode_rewards[-1]) if episode_rewards else None,
        "episode_reward_mean_last10": float(np.mean(episode_rewards[-10:])) if episode_rewards else None,
        "smoothed_reward_final": float(smoothed_rewards[-1]) if smoothed_rewards else None,
        "total_steps": int(total_steps),
    }
    metrics_path = os.path.join(args.log_dir, args.algo, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2, cls=NumpyEncoder)

def evaluate(algorithm, env, n_episodes: int, config: dict) -> float:
    """
    评估算法性能
    
    Args:
        algorithm: 算法实例
        env: 环境实例
        n_episodes: 评估episode数
        config: 配置字典
        
    Returns:
        avg_reward: 平均奖励
    """
    env_info = _normalize_env_info(env.get_env_info())
    n_agents = env_info["n_agents"]
    hidden_dim = config.get("hidden_dim", 64)

    total_rewards = []
    algorithm.eval()

    for _ in range(n_episodes):
        obs, state = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algorithm.device)
        # hidden_state 形状: (num_layers=1, batch=n_agents, hidden_size)
        hidden_state = torch.zeros(1, n_agents, hidden_dim, device=algorithm.device)

        episode_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                actions, hidden_state, _ = algorithm.act(
                    obs_tensor, hidden_state, deterministic=True
                )

            if env_info["action_type"] == "discrete":
                actions_np = actions.cpu().numpy()
            else:
                actions_np = actions.cpu().numpy()

            obs, state, rewards, dones, infos = env.step(actions_np)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algorithm.device)
            episode_reward += rewards[0]
            done = dones[0] > 0.5

        total_rewards.append(episode_reward)

    algorithm.train()
    return float(np.mean(total_rewards))


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )

    # 加载环境配置
    env_config = {}
    # 自动加载默认配置（如果没指定 --env_config）
    if args.env_config and os.path.exists(args.env_config):
        with open(args.env_config, "r") as f:
            env_config = yaml.safe_load(f) or {}
    else:
        # 自动加载 configs/env/{args.env}.yaml
        default_config_path = os.path.join(
            project_root, "configs", "env", f"{args.env}.yaml"
        )
        if os.path.exists(default_config_path):
            with open(default_config_path, "r") as f:
                env_config = yaml.safe_load(f) or {}
            print(f"自动加载环境配置: {default_config_path}")

    # 根据算法类型设置动作类型
    continuous_algos = ["maddpg"]
    if args.algo in continuous_algos:
        env_config["continuous_actions"] = True
    
    # 创建环境（优先使用ENV_REGISTRY，fallback到PettingZoo直接创建）
    try:
        from src.envs import ENV_REGISTRY, make_env
        if args.env in ENV_REGISTRY:
            env = make_env(args.env, env_config)
            print(f"使用ENV_REGISTRY创建环境: {args.env}")
        else:
            # fallback到PettingZoo直接创建
            continuous_algos = ["maddpg"]
            env_action_type = "continuous" if args.algo in continuous_algos else "discrete"
            env = create_env(args.env, action_type=env_action_type)
            print(f"使用PettingZoo创建环境: {args.env}")
    except Exception as e:
        print(f"ENV_REGISTRY创建失败: {e}, fallback到PettingZoo")
        continuous_algos = ["maddpg"]
        env_action_type = "continuous" if args.algo in continuous_algos else "discrete"
        env = create_env(args.env, action_type=env_action_type)
    # 用TrainEnvWrapper包裹，统一dict→array接口
    env = TrainEnvWrapper(env)
    env_info = env.get_env_info()

    print(f"环境信息: {env_info}")

    # 默认算法配置
    default_config = {
        "lr": 0.0005,
        "clip_ratio": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ppo_epoch": 5,
        "num_mini_batch": 1,
        "max_grad_norm": 0.5,
        "hidden_dim": 64,
        "use_rnn": True,
        "share_param": args.share_param,
        "device": str(device),
        "n_steps": 128,
        "batch_size": 256,
        "buffer_size": 10000,
        "learning_starts": 1000,
        "update_interval": 4,
        # PBFT-CG-MAPPO特定参数
        "pbft_f": 1,
        "leader_rotation": True,
        "use_fallback": True,
        "consensus_temperature": 1.0,
        "consensus_threshold": 0.5,
        "consensus_loss_coef": 0.1,
        # QMIX特定参数
        "epsilon_start": 1.0,
        "epsilon_finish": 0.05,
        "epsilon_anneal_time": 50000,
        "mixing_embed_dim": 32,
        "hyper_hidden_dim": 64,
        "target_update_interval": 200,
        "tau": 0.005,
        # MADDPG特定参数
        "lr_actor": 0.0001,
        "lr_critic": 0.001,
        "action_limit": 1.0,
        "noise_std": 0.1,
        # CommNet/TarMAC特定参数
        "comm_steps": 1,
    }

    # 加载配置
    config = load_config(args.algo_config, default_config)
    config["share_param"] = args.share_param
    config["device"] = str(device)

    # 展平嵌套配置（如pbft: {f: 1, ...} → pbft_f: 1, ...）
    if "pbft" in config and isinstance(config["pbft"], dict):
        pbft_cfg = config.pop("pbft")
        for k, v in pbft_cfg.items():
            config[f"pbft_{k}"] = v

    # 创建算法
    algorithm_cls = ALGORITHM_REGISTRY[args.algo]
    algorithm = algorithm_cls(config, env_info)
    algorithm.to(device)
    print(f"[DEBUG] 设备: {device}, 算法: {next(algorithm.parameters()).device}")

    # 注入拜占庭Agent
    if args.byzantine_n > 0 and hasattr(algorithm, "inject_byzantine"):
        algorithm.inject_byzantine(args.byzantine_n, args.byzantine_type)
        print(f"已注入 {args.byzantine_n} 个拜占庭Agent (类型: {args.byzantine_type})")

    # 应用消融实验设置（直接修改共识层属性，而非仅改config字典）
    if args.ablation and hasattr(algorithm, "pbft_consensus"):
        if args.ablation == "no_consensus":
            # 跳过共识：阈值设为0（所有提案自动通过）+ 关闭共识损失
            algorithm.pbft_consensus.consensus_threshold = 0.0
            algorithm.consensus_loss_coef = 0.0
            algorithm.config["consensus_threshold"] = 0.0
            algorithm.config["consensus_loss_coef"] = 0.0
        elif args.ablation == "no_leader_rotation":
            # 固定Leader：关闭轮换
            algorithm.pbft_consensus.leader_rotation = False
            algorithm.config["leader_rotation"] = False
        elif args.ablation == "no_fallback":
            # 关闭降级策略
            algorithm.pbft_consensus.use_fallback = False
            algorithm.config["use_fallback"] = False
        elif args.ablation.startswith("consensus_threshold_"):
            threshold = float(args.ablation.split("_")[-1])
            algorithm.pbft_consensus.consensus_threshold = threshold
            algorithm.config["consensus_threshold"] = threshold
        elif args.ablation.startswith("temperature_"):
            temp = float(args.ablation.split("_")[-1])
            algorithm.pbft_consensus.temperature = temp
            algorithm.config["consensus_temperature"] = temp
        print(f"已应用消融设置: {args.ablation}")

    # 创建日志记录器
    log_dir = os.path.join(args.log_dir, args.algo, f"seed_{args.seed}")
    wandb_config = {
        "algo": args.algo,
        "env": args.env,
        "seed": args.seed,
        "config": config,
    }
    logger = Logger(
        log_dir=log_dir,
        use_wandb=args.use_wandb,
        wandb_project="PBFT-CG-MARL",
        wandb_config=wandb_config,
    )

    # 选择训练循环
    on_policy_algos = ["pbft_cg_mappo", "mappo", "commnet", "tarmac"]
    off_policy_algos = ["qmix", "maddpg"]

    if args.algo in on_policy_algos:
        train_on_policy(algorithm, env, logger, args, config)
    elif args.algo in off_policy_algos:
        train_off_policy(algorithm, env, logger, args, config)
    else:
        raise ValueError(f"未知算法: {args.algo}")


if __name__ == "__main__":
    main()
