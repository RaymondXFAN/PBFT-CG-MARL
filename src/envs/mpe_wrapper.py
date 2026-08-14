"""
MPE环境适配器（PettingZoo）
支持 simple_spread 和 simple_reference 场景
"""

import numpy as np
from typing import Dict, Tuple, Any

from src.envs.base import BaseEnv


class MPEEnv(BaseEnv):
    """
    MPE（Multi-Agent Particle Environment）适配器

    使用 PettingZoo 的 parallel API 封装 MPE 环境。
    支持场景:
    - simple_spread: 多Agent协作覆盖目标点
    - simple_reference: 多Agent通信协作

    Config参数:
        scenario: str, "simple_spread" 或 "simple_reference"
        n_agents: int, 智能体数量（默认3）
        max_steps: int, 最大步数（默认100）
        continuous_actions: bool, 是否使用连续动作（默认False）
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.scenario = config.get("scenario", "simple_spread")
        self.continuous_actions = config.get("continuous_actions", False)
        self._env = None
        self._create_env()

    def _create_env(self):
        """创建PettingZoo MPE环境"""
        try:
            from pettingzoo.mpe import simple_spread_v3, simple_reference_v3

            if self.scenario == "simple_spread":
                self._env = simple_spread_v3.parallel_env(
                    N=self._n_agents,
                    max_cycles=self._max_steps,
                    continuous_actions=self.continuous_actions,
                )
            elif self.scenario == "simple_reference":
                self._env = simple_reference_v3.parallel_env(
                    N=self._n_agents,
                    max_cycles=self._max_steps,
                    continuous_actions=self.continuous_actions,
                )
            else:
                raise ValueError(f"未知MPE场景: {self.scenario}")
        except ImportError:
            print("[警告] PettingZoo未安装，使用自定义MPE环境")
            self._env = _FallbackMPE(
                n_agents=self._n_agents,
                scenario=self.scenario,
                max_steps=self._max_steps,
                continuous_actions=self.continuous_actions,
            )

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置MPE环境

        Returns:
            obs_dict: {agent_id: np.array} 局部观测
            info_dict: 包含全局状态
        """
        self._step_count = 0
        if hasattr(self._env, 'reset'):
            obs, infos = self._env.reset()
        else:
            obs, infos = self._env.reset()

        # 标准化agent_id
        obs_dict = {}
        info_dict = {}

        # PettingZoo返回的obs键可能是字符串如"agent_0"
        agent_keys = sorted(obs.keys()) if isinstance(obs, dict) else [f"agent_{i}" for i in range(self._n_agents)]
        self._agent_ids = [f"agent_{i}" for i in range(len(agent_keys))]
        self._n_agents = len(agent_keys)
        self._key_map = {k: f"agent_{i}" for i, k in enumerate(agent_keys)}

        for orig_key, new_key in self._key_map.items():
            obs_dict[new_key] = np.array(obs[orig_key], dtype=np.float32)

        # 构建全局状态：所有Agent观测拼接
        state = np.concatenate([obs_dict[aid] for aid in self._agent_ids], axis=0)

        for aid in self._agent_ids:
            info_dict[aid] = {"state": state.copy()}

        # 缓存环境信息
        self._obs_shape = obs_dict[self._agent_ids[0]].shape
        self._state_shape = state.shape

        return obs_dict, info_dict

    def step(self, actions_dict: Dict[str, Any]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """
        执行一步MPE环境交互

        Args:
            actions_dict: {agent_id: int/np.array} 动作

        Returns:
            obs_dict, rewards_dict, dones_dict, infos_dict
        """
        self._step_count += 1

        # 将标准agent_id映射回原始键
        reverse_map = {v: k for k, v in self._key_map.items()}
        pet_actions = {}
        for aid, action in actions_dict.items():
            orig_key = reverse_map.get(aid, aid)
            if self.continuous_actions:
                pet_actions[orig_key] = np.array(action, dtype=np.float32)
            else:
                pet_actions[orig_key] = int(action)

        obs, rewards, dones, truncations, infos = self._env.step(pet_actions)

        # 转换为标准格式
        obs_dict = {}
        rewards_dict = {}
        dones_dict = {}
        infos_dict = {}

        for orig_key, new_key in self._key_map.items():
            obs_dict[new_key] = np.array(obs[orig_key], dtype=np.float32)
            rewards_dict[new_key] = float(rewards[orig_key])
            dones_dict[new_key] = bool(dones[orig_key]) or bool(truncations[orig_key])

        # 全局状态
        state = np.concatenate([obs_dict[aid] for aid in self._agent_ids], axis=0)

        for aid in self._agent_ids:
            infos_dict[aid] = {"state": state.copy()}

        # 检查截断
        truncated = self._check_truncation()
        if truncated:
            for aid in self._agent_ids:
                dones_dict[aid] = True

        return obs_dict, rewards_dict, dones_dict, infos_dict

    def get_env_info(self) -> Dict[str, Any]:
        """
        获取MPE环境信息

        Returns:
            dict: 包含 n_agents, obs_shape, state_shape, action_shape, action_type
        """
        # 如果还没有缓存shape信息，先做一次reset
        if not hasattr(self, '_obs_shape') or not hasattr(self, '_state_shape'):
            self.reset()

        if self.continuous_actions:
            action_shape = 5  # MPE连续动作维度（5个力方向）
            action_type = "continuous"
        else:
            action_shape = 5  # 离散动作数：不动+4方向
            action_type = "discrete"

        return {
            "n_agents": self._n_agents,
            "obs_shape": self._obs_shape if hasattr(self, '_obs_shape') else (18,),  # 默认值
            "state_shape": self._state_shape if hasattr(self, '_state_shape') else (54,),
            "action_shape": action_shape,
            "action_type": action_type,
        }

    def close(self):
        """关闭MPE环境"""
        if self._env is not None:
            if hasattr(self._env, 'close'):
                self._env.close()
            self._env = None


class _FallbackMPE:
    """
    MPE环境的自定义回退实现
    当PettingZoo不可用时使用，基于Numpy实现简化的MPE场景
    """

    def __init__(self, n_agents: int = 3, scenario: str = "simple_spread",
                 max_steps: int = 100, continuous_actions: bool = False):
        self.n_agents = n_agents
        self.scenario = scenario
        self.max_steps = max_steps
        self.continuous_actions = continuous_actions

        # 环境参数
        self.world_size = 1.0
        self.n_landmarks = n_agents  # 目标点数量等于Agent数量
        self.agent_pos = None
        self.landmark_pos = None
        self.agent_vel = None

        # 观测维度: 自身位置(2) + 速度(2) + 目标相对位置(2*n_landmarks) + 其他Agent相对位置(2*(n_agents-1))
        self.obs_dim = 2 + 2 + 2 * self.n_landmarks + 2 * (self.n_agents - 1)

        # 动作空间
        if self.continuous_actions:
            self.action_dim = 5  # 5维连续力
        else:
            self.action_dim = 5  # 离散动作

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = self.possible_agents.copy()

    def reset(self):
        """重置环境"""
        self.agents = self.possible_agents.copy()

        # 随机初始化Agent位置
        self.agent_pos = np.random.uniform(-0.5, 0.5, (self.n_agents, 2))
        self.agent_vel = np.zeros((self.n_agents, 2))

        # 随机初始化目标点位置
        self.landmark_pos = np.random.uniform(-0.5, 0.5, (self.n_landmarks, 2))

        obs = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def step(self, actions):
        """执行一步"""
        # 处理动作
        for i, agent in enumerate(self.possible_agents):
            if agent in actions:
                action = actions[agent]
                if self.continuous_actions:
                    force = np.array(action, dtype=np.float32)[:2]
                else:
                    # 离散动作: 0=不动, 1=上, 2=下, 3=左, 4=右
                    action_map = {
                        0: np.array([0.0, 0.0]),
                        1: np.array([0.0, 0.1]),
                        2: np.array([0.0, -0.1]),
                        3: np.array([-0.1, 0.0]),
                        4: np.array([0.1, 0.0]),
                    }
                    force = action_map.get(int(action), np.array([0.0, 0.0]))

                # 更新速度和位置
                self.agent_vel[i] = 0.9 * self.agent_vel[i] + force
                self.agent_pos[i] += self.agent_vel[i]
                # 边界约束
                self.agent_pos[i] = np.clip(self.agent_pos[i], -1.0, 1.0)

        # 计算奖励
        rewards = self._compute_rewards()

        # 计算done
        dones = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        obs = self._get_obs()
        infos = {agent: {} for agent in self.agents}

        return obs, rewards, dones, truncations, infos

    def _get_obs(self):
        """获取所有Agent的观测"""
        obs = {}
        for i, agent in enumerate(self.possible_agents):
            # 自身位置和速度
            own_obs = list(self.agent_pos[i]) + list(self.agent_vel[i])

            # 目标相对位置
            for j in range(self.n_landmarks):
                rel_pos = self.landmark_pos[j] - self.agent_pos[i]
                own_obs.extend(list(rel_pos))

            # 其他Agent相对位置
            for j in range(self.n_agents):
                if j != i:
                    rel_pos = self.agent_pos[j] - self.agent_pos[i]
                    own_obs.extend(list(rel_pos))

            obs[agent] = np.array(own_obs, dtype=np.float32)

        return obs

    def _compute_rewards(self):
        """计算奖励"""
        rewards = {}

        if self.scenario == "simple_spread":
            # 合作覆盖奖励：最小化Agent到最近目标点的距离
            total_reward = 0.0
            for j in range(self.n_landmarks):
                # 每个目标点到最近Agent的距离
                min_dist = min(
                    np.linalg.norm(self.agent_pos[i] - self.landmark_pos[j])
                    for i in range(self.n_agents)
                )
                total_reward -= min_dist

            # 碰撞惩罚
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    dist = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                    if dist < 0.1:
                        total_reward -= 1.0

            for agent in self.possible_agents:
                rewards[agent] = total_reward

        elif self.scenario == "simple_reference":
            # 通信协作奖励
            total_reward = 0.0
            for j in range(self.n_landmarks):
                min_dist = min(
                    np.linalg.norm(self.agent_pos[i] - self.landmark_pos[j])
                    for i in range(self.n_agents)
                )
                total_reward -= min_dist
            for agent in self.possible_agents:
                rewards[agent] = total_reward

        return rewards

    def close(self):
        """关闭环境"""
        pass
