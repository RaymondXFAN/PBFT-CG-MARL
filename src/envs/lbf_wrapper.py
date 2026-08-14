"""
LBF环境适配器（Level-Based Foraging）
支持合作食物采集场景
"""

import numpy as np
from typing import Dict, Tuple, Any

from src.envs.base import BaseEnv


class LBFEnv(BaseEnv):
    """
    LBF（Level-Based Foraging）环境适配器

    使用pettingzoo中的foraging环境或自定义实现。
    支持合作食物采集场景，Agent需要协作采集食物。

    Config参数:
        scenario: str, 场景名称（如 "2s-3f-2co"）
        n_agents: int, 智能体数量（默认2）
        n_food: int, 食物数量（默认3）
        max_steps: int, 最大步数（默认100）
        grid_size: int, 网格大小（默认10）
        force_coop: bool, 是否强制合作（默认True）
    """

    # 动作空间: 0=不动, 1=上, 2=下, 3=左, 4=右, 5=拾取
    N_ACTIONS = 6

    # 动作映射
    ACTION_MAP = {
        0: np.array([0, 0]),   # 不动
        1: np.array([0, 1]),   # 上
        2: np.array([0, -1]),  # 下
        3: np.array([-1, 0]),  # 左
        4: np.array([1, 0]),   # 右
        # 5: 拾取动作
    }

    def __init__(self, config: dict):
        super().__init__(config)
        self.scenario = config.get("scenario", "2s-3f-2co")
        self.n_food = config.get("n_food", 3)
        self.grid_size = config.get("grid_size", 10)
        self.force_coop = config.get("force_coop", True)

        # 解析场景名称
        self._parse_scenario()

        # 尝试使用pettingzoo foraging
        self._pz_env = None
        self._use_fallback = False

        try:
            from pettingzoo.butterfly import cooperative_pong_v5
            # pettingzoo没有直接的LBF，使用自定义实现
            self._use_fallback = True
        except ImportError:
            self._use_fallback = True

        try:
            from lbforaging import ForagingEnv
            self._pz_env = ForagingEnv(
                players=self._n_agents,
                max_player_level=self._max_level,
                field_size=(self.grid_size, self.grid_size),
                n_foods=self.n_food,
                max_food_level=self._max_food_level,
                sight=self._sight,
                force_coop=self.force_coop,
            )
            self._use_fallback = False
        except ImportError:
            print("[警告] lbforaging不可用，使用自定义简化版LBF环境")
            self._use_fallback = True

        # 环境状态
        self._agent_pos = None
        self._agent_level = None
        self._food_pos = None
        self._food_level = None
        self._food_collected = None

        # 观测维度
        self._sight = min(self.grid_size, 5)
        # 自身位置(2) + 等级(1) + 食物信息(n_food * 3: x, y, level) + 其他Agent(n_agents-1 * 3: x, y, level)
        self._obs_dim = 2 + 1 + self.n_food * 3 + (self._n_agents - 1) * 3
        # 全局状态维度
        self._state_dim = self._n_agents * 3 + self.n_food * 3

    def _parse_scenario(self):
        """解析场景名称，如 2s-3f-2co -> 2 agents, 3 food, 2 coop level"""
        parts = self.scenario.split("-")
        if len(parts) >= 3:
            self._n_agents = int(parts[0].replace("s", ""))
            self.n_food = int(parts[1].replace("f", ""))
            self._max_level = int(parts[2].replace("co", "")) if "co" in parts[2] else 1
            self._max_food_level = self._max_level
        else:
            self._n_agents = config.get("n_agents", 2)
            self.n_food = config.get("n_food", 3)
            self._max_level = 2
            self._max_food_level = 2

        self._sight = min(self.grid_size, 5)

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置LBF环境"""
        self._step_count = 0

        if self._pz_env is not None and not self._use_fallback:
            return self._reset_lbforaging()
        else:
            return self._reset_fallback()

    def _reset_lbforaging(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """使用lbforaging重置"""
        obs_list = self._pz_env.reset()
        obs_dict = {}
        info_dict = {}

        for i in range(self._n_agents):
            aid = f"agent_{i}"
            obs_dict[aid] = np.array(obs_list[i], dtype=np.float32).flatten()
            info_dict[aid] = {}

        state = np.concatenate([obs_dict[aid] for aid in self._agent_ids], axis=0)
        for aid in self._agent_ids:
            info_dict[aid]["state"] = state.copy()

        self._obs_shape = obs_dict[self._agent_ids[0]].shape
        self._state_shape = state.shape

        return obs_dict, info_dict

    def _reset_fallback(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """使用自定义回退实现重置"""
        self._agent_pos = np.zeros((self._n_agents, 2), dtype=np.int32)
        self._agent_level = np.ones(self._n_agents, dtype=np.int32)
        self._food_pos = np.zeros((self.n_food, 2), dtype=np.int32)
        self._food_level = np.ones(self.n_food, dtype=np.int32)
        self._food_collected = np.zeros(self.n_food, dtype=bool)

        # 随机初始化Agent位置
        for i in range(self._n_agents):
            while True:
                pos = np.random.randint(0, self.grid_size, 2)
                if not any(np.array_equal(pos, self._agent_pos[j]) for j in range(i)):
                    self._agent_pos[i] = pos
                    break

        # 随机初始化Agent等级
        self._agent_level = np.random.randint(1, self._max_level + 1, self._n_agents)

        # 随机初始化食物位置
        for j in range(self.n_food):
            while True:
                pos = np.random.randint(0, self.grid_size, 2)
                occupied = any(np.array_equal(pos, self._agent_pos[k]) for k in range(self._n_agents))
                occupied = occupied or any(np.array_equal(pos, self._food_pos[k]) for k in range(j))
                if not occupied:
                    self._food_pos[j] = pos
                    break

        # 随机初始化食物等级
        self._food_level = np.random.randint(1, self._max_food_level + 1, self.n_food)

        obs_dict = self._get_fallback_obs()
        state = self._get_fallback_state()
        info_dict = {aid: {"state": state.copy()} for aid in self._agent_ids}

        self._obs_shape = obs_dict[self._agent_ids[0]].shape
        self._state_shape = state.shape

        return obs_dict, info_dict

    def step(self, actions_dict: Dict[str, Any]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """执行一步LBF环境交互"""
        self._step_count += 1

        if self._pz_env is not None and not self._use_fallback:
            return self._step_lbforaging(actions_dict)
        else:
            return self._step_fallback(actions_dict)

    def _step_lbforaging(self, actions_dict: Dict[str, Any]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """使用lbforaging执行一步"""
        actions = [actions_dict[aid] for aid in self._agent_ids]
        obs_list, rewards, dones, infos = self._pz_env.step(actions)

        obs_dict = {}
        rewards_dict = {}
        dones_dict = {}
        infos_dict = {}

        for i, aid in enumerate(self._agent_ids):
            obs_dict[aid] = np.array(obs_list[i], dtype=np.float32).flatten()
            rewards_dict[aid] = float(rewards[i])
            dones_dict[aid] = bool(dones[i])

        state = np.concatenate([obs_dict[aid] for aid in self._agent_ids], axis=0)
        for aid in self._agent_ids:
            infos_dict[aid] = {"state": state.copy()}

        if self._check_truncation():
            for aid in self._agent_ids:
                dones_dict[aid] = True

        return obs_dict, rewards_dict, dones_dict, infos_dict

    def _step_fallback(self, actions_dict: Dict[str, Any]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """使用自定义回退实现执行一步"""
        total_reward = 0.0

        # 处理每个Agent的动作
        for i, aid in enumerate(self._agent_ids):
            action = int(actions_dict[aid])

            if action == 5:
                # 拾取动作
                # 检查当前位置是否有食物
                for j in range(self.n_food):
                    if not self._food_collected[j] and np.array_equal(self._agent_pos[i], self._food_pos[j]):
                        # 检查是否需要合作
                        if self.force_coop and self._food_level[j] > 1:
                            # 需要多个Agent在同一位置且等级之和>=食物等级
                            agents_at_food = [
                                k for k in range(self._n_agents)
                                if np.array_equal(self._agent_pos[k], self._food_pos[j])
                            ]
                            total_level = sum(self._agent_level[k] for k in agents_at_food)
                            if total_level >= self._food_level[j]:
                                self._food_collected[j] = True
                                total_reward += self._food_level[j] * 10.0
                        else:
                            if self._agent_level[i] >= self._food_level[j]:
                                self._food_collected[j] = True
                                total_reward += self._food_level[j] * 10.0
            else:
                # 移动动作
                delta = self.ACTION_MAP.get(action, np.array([0, 0]))
                new_pos = self._agent_pos[i] + delta
                # 边界检查
                new_pos = np.clip(new_pos, 0, self.grid_size - 1)
                self._agent_pos[i] = new_pos

        # 检查是否所有食物都被采集
        all_collected = all(self._food_collected)

        # 构建返回
        obs_dict = self._get_fallback_obs()
        state = self._get_fallback_state()
        rewards_dict = {aid: total_reward for aid in self._agent_ids}
        dones_dict = {aid: all_collected or self._check_truncation() for aid in self._agent_ids}
        infos_dict = {aid: {"state": state.copy()} for aid in self._agent_ids}

        return obs_dict, rewards_dict, dones_dict, infos_dict

    def _get_fallback_obs(self) -> Dict[str, np.ndarray]:
        """获取回退实现的观测"""
        obs_dict = {}

        for i, aid in enumerate(self._agent_ids):
            obs = []
            # 自身位置（归一化）
            obs.extend(list(self._agent_pos[i] / self.grid_size))
            # 自身等级
            obs.append(self._agent_level[i] / self._max_level)

            # 食物信息
            for j in range(self.n_food):
                if not self._food_collected[j]:
                    obs.extend(list(self._food_pos[j] / self.grid_size))
                    obs.append(self._food_level[j] / self._max_food_level)
                else:
                    obs.extend([0.0, 0.0, 0.0])

            # 其他Agent信息
            for j in range(self._n_agents):
                if j != i:
                    rel_pos = (self._agent_pos[j] - self._agent_pos[i]) / self.grid_size
                    obs.extend(list(rel_pos))
                    obs.append(self._agent_level[j] / self._max_level)

            obs_dict[aid] = np.array(obs, dtype=np.float32)

        return obs_dict

    def _get_fallback_state(self) -> np.ndarray:
        """获取回退实现的全局状态"""
        state = []

        for i in range(self._n_agents):
            state.extend(list(self._agent_pos[i] / self.grid_size))
            state.append(self._agent_level[i] / self._max_level)

        for j in range(self.n_food):
            state.extend(list(self._food_pos[j] / self.grid_size))
            state.append(self._food_level[j] / self._max_food_level if not self._food_collected[j] else 0.0)

        return np.array(state, dtype=np.float32)

    def get_env_info(self) -> Dict[str, Any]:
        """获取LBF环境信息"""
        return {
            "n_agents": self._n_agents,
            "obs_shape": (self._obs_dim,),
            "state_shape": (self._state_dim,),
            "action_shape": self.N_ACTIONS,
            "action_type": "discrete",
        }

    def close(self):
        """关闭环境"""
        if self._pz_env is not None:
            self._pz_env.close()
            self._pz_env = None
