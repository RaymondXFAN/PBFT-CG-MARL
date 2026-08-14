"""
VMAS环境适配器
支持 UAV覆盖 和 编队控制 场景
"""

import numpy as np
from typing import Dict, Tuple, Any

from src.envs.base import BaseEnv


class VMASEnv(BaseEnv):
    """
    VMAS（Vectorized Multi-Agent Simulator）环境适配器

    使用vmas包封装VMAS环境。
    当vmas不可用时，使用自定义的简化版VMAS环境（Numpy实现）。

    支持场景:
    - uav_coverage: 无人机覆盖区域，每个Agent覆盖圆形区域
    - formation: 编队控制，Agent需要保持特定队形

    Config参数:
        scenario: str, "uav_coverage" 或 "formation"
        n_agents: int, 智能体数量（默认4）
        max_steps: int, 最大步数（默认100）
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.scenario = config.get("scenario", "uav_coverage")
        self._vmas_env = None
        self._use_fallback = False

        # 尝试使用vmas
        try:
            import vmas
            self._vmas_env = vmas.make_env(
                scenario=self.scenario,
                num_envs=1,
                n_agents=self._n_agents,
                max_steps=self._max_steps,
                device="cpu",
            )
        except (ImportError, Exception) as e:
            print(f"[警告] vmas不可用({e})，使用自定义简化版VMAS环境")
            self._use_fallback = True

        # 环境参数
        self._world_size = 10.0
        self._coverage_radius = 2.0  # UAV覆盖半径
        self._dt = 0.1  # 时间步长
        self._max_speed = 2.0  # 最大速度

        # 编队目标形状（formation场景）
        if self.scenario == "formation":
            # 正方形编队
            angles = np.linspace(0, 2 * np.pi, self._n_agents, endpoint=False)
            self._formation_target = np.stack([
                2.0 * np.cos(angles),
                2.0 * np.sin(angles)
            ], axis=1)

        # 覆盖目标点（uav_coverage场景）
        self._target_positions = None

        # 环境状态
        self._agent_pos = None
        self._agent_vel = None

        # 观测维度: 自身位置(2) + 速度(2) + 目标相对位置(2)
        self._obs_dim = 2 + 2 + 2
        # 如果是formation，额外包含编队目标相对位置
        if self.scenario == "formation":
            self._obs_dim = 2 + 2 + 2 + 2  # +编队目标

        # 全局状态维度: 所有Agent位置+速度+目标位置
        self._state_dim = self._n_agents * 4 + 2  # 目标中心点

        # 动作维度: 2D力/速度
        self._action_dim = 2

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置VMAS环境"""
        self._step_count = 0

        if self._vmas_env is not None and not self._use_fallback:
            return self._reset_vmas()
        else:
            return self._reset_fallback()

    def _reset_vmas(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """使用vmas重置"""
        obs_list = self._vmas_env.reset()
        obs_dict = {}
        info_dict = {}

        for i in range(self._n_agents):
            aid = f"agent_{i}"
            obs_dict[aid] = np.array(obs_list[i], dtype=np.float32).flatten()
            info_dict[aid] = {}

        # 构建全局状态
        state = np.concatenate([obs_dict[aid] for aid in self._agent_ids], axis=0)
        for aid in self._agent_ids:
            info_dict[aid]["state"] = state.copy()

        self._obs_shape = obs_dict[self._agent_ids[0]].shape
        self._state_shape = state.shape

        return obs_dict, info_dict

    def _reset_fallback(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """使用自定义回退实现重置"""
        # 随机初始化Agent位置
        self._agent_pos = np.random.uniform(
            -self._world_size / 2, self._world_size / 2, (self._n_agents, 2)
        )
        self._agent_vel = np.zeros((self._n_agents, 2))

        # 初始化目标点
        if self.scenario == "uav_coverage":
            # 随机目标区域中心
            self._target_positions = np.random.uniform(
                -3.0, 3.0, (self._n_agents, 2)
            )
        elif self.scenario == "formation":
            # 编队中心
            self._formation_center = np.random.uniform(-2.0, 2.0, 2)
            self._target_positions = self._formation_center + self._formation_target

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
        """执行一步VMAS环境交互"""
        self._step_count += 1

        if self._vmas_env is not None and not self._use_fallback:
            return self._step_vmas(actions_dict)
        else:
            return self._step_fallback(actions_dict)

    def _step_vmas(self, actions_dict: Dict[str, Any]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """使用vmas执行一步"""
        actions = []
        for aid in self._agent_ids:
            action = np.array(actions_dict[aid], dtype=np.float32)
            actions.append(action)

        obs_list, rewards, dones, infos = self._vmas_env.step(actions)

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
        # 处理动作（2D力/速度）
        for i, aid in enumerate(self._agent_ids):
            action = np.array(actions_dict[aid], dtype=np.float32).flatten()
            # 限制动作范围
            action = np.clip(action, -self._max_speed, self._max_speed)

            # 更新速度和位置
            self._agent_vel[i] = 0.8 * self._agent_vel[i] + action * self._dt
            speed = np.linalg.norm(self._agent_vel[i])
            if speed > self._max_speed:
                self._agent_vel[i] = self._agent_vel[i] / speed * self._max_speed

            self._agent_pos[i] += self._agent_vel[i] * self._dt
            # 边界约束
            self._agent_pos[i] = np.clip(
                self._agent_pos[i],
                -self._world_size / 2,
                self._world_size / 2,
            )

        # 计算奖励
        total_reward = self._compute_fallback_reward()

        obs_dict = self._get_fallback_obs()
        state = self._get_fallback_state()
        rewards_dict = {aid: total_reward for aid in self._agent_ids}
        dones_dict = {aid: self._check_truncation() for aid in self._agent_ids}
        infos_dict = {aid: {"state": state.copy()} for aid in self._agent_ids}

        return obs_dict, rewards_dict, dones_dict, infos_dict

    def _compute_fallback_reward(self) -> float:
        """计算回退实现的奖励"""
        total_reward = 0.0

        if self.scenario == "uav_coverage":
            # 覆盖奖励：每个目标点被覆盖的奖励
            for j in range(self._n_agents):
                if self._target_positions is not None:
                    dist = np.linalg.norm(self._agent_pos[j] - self._target_positions[j])
                    total_reward -= dist * 0.5  # 距离惩罚

                    # 覆盖奖励
                    if dist < self._coverage_radius:
                        total_reward += 1.0

            # 碰撞惩罚
            for i in range(self._n_agents):
                for j in range(i + 1, self._n_agents):
                    dist = np.linalg.norm(self._agent_pos[i] - self._agent_pos[j])
                    if dist < 0.5:
                        total_reward -= 2.0

        elif self.scenario == "formation":
            # 编队奖励：Agent到目标编队位置的距离
            for i in range(self._n_agents):
                target_pos = self._formation_center + self._formation_target[i]
                dist = np.linalg.norm(self._agent_pos[i] - target_pos)
                total_reward -= dist * 0.5

                # 编队精度奖励
                if dist < 0.5:
                    total_reward += 1.0

            # 编队保持奖励
            for i in range(self._n_agents):
                for j in range(i + 1, self._n_agents):
                    target_dist = np.linalg.norm(self._formation_target[i] - self._formation_target[j])
                    actual_dist = np.linalg.norm(self._agent_pos[i] - self._agent_pos[j])
                    dist_diff = abs(target_dist - actual_dist)
                    total_reward -= dist_diff * 0.3

        return total_reward

    def _get_fallback_obs(self) -> Dict[str, np.ndarray]:
        """获取回退实现的观测"""
        obs_dict = {}

        for i, aid in enumerate(self._agent_ids):
            obs = []
            # 自身位置
            obs.extend(list(self._agent_pos[i] / self._world_size))
            # 自身速度
            obs.extend(list(self._agent_vel[i] / self._max_speed))

            # 目标相对位置
            if self._target_positions is not None:
                rel_target = (self._target_positions[i] - self._agent_pos[i]) / self._world_size
                obs.extend(list(rel_target))

            if self.scenario == "formation":
                # 编队目标相对位置
                target_pos = self._formation_center + self._formation_target[i]
                rel_formation = (target_pos - self._agent_pos[i]) / self._world_size
                obs.extend(list(rel_formation))

            obs_dict[aid] = np.array(obs, dtype=np.float32)

        return obs_dict

    def _get_fallback_state(self) -> np.ndarray:
        """获取回退实现的全局状态"""
        state = []

        for i in range(self._n_agents):
            state.extend(list(self._agent_pos[i] / self._world_size))
            state.extend(list(self._agent_vel[i] / self._max_speed))

        # 目标中心
        if self._target_positions is not None:
            target_center = np.mean(self._target_positions, axis=0)
            state.extend(list(target_center / self._world_size))

        return np.array(state, dtype=np.float32)

    def get_env_info(self) -> Dict[str, Any]:
        """获取VMAS环境信息"""
        return {
            "n_agents": self._n_agents,
            "obs_shape": (self._obs_dim,),
            "state_shape": (self._state_dim,),
            "action_shape": self._action_dim,
            "action_type": "continuous",
        }

    def close(self):
        """关闭环境"""
        if self._vmas_env is not None:
            self._vmas_env.close()
            self._vmas_env = None
