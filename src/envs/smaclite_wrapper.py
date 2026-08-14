"""
SMAClite环境适配器
支持 StarCraft 微操场景（5m_vs_6m, 3s5z等）
"""

import numpy as np
from typing import Dict, Tuple, Any, List, Optional

from src.envs.base import BaseEnv


class SMACliteEnv(BaseEnv):
    """
    SMAClite环境适配器

    使用smaclite包封装StarCraft微操环境。
    当smaclite不可用时，使用自定义的简化版SMAC环境（Numpy实现）。

    Config参数:
        map_name: str, 地图名称（如 "5m_vs_6m", "3s5z"）
        max_steps: int, 最大步数（默认200）
    """

    # 地图配置：地图名称 -> (n_allies, n_enemies, ally_type, enemy_type)
    MAP_CONFIG = {
        "5m_vs_6m": {
            "n_allies": 5,
            "n_enemies": 6,
            "ally_type": "Marine",
            "enemy_type": "Marine",
            "ally_hp": 45,
            "ally_attack": 6,
            "ally_range": 5.0,
            "enemy_hp": 45,
            "enemy_attack": 6,
            "enemy_range": 5.0,
        },
        "3s5z": {
            "n_allies": 8,  # 3 Stalkers + 5 Zealots
            "n_enemies": 8,
            "ally_type": "Stalker_Zealot",
            "enemy_type": "Stalker_Zealot",
            "ally_hp": 80,
            "ally_attack": 10,
            "ally_range": 6.0,
            "enemy_hp": 80,
            "enemy_attack": 10,
            "enemy_range": 6.0,
        },
    }

    # 动作空间: 0=停止, 1-4=上下左右移动, 5=攻击最近敌人
    N_ACTIONS = 6

    def __init__(self, config: dict):
        self.map_name = config.get("map_name", "5m_vs_6m")
        map_cfg = self.MAP_CONFIG.get(self.map_name, self.MAP_CONFIG["5m_vs_6m"])
        config["n_agents"] = map_cfg["n_allies"]
        super().__init__(config)

        self._map_config = map_cfg
        self._n_allies = map_cfg["n_allies"]
        self._n_enemies = map_cfg["n_enemies"]
        self._smaclite_env = None
        self._use_fallback = False

        # 尝试使用smaclite
        try:
            import smaclite
            self._smaclite_env = smaclite.Sc2Env(map_name=self.map_name)
        except (ImportError, Exception) as e:
            print(f"[警告] smaclite不可用({e})，使用自定义简化版SMAC环境")
            self._use_fallback = True

        # 环境状态（fallback用）
        self._ally_pos = None
        self._ally_hp = None
        self._enemy_pos = None
        self._enemy_hp = None
        self._battle_won = False
        self._battle_lost = False

        # 观测维度:
        # 自身属性(2: hp_norm, weapon_cooldown) +
        # 可见敌人信息(n_enemies * 3: hp_norm, distance, visible) +
        # 可见友军信息(n_allies * 3: hp_norm, distance, relative_x, relative_y)
        self._obs_dim = 2 + self._n_enemies * 3 + (self._n_allies - 1) * 4

        # 全局状态维度: 所有单位信息
        # 友军(n_allies * 4: x, y, hp_norm, weapon_cd) + 敌军(n_enemies * 4: x, y, hp_norm, visible)
        self._state_dim = self._n_allies * 4 + self._n_enemies * 4

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置SMAClite环境"""
        self._step_count = 0
        self._battle_won = False
        self._battle_lost = False

        if self._smaclite_env is not None and not self._use_fallback:
            return self._reset_smaclite()
        else:
            return self._reset_fallback()

    def _reset_smaclite(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """使用smaclite重置"""
        obs_list = self._smaclite_env.reset()
        obs_dict = {}
        info_dict = {}
        for i, obs in enumerate(obs_list):
            aid = f"agent_{i}"
            obs_dict[aid] = np.array(obs, dtype=np.float32)
            info_dict[aid] = {}

        # 全局状态
        state = self._smaclite_env.get_state()
        for aid in self._agent_ids:
            info_dict[aid]["state"] = np.array(state, dtype=np.float32)

        self._obs_shape = obs_dict[self._agent_ids[0]].shape
        self._state_shape = info_dict[self._agent_ids[0]]["state"].shape

        return obs_dict, info_dict

    def _reset_fallback(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """使用自定义回退实现重置"""
        # 初始化友军位置（左侧区域）
        self._ally_pos = np.random.uniform(-5.0, 0.0, (self._n_allies, 2))
        self._ally_hp = np.full(self._n_allies, self._map_config["ally_hp"], dtype=np.float32)
        self._ally_weapon_cd = np.zeros(self._n_allies, dtype=np.float32)

        # 初始化敌军位置（右侧区域）
        self._enemy_pos = np.random.uniform(0.0, 5.0, (self._n_enemies, 2))
        self._enemy_hp = np.full(self._n_enemies, self._map_config["enemy_hp"], dtype=np.float32)

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
        """执行一步SMAClite环境交互"""
        self._step_count += 1

        if self._smaclite_env is not None and not self._use_fallback:
            return self._step_smaclite(actions_dict)
        else:
            return self._step_fallback(actions_dict)

    def _step_smaclite(self, actions_dict: Dict[str, Any]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """使用smaclite执行一步"""
        actions = [actions_dict[aid] for aid in self._agent_ids]
        obs_list, rewards, dones, infos = self._smaclite_env.step(actions)

        obs_dict = {}
        rewards_dict = {}
        dones_dict = {}
        infos_dict = {}

        for i, aid in enumerate(self._agent_ids):
            obs_dict[aid] = np.array(obs_list[i], dtype=np.float32)
            rewards_dict[aid] = float(rewards)
            dones_dict[aid] = bool(dones)

        state = self._smaclite_env.get_state()
        for aid in self._agent_ids:
            infos_dict[aid] = {"state": np.array(state, dtype=np.float32)}

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
        move_speed = 0.5
        attack_range = self._map_config["ally_range"]
        attack_damage = self._map_config["ally_attack"]
        enemy_attack = self._map_config["enemy_attack"]
        enemy_range = self._map_config["enemy_range"]

        # 处理友军动作
        for i, aid in enumerate(self._agent_ids):
            if self._ally_hp[i] <= 0:
                continue

            action = int(actions_dict[aid])

            # 武器冷却递减
            if self._ally_weapon_cd[i] > 0:
                self._ally_weapon_cd[i] -= 1

            if action == 0:
                # 停止
                pass
            elif action == 1:
                # 上移
                self._ally_pos[i][1] += move_speed
            elif action == 2:
                # 下移
                self._ally_pos[i][1] -= move_speed
            elif action == 3:
                # 左移
                self._ally_pos[i][0] -= move_speed
            elif action == 4:
                # 右移
                self._ally_pos[i][0] += move_speed
            elif action == 5:
                # 攻击最近敌人
                if self._ally_weapon_cd[i] <= 0:
                    alive_enemies = [j for j in range(self._n_enemies) if self._enemy_hp[j] > 0]
                    if alive_enemies:
                        dists = [np.linalg.norm(self._ally_pos[i] - self._enemy_pos[j]) for j in alive_enemies]
                        nearest_idx = alive_enemies[np.argmin(dists)]
                        if dists[alive_enemies.index(nearest_idx)] <= attack_range:
                            self._enemy_hp[nearest_idx] -= attack_damage
                            self._ally_weapon_cd[i] = 3  # 冷却3步

        # 敌军AI：攻击最近的友军
        for j in range(self._n_enemies):
            if self._enemy_hp[j] <= 0:
                continue
            alive_allies = [i for i in range(self._n_allies) if self._ally_hp[i] > 0]
            if alive_allies:
                dists = [np.linalg.norm(self._enemy_pos[j] - self._ally_pos[i]) for i in alive_allies]
                nearest_idx = alive_allies[np.argmin(dists)]
                if min(dists) <= enemy_range:
                    self._ally_hp[nearest_idx] -= enemy_attack

        # 检查胜负
        self._battle_won = all(hp <= 0 for hp in self._enemy_hp)
        self._battle_lost = all(hp <= 0 for hp in self._ally_hp)

        # 计算奖励
        total_reward = 0.0
        # 击杀敌人奖励
        for j in range(self._n_enemies):
            if self._enemy_hp[j] <= 0:
                total_reward += 10.0
        # 存活奖励
        alive_allies = sum(1 for hp in self._ally_hp if hp > 0)
        total_reward += alive_allies * 0.1
        # 胜利奖励
        if self._battle_won:
            total_reward += 200.0
        # 失败惩罚
        if self._battle_lost:
            total_reward -= 100.0

        # 构建返回
        obs_dict = self._get_fallback_obs()
        state = self._get_fallback_state()
        rewards_dict = {aid: total_reward for aid in self._agent_ids}
        dones_dict = {aid: self._battle_won or self._battle_lost or self._check_truncation()
                      for aid in self._agent_ids}
        infos_dict = {aid: {"state": state.copy(), "battle_won": self._battle_won}
                      for aid in self._agent_ids}

        return obs_dict, rewards_dict, dones_dict, infos_dict

    def _get_fallback_obs(self) -> Dict[str, np.ndarray]:
        """获取回退实现的观测"""
        obs_dict = {}

        for i, aid in enumerate(self._agent_ids):
            obs = []

            # 自身属性
            hp_norm = self._ally_hp[i] / self._map_config["ally_hp"] if self._ally_hp[i] > 0 else 0.0
            weapon_cd = self._ally_weapon_cd[i] / 3.0
            obs.extend([hp_norm, weapon_cd])

            # 敌人信息
            for j in range(self._n_enemies):
                if self._enemy_hp[j] > 0:
                    dist = np.linalg.norm(self._ally_pos[i] - self._enemy_pos[j])
                    enemy_hp_norm = self._enemy_hp[j] / self._map_config["enemy_hp"]
                    obs.extend([enemy_hp_norm, min(dist / 10.0, 1.0), 1.0])  # visible=1
                else:
                    obs.extend([0.0, 1.0, 0.0])  # dead enemy

            # 友军信息
            for j in range(self._n_allies):
                if j != i:
                    if self._ally_hp[j] > 0:
                        rel_pos = (self._ally_pos[j] - self._ally_pos[i]) / 10.0
                        ally_hp_norm = self._ally_hp[j] / self._map_config["ally_hp"]
                        obs.extend([ally_hp_norm, rel_pos[0], rel_pos[1], 1.0])
                    else:
                        obs.extend([0.0, 0.0, 0.0, 0.0])

            obs_dict[aid] = np.array(obs, dtype=np.float32)

        return obs_dict

    def _get_fallback_state(self) -> np.ndarray:
        """获取回退实现的全局状态"""
        state = []

        # 友军信息
        for i in range(self._n_allies):
            hp_norm = self._ally_hp[i] / self._map_config["ally_hp"] if self._ally_hp[i] > 0 else 0.0
            cd_norm = self._ally_weapon_cd[i] / 3.0
            pos_norm = self._ally_pos[i] / 10.0
            state.extend([pos_norm[0], pos_norm[1], hp_norm, cd_norm])

        # 敌军信息
        for j in range(self._n_enemies):
            hp_norm = self._enemy_hp[j] / self._map_config["enemy_hp"] if self._enemy_hp[j] > 0 else 0.0
            pos_norm = self._enemy_pos[j] / 10.0
            visible = 1.0 if self._enemy_hp[j] > 0 else 0.0
            state.extend([pos_norm[0], pos_norm[1], hp_norm, visible])

        return np.array(state, dtype=np.float32)

    def get_env_info(self) -> Dict[str, Any]:
        """获取SMAClite环境信息"""
        return {
            "n_agents": self._n_allies,
            "obs_shape": (self._obs_dim,),
            "state_shape": (self._state_dim,),
            "action_shape": self.N_ACTIONS,
            "action_type": "discrete",
        }

    def close(self):
        """关闭环境"""
        if self._smaclite_env is not None:
            self._smaclite_env.close()
            self._smaclite_env = None
