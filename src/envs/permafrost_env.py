"""
Permafrost Monitoring Network (PMN) MARL 环境

基于 TPDC 青藏高原多年冻土综合监测数据集(2002-2018)

【场景设计】
- 每个冻土监测站 = 一个 MARL Agent
- 12 个 Agent 协同监测冻土温度异常
- 任务：分布式异常检测 (permafrost temperature anomaly detection)
- 核心挑战：站点观测存在噪声 + 局部观测不完整

【数据流】
- 输入：每个站点的历史地温序列 + 气象数据
- 动作：是否报告"异常" (binary action)
- 奖励：基于集体异常检测的 F1 分数
- PBFT 共识：确保多 Agent 的异常报告一致

【引用】
赵林等. 青藏高原多年冻土综合监测数据集(2002-2018).
国家青藏高原科学数据中心. https://doi.org/10.11888/Geocry.tpdc.271107
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, Any, List, Optional

from src.envs.base import BaseEnv


class PermafrostMonitoringEnv(BaseEnv):
    """
    冻土监测网络 MARL 环境

    状态空间（每个 Agent）：
    - 历史地温（最近 K 天）
    - 当前气象（温度、湿度、降水等）
    - 邻居 Agent 的近期共识结果

    动作空间（每个 Agent）：
    - 0: 报告"正常"
    - 1: 报告"异常"

    奖励：
    - 团队正确检测异常：+1
    - 误报：-0.5
    - 漏报：-1
    - 共识达成：+0.1 (额外奖励)
    """

    def __init__(self, config: dict):
        # 调用 super 之前先决定好 _n_agents
        self.data_path = config.get(
            "data_path",
            "/root/autodl-tmp/tpdc_data/raw/Active layer of ground temperature.xlsx"
        )
        self.n_history = config.get("n_history", 10)
        self.window_size = config.get("window_size", 30)
        self.anomaly_threshold = config.get("anomaly_threshold", 2.0)
        self.n_sites = config.get("n_sites", 12)
        self.episode_length = config.get("episode_length", 100)
        self.use_real_data = config.get("use_real_data", True)

        # 先决定 n_agents（base.py 默认 3，我们要覆盖）
        # 先尝试加载数据以确定站点数
        self.real_data = None
        self.site_names = []

        if self.use_real_data and os.path.exists(self.data_path):
            try:
                self.real_data = self._load_real_data()
                self.site_names = list(self.real_data.keys())
                if len(self.site_names) > self.n_sites:
                    self.site_names = self.site_names[:self.n_sites]
                self._actual_n_agents = len(self.site_names)
            except Exception as e:
                print(f"[警告] 加载真实数据失败: {e}, 使用合成数据")
                self.use_real_data = False
                self._actual_n_agents = self.n_sites
        else:
            self.use_real_data = False
            self._actual_n_agents = self.n_sites

        # 用正确的 n_agents 初始化 base
        config_with_n = {**config, "n_agents": self._actual_n_agents}
        super().__init__(config_with_n)

        # 合成数据 fallback
        if not self.use_real_data:
            self._generate_synthetic_data()

        # 状态空间设计
        # obs: n_history * 4(多深度) + 4(气象) + 3(邻居) = 47 维
        self.obs_dim = self.n_history * 4 + 4 + 3
        self.state_dim = self.obs_dim * self._n_agents

        # 时间游标
        self._current_time = 0
        self._current_step = 0
        self._agent_observations = {}
        self._agent_decisions = {}

        # 当前 episode 的异常状态
        self._true_anomaly = False
        self._anomaly_start = -1
        self._anomaly_site = -1

    def _load_real_data(self) -> Dict[str, np.ndarray]:
        """从 TPDC Excel 加载地温数据"""
        xls = pd.ExcelFile(self.data_path)
        data = {}

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(self.data_path, sheet_name=sheet_name)

            # 提取地温列
            temp_cols = [c for c in df.columns if any(k in str(c) for k in ["GT", "Temp", "Mean"])]
            if not temp_cols:
                continue

            temps = df[temp_cols].values.astype(np.float32)

            # 处理异常值
            temps[temps < -50] = np.nan
            temps[temps > 50] = np.nan
            temps = self._fill_nans(temps)

            data[sheet_name] = temps

        return data

    def _fill_nans(self, data: np.ndarray) -> np.ndarray:
        """线性插值填充 NaN"""
        data = data.copy()
        for col in range(data.shape[1]):
            series = data[:, col]
            mask = np.isnan(series)
            if mask.any() and not mask.all():
                valid_idx = np.where(~mask)[0]
                invalid_idx = np.where(mask)[0]
                if len(valid_idx) > 1:
                    series[invalid_idx] = np.interp(invalid_idx, valid_idx, series[valid_idx])
                else:
                    series[invalid_idx] = series[valid_idx[0]] if len(valid_idx) > 0 else 0.0
                data[:, col] = series
        return data

    def _generate_synthetic_data(self):
        """合成数据 fallback"""
        np.random.seed(42)
        self.real_data = {}
        for i in range(self.n_sites):
            baseline = -5 + np.random.rand() * 5
            t = np.arange(365 * 17)
            seasonal = 5 * np.sin(2 * np.pi * t / 365)
            noise = np.random.randn(len(t)) * 0.5
            warming = 0.02 * t / 365
            temps = baseline + seasonal + noise + warming
            depths = np.column_stack([
                temps + np.random.randn(len(t)) * 0.3,
                temps * 0.8 + np.random.randn(len(t)) * 0.3,
                temps * 0.5 + np.random.randn(len(t)) * 0.3,
                temps * 0.3 + np.random.randn(len(t)) * 0.3,
            ])
            self.real_data[f"site_{i}"] = depths.astype(np.float32)
        self.site_names = list(self.real_data.keys())

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境"""
        max_start = min(len(d) for d in self.real_data.values()) - self.episode_length - self.n_history
        if max_start < 0:
            max_start = 0
        self._current_time = np.random.randint(0, max_start + 1)
        self._current_step = 0

        # 30%概率有真实异常
        self._true_anomaly = np.random.rand() < 0.3
        self._anomaly_start = np.random.randint(20, self.episode_length - 20) if self._true_anomaly else -1
        self._anomaly_site = np.random.randint(0, self._n_agents) if self._true_anomaly else -1

        self._update_observations()

        state = np.concatenate([
            self._agent_observations[aid] for aid in self._agent_ids
        ]).astype(np.float32)

        info_dict = {aid: {"state": state.copy()} for aid in self._agent_ids}
        return self._agent_observations.copy(), info_dict

    def _update_observations(self):
        """更新每个 Agent 的观测"""
        for i, aid in enumerate(self._agent_ids):
            site_data = self.real_data[self.site_names[i]]

            # 历史地温（最近 n_history 个时间步）
            hist_end = self._current_time + self._current_step + self.n_history
            if hist_end <= len(site_data):
                history = site_data[self._current_time + self._current_step:hist_end].flatten()
            else:
                history = np.zeros(self.n_history * site_data.shape[1])
                available = min(self.n_history, len(site_data) - self._current_time - self._current_step)
                if available > 0:
                    end_idx = self._current_time + self._current_step + available
                    history[:available * site_data.shape[1]] = site_data[self._current_time + self._current_step:end_idx].flatten()

            # 气象
            weather = np.array([
                np.sin(2 * np.pi * (self._current_time + self._current_step) / 365) * 10,
                0.3 + 0.2 * np.cos(2 * np.pi * (self._current_time + self._current_step) / 365),
                np.random.rand() * 5,
                max(0, np.random.randn()) * 2,
            ], dtype=np.float32)

            # 邻居共识
            neighbors = [(i - 1) % self._n_agents, (i + 1) % self._n_agents, (i + 2) % self._n_agents]
            neighbor_decisions = np.array([
                self._agent_decisions.get(self._agent_ids[n], 0) for n in neighbors
            ], dtype=np.float32)

            # 注入异常
            if (self._true_anomaly and
                i == self._anomaly_site and
                self._current_step >= self._anomaly_start and
                self._current_step < self._anomaly_start + 10):
                history[-4:] += 3.0

            obs = np.concatenate([history, weather, neighbor_decisions]).astype(np.float32)
            self._agent_observations[aid] = obs

    def step(self, actions_dict):
        """执行一步"""
        self._current_step += 1

        # 记录决策
        for aid, action in actions_dict.items():
            if hasattr(action, 'item'):
                action = action.item()
            self._agent_decisions[aid] = int(action)

        # 集体决策
        abnormal_votes = sum(1 for a in actions_dict.values() if int(a) == 1)
        collective_decision = abnormal_votes > self._n_agents / 2

        # 奖励
        correct = collective_decision == self._true_anomaly
        consensus_bonus = 0.1 if correct else 0

        rewards = {}
        for aid in self._agent_ids:
            if correct:
                rewards[aid] = 1.0 + consensus_bonus
            elif collective_decision and not self._true_anomaly:
                rewards[aid] = -0.5
            else:
                rewards[aid] = -1.0

        self._update_observations()

        state = np.concatenate([
            self._agent_observations[aid] for aid in self._agent_ids
        ]).astype(np.float32)

        done = self._current_step >= self.episode_length
        dones_dict = {aid: done for aid in self._agent_ids}

        info_dict = {
            aid: {
                "state": state.copy(),
                "collective_decision": collective_decision,
                "true_anomaly": self._true_anomaly,
            }
            for aid in self._agent_ids
        }

        return (
            self._agent_observations.copy(),
            rewards,
            dones_dict,
            info_dict,
        )

    def get_env_info(self) -> Dict[str, Any]:
        return {
            "n_agents": self._n_agents,
            "obs_shape": (self.obs_dim,),
            "state_shape": (self.state_dim,),
            "action_shape": 2,
            "action_type": "discrete",
            "site_names": self.site_names,
            "use_real_data": self.use_real_data,
        }

    def close(self):
        pass


if __name__ == "__main__":
    print("测试 PermafrostMonitoringEnv...")
    env = PermafrostMonitoringEnv({'n_sites': 12, 'use_real_data': False})
    info = env.get_env_info()
    print(f"  n_agents: {info['n_agents']}")
    print(f"  obs_shape: {info['obs_shape']}")
    print(f"  sites: {info['site_names'][:3]}...")
    obs, info_dict = env.reset()
    print(f"  reset OK")
    for step in range(5):
        actions = {aid: np.random.randint(0, 2) for aid in obs}
        obs, rewards, dones, infos = env.step(actions)
    print(f"  ✅ 测试通过")