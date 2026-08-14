"""
PBFT共识层 - 核心创新模块

将PBFT拜占庭容错共识协议嵌入MARL，实现共识引导的协作决策。
三阶段共识：Pre-prepare → Prepare → Commit

支持：
- 离散动作：投票机制（多数决定）
- 连续动作：加权平均（权重由prepare阶段相似度决定）
- 拜占庭Agent注入：random, adversarial, silence
- Leader轮换制（Round-Robin）
- 降级策略（fallback）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any


class PBFTConsensusLayer(nn.Module):
    """
    PBFT共识层实现
    
    在MARL中嵌入PBFT共识协议，使得多Agent在执行动作前达成共识，
    从而提升协作效率并容忍拜占庭Agent的干扰。
    
    Args:
        n_agents: Agent数量
        f: 容忍的拜占庭Agent数量，需满足 n_agents >= 3f + 1
        leader_rotation: 是否启用Leader轮换
        use_fallback: 共识失败时是否降级为本地Critic决策
        temperature: 连续动作加权平均的softmax温度
        consensus_threshold: 共识达成阈值（prepare阶段投票比例）
    """

    def __init__(
        self,
        n_agents: int,
        f: int = 1,
        leader_rotation: bool = True,
        use_fallback: bool = True,
        temperature: float = 1.0,
        consensus_threshold: float = 0.5,
        action_dim: int = 5,
    ):
        super().__init__()
        # 自动调整f以满足PBFT约束 n_agents >= 3f+1
        max_f = (n_agents - 1) // 3
        if f > max_f:
            print(f"警告：PBFT需要 n_agents >= 3f+1, 但 n_agents={n_agents}, f={f}. 自动调整f={max_f}")
            f = max_f
        if f < 0:
            f = 0
        self.n_agents = n_agents
        self.f = f
        self.leader_rotation = leader_rotation
        self.use_fallback = use_fallback
        self.temperature = temperature
        self.consensus_threshold = consensus_threshold
        self.action_dim = action_dim

        # 拜占庭Agent相关
        self.byzantine_agents: List[int] = []
        self.byzantine_type: str = "random"

        # Leader状态
        self.current_leader_id: int = 0

        # 共识统计
        self._total_consensus_calls = 0
        self._successful_consensus = 0

    def forward(
        self,
        agent_proposals: List[Tuple[torch.Tensor, torch.Tensor]],
        obs: Optional[torch.Tensor] = None,
        step: int = 0,
        action_type: str = "discrete",
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        PBFT共识前向传播
        
        Args:
            agent_proposals: 每个Agent的(action, log_prob)列表
            obs: 各Agent的观测（用于降级决策），shape=(n_agents, obs_dim)
            step: 当前步数（用于Leader轮换）
            action_type: 动作类型 "discrete" 或 "continuous"
            
        Returns:
            consensus_actions: 共识后的动作列表
            consensus_info: 共识信息字典
        """
        self._total_consensus_calls += 1

        # ===== Phase 1: Pre-prepare =====
        leader_proposal, leader_id = self.pre_prepare(agent_proposals, step)

        # ===== Phase 2: Prepare =====
        prepare_votes, prepare_results = self.prepare(
            leader_proposal, agent_proposals, obs, action_type
        )

        # ===== Phase 3: Commit =====
        consensus_actions, commit_results = self.commit(
            prepare_votes, leader_proposal, agent_proposals, action_type
        )

        # 计算共识率
        consensus_rate = sum(prepare_votes) / self.n_agents

        # 判断共识是否达成
        consensus_achieved = self._check_consensus(prepare_votes)

        if consensus_achieved:
            self._successful_consensus += 1
        elif self.use_fallback and obs is not None:
            # 只对不同意者应用fallback
            for agent_id in range(self.n_agents):
                if not prepare_votes[agent_id] and agent_id not in self.byzantine_agents:
                    fallback_action = self.fallback(
                        obs[agent_id] if obs.dim() > 1 else obs,
                        agent_id, action_type
                    )
                    ref_shape = consensus_actions[0].shape
                    if fallback_action.shape != ref_shape:
                        if fallback_action.numel() == consensus_actions[0].numel():
                            fallback_action = fallback_action.reshape(ref_shape)
                    consensus_actions[agent_id] = fallback_action

        # 计算共识轮数（简化为1轮，实际可多轮）
        consensus_rounds = 1 if consensus_achieved else 2

        # 构建共识信息
        consensus_info = {
            "consensus_rate": consensus_rate,
            "consensus_achieved": consensus_achieved,
            "consensus_rounds": consensus_rounds,
            "leader_id": leader_id,
            "prepare_votes": prepare_votes,
            "phase_results": {
                "pre_prepare": {
                    "leader_id": leader_id,
                    "leader_action": leader_proposal.detach().cpu() if torch.is_tensor(leader_proposal) else leader_proposal,
                },
                "prepare": prepare_results,
                "commit": commit_results,
            },
            "n_byzantine": len(self.byzantine_agents),
            "byzantine_type": self.byzantine_type,
            "consensus_reference": leader_proposal.detach().cpu() if torch.is_tensor(leader_proposal) else leader_proposal,
        }

        return consensus_actions, consensus_info

    def pre_prepare(
        self,
        proposals: List[Tuple[torch.Tensor, torch.Tensor]],
        step: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Pre-prepare阶段：Leader选举 + 提案
        
        Leader从所有Agent的提案中选择一个作为初始提案。
        
        Args:
            proposals: 各Agent的(action, log_prob)列表
            step: 当前步数
            
        Returns:
            leader_proposal: Leader的提案动作
            leader_id: Leader的ID
        """
        # 选举Leader
        leader_id = self.elect_leader(step)
        self.current_leader_id = leader_id

        # Leader的提案就是其自身的动作
        leader_action, leader_log_prob = proposals[leader_id]
        leader_proposal = leader_action

        return leader_proposal, leader_id

    def prepare(
        self,
        leader_proposal: torch.Tensor,
        proposals: List[Tuple[torch.Tensor, torch.Tensor]],
        obs: Optional[torch.Tensor],
        action_type: str = "discrete",
    ) -> Tuple[List[bool], Dict[str, Any]]:
        """
        Prepare阶段：各Agent确认/投票
        
        离散动作：各Agent投票是否同意Leader的提案
        连续动作：计算各Agent提案与Leader提案的相似度
        
        Args:
            leader_proposal: Leader的提案动作
            proposals: 各Agent的(action, log_prob)列表
            obs: 各Agent的观测
            action_type: 动作类型
            
        Returns:
            prepare_votes: 各Agent的投票结果（True=同意）
            prepare_results: Prepare阶段的详细信息
        """
        prepare_votes = []
        vote_details = []
        similarities = []

        for agent_id in range(self.n_agents):
            agent_action, agent_log_prob = proposals[agent_id]

            if agent_id in self.byzantine_agents:
                # 拜占庭Agent的处理
                vote = self._byzantine_prepare_vote(
                    agent_id, leader_proposal, agent_action, action_type
                )
            else:
                # 正常Agent的投票逻辑
                if action_type == "discrete":
                    # 离散动作：直接比较是否与Leader提案一致
                    vote = self._discrete_prepare_vote(
                        agent_action, leader_proposal
                    )
                else:
                    # 连续动作：基于相似度投票
                    similarity = self._compute_similarity(
                        agent_action, leader_proposal
                    )
                    vote = similarity >= self.consensus_threshold
                    similarities.append(similarity)

            prepare_votes.append(vote)
            vote_details.append({
                "agent_id": agent_id,
                "vote": vote,
                "is_byzantine": agent_id in self.byzantine_agents,
            })

        prepare_results = {
            "votes": vote_details,
            "n_agree": sum(prepare_votes),
            "n_disagree": self.n_agents - sum(prepare_votes),
            "similarities": similarities if action_type == "continuous" else None,
        }

        return prepare_votes, prepare_results

    def commit(
        self,
        prepare_votes: List[bool],
        leader_proposal: torch.Tensor,
        proposals: List[Tuple[torch.Tensor, torch.Tensor]],
        action_type: str = "discrete",
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Commit阶段：2f+1确认后执行
        
        满足2f+1个Agent确认后，执行共识动作。
        离散动作：采用多数投票决定
        连续动作：加权平均
        
        Args:
            prepare_votes: 各Agent的投票结果
            leader_proposal: Leader的提案动作
            proposals: 各Agent的(action, log_prob)列表
            action_type: 动作类型
            
        Returns:
            consensus_actions: 共识后的动作列表
            commit_results: Commit阶段的详细信息
        """
        n_agree = sum(prepare_votes)
        quorum = 2 * self.f + 1  # PBFT法定人数
        consensus_achieved = n_agree >= quorum

        if action_type == "discrete":
            consensus_actions, commit_details = self._discrete_commit(
                prepare_votes, leader_proposal, proposals, consensus_achieved
            )
        else:
            consensus_actions, commit_details = self._continuous_commit(
                prepare_votes, leader_proposal, proposals, consensus_achieved
            )

        commit_results = {
            "consensus_achieved": consensus_achieved,
            "n_agree": n_agree,
            "quorum": quorum,
            "details": commit_details,
        }

        return consensus_actions, commit_results

    def elect_leader(self, step: int) -> int:
        """
        Leader选举 - Round-Robin轮换制
        
        Args:
            step: 当前步数
            
        Returns:
            leader_id: Leader的ID
        """
        if self.leader_rotation:
            # Round-Robin轮换，跳过拜占庭Agent
            leader_id = step % self.n_agents
            # 如果选中的是拜占庭Agent，尝试选下一个
            attempts = 0
            while leader_id in self.byzantine_agents and attempts < self.n_agents:
                leader_id = (leader_id + 1) % self.n_agents
                attempts += 1
            return leader_id
        else:
            return self.current_leader_id

    def fallback(self, obs: torch.Tensor, agent_id: int, action_type: str = "discrete") -> torch.Tensor:
        """
        降级策略：当共识失败时，使用本地观测的简单决策
        
        Args:
            obs: Agent的观测
            agent_id: Agent的ID
            action_type: 动作类型 "discrete" 或 "continuous"
            
        Returns:
            action: 降级决策的动作
        """
        # 简单的降级策略：基于观测的线性映射
        # 在实际应用中，这里可以是本地Critic网络
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        # 基于观测的简单决策
        if action_type == "discrete":
            # 离散动作：返回一个整数值（0到action_dim-1）
            score = obs.mean()
            action = torch.tensor(
                min(int(torch.sigmoid(score).item() * self.action_dim), self.action_dim - 1),
                dtype=torch.long,
                device=obs.device,  # 关键修复：fallback 默认建在 CPU，会导致后续 Categorical.log_prob 内部 gather 时设备不一致
            )
        else:
            # 连续动作：返回与观测前几维相同大小的连续值
            # 使用obs的前几维作为动作（简单映射）
            n_dims = obs.size(-1) if obs.dim() > 1 else obs.size(0)
            action = torch.sigmoid(obs.flatten()[:min(n_dims, 5)])
        return action

    def inject_byzantine(
        self, n_byzantine: int, byzantine_type: str = "random"
    ) -> None:
        """
        注入拜占庭Agent
        
        Args:
            n_byzantine: 拜占庭Agent数量
            byzantine_type: 拜占庭类型
                - "random": 随机动作
                - "adversarial": 对抗动作（故意选择最差动作）
                - "silence": 不响应（不参与投票）
        """
        assert n_byzantine <= self.f, (
            f"拜占庭Agent数量({n_byzantine})不能超过容错上限({self.f})"
        )
        self.byzantine_type = byzantine_type
        # 随机选择拜占庭Agent
        if n_byzantine > 0:
            self.byzantine_agents = list(
                np.random.choice(
                    self.n_agents, size=n_byzantine, replace=False
                ).tolist()
            )
        else:
            self.byzantine_agents = []

    def _check_consensus(self, prepare_votes: List[bool]) -> bool:
        """检查是否达成共识（2f+1确认）"""
        n_agree = sum(prepare_votes)
        return n_agree >= 2 * self.f + 1

    def _discrete_prepare_vote(
        self, agent_action: torch.Tensor, leader_proposal: torch.Tensor
    ) -> bool:
        """
        离散动作的Prepare投票：比较Agent动作与Leader提案是否一致
        """
        if agent_action.dim() == 0:
            return agent_action.item() == leader_proposal.item()
        else:
            return torch.equal(agent_action, leader_proposal)

    def _compute_similarity(
        self, agent_action: torch.Tensor, leader_proposal: torch.Tensor
    ) -> float:
        """
        计算连续动作的相似度（余弦相似度）
        """
        if agent_action.dim() == 0:
            return 1.0 if torch.isclose(agent_action, leader_proposal, atol=0.1) else 0.0
        sim = F.cosine_similarity(
            agent_action.unsqueeze(0), leader_proposal.unsqueeze(0), dim=-1
        )
        return sim.item()

    def _byzantine_prepare_vote(
        self,
        agent_id: int,
        leader_proposal: torch.Tensor,
        agent_action: torch.Tensor,
        action_type: str,
    ) -> bool:
        """
        拜占庭Agent在Prepare阶段的投票行为
        """
        if self.byzantine_type == "silence":
            # 不响应，视为反对
            return False
        elif self.byzantine_type == "adversarial":
            # 对抗：故意投反对票
            return False
        else:  # random
            # 随机投票
            return bool(np.random.random() > 0.5)

    def _discrete_commit(
        self,
        prepare_votes: List[bool],
        leader_proposal: torch.Tensor,
        proposals: List[Tuple[torch.Tensor, torch.Tensor]],
        consensus_achieved: bool,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        离散动作的Commit阶段：多数投票决定
        """
        # 收集所有同意的Agent的动作
        action_counts: Dict[Any, int] = {}
        for agent_id, vote in enumerate(prepare_votes):
            if vote and agent_id not in self.byzantine_agents:
                action, _ = proposals[agent_id]
                if action.dim() == 0:
                    key = action.item()
                else:
                    key = tuple(action.detach().cpu().numpy().flatten())
                action_counts[key] = action_counts.get(key, 0) + 1

        if action_counts:
            # 选择获得最多投票的动作
            best_action_key = max(action_counts, key=action_counts.get)
            # 找到对应的原始动作tensor
            best_action = None
            for agent_id, vote in enumerate(prepare_votes):
                if vote and agent_id not in self.byzantine_agents:
                    action, _ = proposals[agent_id]
                    if action.dim() == 0:
                        if action.item() == best_action_key:
                            best_action = action
                            break
                    else:
                        key = tuple(action.detach().cpu().numpy().flatten())
                        if key == best_action_key:
                            best_action = action
                            break
            if best_action is None:
                best_action = leader_proposal
        else:
            best_action = leader_proposal

        # 共识校准：保留每个Agent的原始动作，只校准不同意者
        consensus_actions = []
        for agent_id in range(self.n_agents):
            action, _ = proposals[agent_id]
            if agent_id in self.byzantine_agents:
                # 拜占庭Agent保持原始动作
                consensus_actions.append(action.clone())
            elif not prepare_votes[agent_id]:
                # 不同意leader的Agent：用leader的动作替换（PBFT校正）
                consensus_actions.append(leader_proposal.clone())
            else:
                # 同意leader的Agent：保持原始动作
                consensus_actions.append(action.clone())

        commit_details = {
            "action_counts": {str(k): v for k, v in action_counts.items()},
            "chosen_action": best_action.detach().cpu() if torch.is_tensor(best_action) else best_action,
        }

        return consensus_actions, commit_details

    def _continuous_commit(
        self,
        prepare_votes: List[bool],
        leader_proposal: torch.Tensor,
        proposals: List[Tuple[torch.Tensor, torch.Tensor]],
        consensus_achieved: bool,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        连续动作的Commit阶段：加权平均
        
        权重由prepare阶段的相似度决定
        """
        # 收集同意Agent的动作和权重
        agreeing_actions = []
        agreeing_weights = []

        for agent_id, vote in enumerate(prepare_votes):
            if vote and agent_id not in self.byzantine_agents:
                action, _ = proposals[agent_id]
                agreeing_actions.append(action)
                # 计算与Leader提案的相似度作为权重
                similarity = self._compute_similarity(action, leader_proposal)
                weight = np.exp(similarity / self.temperature)
                agreeing_weights.append(weight)

        if agreeing_actions:
            # 加权平均
            weights_tensor = torch.tensor(
                agreeing_weights, dtype=torch.float32, device=leader_proposal.device
            )
            weights_tensor = weights_tensor / weights_tensor.sum()

            # Stack actions and compute weighted average
            stacked_actions = torch.stack(agreeing_actions)
            if stacked_actions.dim() == 1:
                # 每个action是标量
                consensus_action = (stacked_actions * weights_tensor).sum()
            else:
                # 每个action是向量
                consensus_action = (stacked_actions * weights_tensor.unsqueeze(-1)).sum(dim=0)
        else:
            consensus_action = leader_proposal

        # 共识校准：保留每个Agent的原始动作，软校准不同意者
        blend_ratio = 0.3  # 30%来自共识参考，70%来自自身
        consensus_actions = []
        for agent_id in range(self.n_agents):
            action, _ = proposals[agent_id]
            if agent_id in self.byzantine_agents:
                # 拜占庭Agent保持原始动作
                consensus_actions.append(action.clone())
            elif not prepare_votes[agent_id]:
                # 不同意leader的Agent：软混合
                blended = (1 - blend_ratio) * action + blend_ratio * consensus_action
                consensus_actions.append(blended.clone())
            else:
                # 同意leader的Agent：保持原始动作
                consensus_actions.append(action.clone())

        commit_details = {
            "n_agreeing": len(agreeing_actions),
            "weights": agreeing_weights,
            "consensus_action": consensus_action.detach().cpu(),
        }

        return consensus_actions, commit_details

    def _apply_fallback(
        self,
        consensus_actions: List[torch.Tensor],
        proposals: List[Tuple[torch.Tensor, torch.Tensor]],
        obs: torch.Tensor,
        action_type: str,
    ) -> List[torch.Tensor]:
        """
        降级策略：共识失败时，使用本地Critic决策
        """
        # 获取参考形状（来自proposals中的第一个动作）
        ref_action, _ = proposals[0]
        ref_shape = ref_action.shape
        
        fallback_actions = []
        for agent_id in range(self.n_agents):
            if agent_id in self.byzantine_agents:
                # 拜占庭Agent不参与降级
                action, _ = proposals[agent_id]
                fallback_actions.append(action.reshape(ref_shape))
            else:
                # 正常Agent使用本地Critic决策
                agent_obs = obs[agent_id] if obs.dim() > 1 else obs
                fallback_action = self.fallback(agent_obs, agent_id, action_type)
                # 确保形状与proposals一致
                if fallback_action.shape != ref_shape:
                    if fallback_action.numel() == ref_action.numel():
                        fallback_action = fallback_action.reshape(ref_shape)
                    else:
                        # 如果元素数不匹配，截断或重复来匹配ref_shape
                        n_needed = ref_action.numel()
                        if n_needed == 1:
                            # 目标是标量（离散动作）
                            fallback_action = fallback_action.flatten()[0].reshape(ref_shape)
                        else:
                            # 目标是向量（连续动作），截断或重复
                            flat = fallback_action.flatten()
                            if flat.numel() >= n_needed:
                                fallback_action = flat[:n_needed].reshape(ref_shape)
                            else:
                                # 重复填充
                                repeats = (n_needed + flat.numel() - 1) // flat.numel()
                                fallback_action = flat.repeat(repeats)[:n_needed].reshape(ref_shape)
                fallback_actions.append(fallback_action)
        return fallback_actions

    def get_consensus_stats(self) -> Dict[str, Any]:
        """获取共识统计信息"""
        rate = (
            self._successful_consensus / self._total_consensus_calls
            if self._total_consensus_calls > 0
            else 0.0
        )
        return {
            "total_calls": self._total_consensus_calls,
            "successful_consensus": self._successful_consensus,
            "consensus_success_rate": rate,
            "n_byzantine": len(self.byzantine_agents),
            "byzantine_type": self.byzantine_type,
            "current_leader": self.current_leader_id,
        }

    def reset_stats(self) -> None:
        """重置共识统计"""
        self._total_consensus_calls = 0
        self._successful_consensus = 0
