# PBFT-CG-MAPPO 论文顶层设计 v0.4 (修订)

> 2026-08-12 | 基于两篇参考论文（Sensors 2025 / Future Internet 2025）骨架
> 修订：第一章篇幅从 2000 词压到 1100 词；实验章升为最大；参考文献改为灵活区间优先 2024+

---

## 一、全文结构（7 章）

```
Abstract (两段式, ~250 词)
  段1: 部署痛点 → 现有方法不足 → 本文方法
  段2: 实验设置 → 核心结果(待填) → 价值

1. Introduction (含 Background, 不分节, ~1100 词)   ← 压缩
   - 背景 → 不足(i)(ii)(iii) → 预备知识(一句带过) → 目的意义 → 方法概述 → ●贡献 → 导览

2. Related Work (~1300 词)
   2.1 Robust MARL  2.2 Comm-Efficient MARL  2.3 BFT & Consensus  2.4 MARL Env Monitoring  2.5 Gap

3. PBFT-CG-MAPPO: Proposed Framework (~1600 词)   ← 主体
   3.1 System Architecture  3.2 CG Layer(核心)  3.3 CTDE-MAPPO w/ Filtered Msgs
   3.4 Byzantine Robustness Analysis (f<n/3)  3.5 Communication Model

4. Algorithms (~500 词)
   4.1 PBFT Commit (Algo 1)  4.2 PBFT-CG-MAPPO Training (Algo 2)  4.3 Complexity & Overhead

5. Experimental Results (~2000 词, 待填)   ← 最大章
   5.1 Env & Datasets  5.2 Setup  5.3 Baselines  5.4 Byzantine Eval
   5.5 Ablation  5.6 Comm Efficiency  5.7 Permafrost Transfer  5.8 Discussion & Limits

6. Discussion (~400 词)
7. Conclusions (~300 词)

Appendix A / Data Availability / Declarations
```

---

## 二、篇幅分配（全文 ~8000 词，SCI 内行比例）

| 章节 | 目标词数 | 占比 | 说明 |
|------|---------|------|------|
| Abstract | 250 | 3% | 两段式，段2待填 |
| **1. Introduction (含BG)** | **1100** | **14%** | ← 修订：从2000压到1100，预备知识不展开 |
| 2. Related Work | 1300 | 16% | 2.5 Gap收口 |
| **3. Proposed Framework** | **1600** | **20%** | 核心创新，数学细节在此展开 |
| 4. Algorithms | 500 | 6% | 伪代码+开销 |
| **5. Experiments** | **2000** | **25%** | ← 修订：升为最大章，方法/实验是论文主体 |
| 6. Discussion | 400 | 5% | 独立章 |
| 7. Conclusions | 300 | 4% | 精炼 |
| 其他(过渡/声明) | 550 | 7% | |
| **总计** | **~8000** | **100%** | |

**调整逻辑**：原设计把预备知识（Dec-POMDP/CTDE/PBFT）在第一章用 500 词展开，挤占了 Intro 合理篇幅且方法论细节前置错乱。修订后第一章只给**定义式一句话**，数学/算法细节全部留给 §3-§4，使 Method(20%)+Exp(25%) 成为主体——这符合 SCI 实证论文"方法可复现、实验充分"的审稿期待。

---

## 三、参考文献（灵活区间，优先 2024+）

**不再硬性卡 30+10**。原则：
- 总量约 **35–45 篇**，按需增减
- **优先 2024 年及以后新文献**（鲁棒MARL/BFT/通信/冻土应用）
- 不相关的经典文献（如纯区块链BFT、无关ML综述）不强行引用
- 必引：PBFT 原始(1999)、MAPPO(2022)、Dec-POMDP 原始、各基线原始论文（这些是方法论根基，虽老但必须）

**预计分布**：
- 第一章（背景+基线+竞争者+数据）：~15-18 篇
- 第二章（Related Work）：~10-12 篇
- 第三/四章（方法/算法）：~2-3 篇
- 第五章（§5.1 环境 + §5.7 数据集对比 + 基线）：~5-8 篇
- 其他：~2-3 篇

**核心 2024+ 文献（已检索确认）**：
- Li et al. Byzantine Robust Cooperative MARL as Bayesian Game. **ICLR 2024**
- Xie et al. Communication-Efficient and Resilient Distributed Q-Learning. **IEEE TNNLS 2024**
- Ye et al. Resilient Multiagent RL with Function Approximation. **IEEE TAC 2024**
- Ding et al. Multi-Agent Coordination via Multi-Level Communication (SeqComm). **NeurIPS 2024**
- Lee et al. Wolfpack Adversarial Attack for Robust MARL. **ICML 2025**
- Nanapu et al. D2BFT: Dual Byzantine Fault Tolerance for Drone Surveillance with DRL. **Computer Networks 2025**
- CC-MARL: Communication-Constrained Priors. **NeurIPS 2025**
- Lee et al. Fully Byzantine-Resilient Distributed Multi-Agent Q-Learning. **arXiv 2026**
- RIAHI et al. Robust Lightweight Consensus for Permissioned Blockchains. **2025**
- A Dynamic Adaptive Framework for PBFT in IoT. **IEEE TC 2024**
- TL-PBFT: Tree-layered PBFT. **TIIS 2025**
- Enhanced Detection of Permafrost Deformation with ML+InSAR along QTP. **Remote Sensing 2025**
- Real-Time Depth-Resolved Permafrost Thermal Monitoring via Bayesian-Optimized DL. **2025**
- Machine learning-based prediction of permafrost degradation: review. **2024**

---

## 四、第一章（Intro+Background合并）精简结构

> 不分节，段落自然过渡，预备知识**只给定义不展开**。

**P1-2 研究背景**（~250 词）
- 多智能体从仿真走向物理部署（无人机群/冻土监测站网）
- 真实链路不可靠：丢包、间歇、拜占庭节点
- 场景锚定：青藏高原冻土站网暴风雪中丢包、节点冻结

**P3-4 现有不足**（~200 词）
- CTDE 假设诚实链路 → 单条毒消息劫持 critic
- (i) 对抗训练仅经验鲁棒，无形式化保证
- (ii) 通信高效方法(CommNet/TarMAC)仍信任每条消息
- (iii) 经典BFT从未嵌入MARL训练循环

**P5-6 预备知识（一句话定义）**（~200 词）
- Dec-POMDP [3][17]：协作任务形式化为分散POMDP（一句话）
- CTDE-MAPPO [2][5][6]：GRU actor + 集中critic + PPO裁剪（一句话）
- PBFT [1]：三阶段提交(pre-prepare/prepare/commit) + f<n/3容错（一句话）
- *注：数学细节留 §3*

**P7 目的与意义**（~150 词）
- "The primary objective of this research is to design and evaluate a MARL method carrying a formal Byzantine bound yet fully trainable."
- 理论意义 + 应用意义（冻土监测 [13][14]）

**P8 方法概述**（~150 词）
- PBFT-CG-MAPPO：CTDE-MAPPO 通信路径嵌 PBFT 共识层，quorum(2f+1) 才放行

**P9 ●贡献**（~100 词）
- 四条贡献

**P10 章节导览**（~50 词）

---

## 五、风格指南（消除 AI 味，学两篇参考）

- 开场真实场景锚定（冻土站网/无人机群）｜ 老外篇 Verkada breach 写法
- 限制用 (i)(ii)(iii) ｜ 老外篇
- 目标句 "The primary objective of this research is to..." ｜ 老外篇
- Gap "relatively few studies... This gap presents a significant opportunity" ｜ 老外篇（替代 to the best of our knowledge）
- 图/表 "Figure X illustrates" / "Table Y summarizes" ｜ 老外篇
- 过渡 Nevertheless, / Despite these benefits, / In summary, ｜ 老外篇
- 自我限定 "The X scenario was deliberately excluded, as it represents an extreme case" ｜ 老外篇
- ● 四条贡献 ｜ 国人篇
- §3 Framework + §4 Algorithms 拆分 ｜ 国人篇
- 独立 §6 Discussion ｜ 国人篇

---

## 六、决策状态

- ✅ 五个原决策点将军已确认（不分节 / 预备知识嵌入 / 文献按需 / 冻土应用放§2.4+§5.7 / Abstract两段式）
- ✅ 第一章篇幅修订为 1100 词（14%），实验章升 25%
- ✅ 参考文献改为灵活区间 35-45，优先 2024+
- ⏳ 待将军确认本修订后，动笔重写第一章
