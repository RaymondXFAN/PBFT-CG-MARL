# PBFT-Consensus-Guided MARL

## 项目简介

**PBFT-Consensus-Guided MARL**（基于PBFT拜占庭容错共识引导的多智能体强化学习）是一个将PBFT（Practical Byzantine Fault Tolerance）共识协议嵌入多智能体强化学习（MARL）框架的研究项目。

### 核心创新点

- **PBFT共识嵌入**：将PBFT拜占庭容错共识机制集成到MARL的决策过程中，提高多智能体系统的鲁棒性
- **共识引导策略**：通过共识机制引导智能体策略更新，减少拜占庭Agent对系统的影响
- **Leader轮换机制**：动态Leader选举和轮换，避免单点故障
- **Fallback机制**：当共识失败时自动降级，保证系统可用性

### 支持环境

| 环境 | 场景 | 动作类型 | 智能体数 |
|------|------|----------|----------|
| MPE (PettingZoo) | simple_spread, simple_reference | 离散 | 3 |
| SMAClite | 5m_vs_6m, 3s5z | 离散 | 5/8 |
| VMAS | uav_coverage, formation | 连续 | 4 |
| LBF | 2s-3f-2co | 离散 | 2 |

### 支持算法

| 算法 | 类型 | 通信 |
|------|------|------|
| PBFT-CG-MAPPO | Actor-Critic (PPO) | PBFT共识 |
| MAPPO | Actor-Critic (PPO) | 无 |
| QMIX | Value-Based | 无 |
| MADDPG | Actor-Critic (DDPG) | 无 |
| CommNet | Actor-Critic (PPO) | 隐式通信 |
| TarMAC | Actor-Critic (PPO) | 注意力通信 |

## 安装说明

### 环境要求

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA >= 11.7（推荐GPU训练）

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/PBFT-CG-MARL.git
cd PBFT-CG-MARL

# 2. 创建虚拟环境
conda create -n pbft_marl python=3.10 -y
conda activate pbft_marl

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "from src.envs import ENV_REGISTRY; print('环境注册:', list(ENV_REGISTRY.keys()))"
```

### 可选依赖

```bash
# SMAClite（需要StarCraft II）
pip install smaclite

# LBF
pip install lbforaging

# VMAS
pip install vmas

# WandB日志
wandb login
```

> **注意**：如果某个环境包不可用，系统会自动使用内置的简化版回退实现（基于Numpy），无需额外安装。

## 快速开始

### 单次实验

```bash
# 运行PBFT-CG-MAPPO在MPE-spread上
python -u src/train.py \
    --algo pbft_cg_mappo \
    --env mpe_spread \
    --env_config configs/env/mpe_spread.yaml \
    --algo_config configs/algo/pbft_cg_mappo.yaml \
    --seed 1 \
    --n_timesteps 3000000 \
    --eval_interval 10000 \
    --eval_episodes 10 \
    --log_dir results/test \
    --gpu 0
```

### 批量实验

```bash
# 运行所有实验（默认GPU 0）
bash run_all.sh

# 指定GPU
bash run_all.sh 1

# 从第13个实验开始
bash run_all.sh 0 13
```

### 拜占庭容错实验

```bash
# 1个拜占庭Agent
python -u src/train.py \
    --algo pbft_cg_mappo \
    --env mpe_spread \
    --env_config configs/env/mpe_spread.yaml \
    --algo_config configs/algo/pbft_cg_mappo.yaml \
    --seed 1 \
    --n_timesteps 3000000 \
    --byzantine_n 1 \
    --gpu 0
```

### 消融实验

```bash
# 无共识机制
python -u src/train.py \
    --algo pbft_cg_mappo \
    --env mpe_spread \
    --env_config configs/env/mpe_spread.yaml \
    --algo_config configs/algo/pbft_cg_mappo.yaml \
    --seed 1 \
    --n_timesteps 3000000 \
    --ablation no_consensus \
    --gpu 0
```

## 实验说明

### 实验矩阵

| 编号 | 环境 | 算法数 | Seeds | 总时间步 |
|------|------|--------|-------|----------|
| E1 | MPE-spread | 6 | 10 | 3M |
| E2 | MPE-reference | 6 | 10 | 3M |
| E3 | SMAClite 5m_vs_6m | 6 | 5 | 10M |
| E4 | SMAClite 3s5z | 6 | 5 | 10M |
| E5 | VMAS UAV覆盖 | 6 | 5 | 5M |
| E6 | VMAS 编队控制 | 6 | 5 | 5M |
| E7 | LBF 合作采集 | 6 | 10 | 1M |
| E8 | 拜占庭容错 (MPE) | 2 | 5 | 3M |
| E9 | 拜占庭容错 (SMAClite) | 2 | 5 | 10M |
| E10 | 消融-PBFT组件 | 1 | 5 | 3M |
| E11 | 消融-通信频率 | 1 | 5 | 3M |
| E12 | 消融-LBF环境 | 1 | 5 | 1M |

### 结果可视化

```bash
# 绘制训练曲线
python src/scripts/visualize.py --log_dir results/ --output_dir figures/

# 绘制算法对比图
python src/scripts/visualize.py --log_dir results/ --plot_type comparison

# 绘制拜占庭容错曲线
python src/scripts/visualize.py --log_dir results/ --plot_type byzantine

# 绘制消融实验对比
python src/scripts/visualize.py --log_dir results/ --plot_type ablation
```

### GTN-P地温数据验证

```bash
# 运行地温数据验证（使用合成数据）
python src/scripts/gtnp_data.py --use_synthetic --n_stations 5

# 运行地温数据验证（使用真实数据）
python src/scripts/gtnp_data.py --data_dir /path/to/gtnp/data
```

## 项目结构

```
PBFT-CG-MARL/
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖列表
├── run_all.sh                         # 完整实验脚本
├── configs/
│   ├── algo/                          # 算法配置
│   │   ├── pbft_cg_mappo.yaml         # PBFT-CG-MAPPO
│   │   ├── mappo.yaml                 # MAPPO
│   │   ├── qmix.yaml                  # QMIX
│   │   ├── maddpg.yaml                # MADDPG
│   │   ├── commnet.yaml               # CommNet
│   │   └── tarmac.yaml                # TarMAC
│   └── env/                           # 环境配置
│       ├── mpe_spread.yaml
│       ├── mpe_reference.yaml
│       ├── smaclite_5m_vs_6m.yaml
│       ├── smaclite_3s5z.yaml
│       ├── vmas_uav_coverage.yaml
│       ├── vmas_formation.yaml
│       └── lbf_2s3f.yaml
├── src/
│   ├── envs/                          # 环境适配器
│   │   ├── __init__.py                # 环境注册
│   │   ├── base.py                    # 基础环境类
│   │   ├── mpe_wrapper.py             # MPE适配器
│   │   ├── smaclite_wrapper.py        # SMAClite适配器
│   │   ├── vmas_wrapper.py            # VMAS适配器
│   │   └── lbf_wrapper.py             # LBF适配器
│   └── scripts/                       # 工具脚本
│       ├── visualize.py               # 结果可视化
│       └── gtnp_data.py               # GTN-P数据验证
└── results/                           # 实验结果
```

## 引用

如果您使用了本项目，请引用：

```bibtex
@article{pbft_cg_marl2024,
  title={PBFT-Consensus-Guided Multi-Agent Reinforcement Learning},
  author={Your Name},
  journal={Conference/Journal},
  year={2024}
}
```

## 许可证

MIT License
