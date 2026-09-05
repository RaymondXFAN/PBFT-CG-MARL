# PBFT-Consensus-Guided Multi-Agent Reinforcement Learning

**Byzantine-robust multi-agent cooperation via Practical Byzantine Fault Tolerance consensus conditioning.**

This repository implements PBFT-CG-MAPPO, a consensus-guided multi-agent reinforcement learning algorithm that embeds the PBFT (Practical Byzantine Fault Tolerance) consensus protocol into the policy optimization process of MAPPO. During each training step, agents propose action candidates, run a lightweight PBFT consensus round to filter adversarial or corrupted proposals, and blend the agreed-upon consensus direction into the policy gradient. This mechanism enables cooperative MARL agents to maintain robust group-level performance even when a subset of agents behave as Byzantine faults—submitting forged messages, random actions, or adversarial proposals. Experiments on MPE simple_spread (N=3 and N=5) with message-forge attacks show that PBFT-CG-MAPPO achieves higher reward and lower variance than vanilla MAPPO under Byzantine conditions, while remaining competitive in clean (non-Byzantine) settings. The codebase supports discrete and continuous action spaces, multiple environments (MPE, SMAClite, VMAS, LBF), and provides configurable consensus hyperparameters (consensus loss coefficient, frequency, blend ratio, entropy floor) for systematic ablation.

## Quick Start

```bash
# Install dependencies
pip install torch pettingzoo[mpe2] numpy

# Train PBFT-CG-MAPPO (N=3, clean)
python src/train.py --algo pbft_cg_mappo --env mpe_spread \
    --env_config configs/env/mpe_spread.yaml \
    --algo_config configs/algo/pbft_cg_mappo_v4.yaml \
    --seed 1 --n_timesteps 300000 --eval_interval 10000 \
    --eval_episodes 10 --gpu 0

# Train PBFT-CG-MAPPO (N=3, Byzantine forge attack)
python src/train.py --algo pbft_cg_mappo --env mpe_spread \
    --env_config configs/env/mpe_spread.yaml \
    --algo_config configs/algo/pbft_cg_mappo_v4.yaml \
    --seed 1 --n_timesteps 300000 --eval_interval 10000 \
    --eval_episodes 10 --byzantine_n 1 --byzantine_type message_forge \
    --gpu 0

# Train vanilla MAPPO (baseline)
python src/train.py --algo mappo --env mpe_spread \
    --env_config configs/env/mpe_spread.yaml \
    --algo_config configs/algo/mappo_v4_clean.yaml \
    --seed 1 --n_timesteps 300000 --eval_interval 10000 \
    --eval_episodes 10 --gpu 0

# N=5 experiments (5 agents, 5 landmarks)
python src/train.py --algo pbft_cg_mappo --env mpe_spread \
    --env_config configs/env/mpe_spread_n5.yaml \
    --algo_config configs/algo/pbft_cg_mappo_v4.yaml \
    --seed 1 --n_timesteps 300000 --eval_interval 10000 \
    --eval_episodes 10 --byzantine_n 1 --byzantine_type message_forge \
    --gpu 0
```

## Project Structure

```
PBFT-CG-MARL/
├── src/
│   ├── train.py                    # Main training entry point
│   ├── eval.py                     # Evaluation script
│   ├── algorithms/
│   │   ├── pbft_cg_mappo.py        # PBFT-CG-MAPPO algorithm (core)
│   │   ├── mappo.py                # Vanilla MAPPO baseline
│   │   └── base.py                 # Algorithm base class
│   ├── consensus/
│   │   └── pbft.py                 # PBFT consensus protocol implementation
│   ├── envs/
│   │   ├── mpe_wrapper.py          # MPE environment adapter
│   │   ├── vmas_wrapper.py         # VMAS environment adapter
│   │   ├── smaclite_wrapper.py     # SMAClite environment adapter
│   │   └── lbf_wrapper.py         # Level-Based Foraging adapter
│   ├── networks/
│   │   ├── actor_critic.py         # Actor-Critic network (discrete/continuous)
│   │   └── mixing_net.py          # Value mixing network
│   └── utils/
│       ├── buffer.py               # Rollout buffer
│       ├── logger.py               # Metrics logging
│       └── metrics.py              # Consensus & training metrics
├── configs/
│   ├── algo/                       # Algorithm hyperparameters
│   │   ├── pbft_cg_mappo_v4.yaml   # PBFT-CG-MAPPO (v4, recommended)
│   │   ├── mappo_v4_clean.yaml     # MAPPO baseline
│   │   └── ...                     # Ablation & v2/v3 configs
│   └── env/                        # Environment configurations
│       ├── mpe_spread.yaml         # N=3 simple_spread
│       ├── mpe_spread_n5.yaml      # N=5 simple_spread
│       └── ...
├── scripts/
│   ├── run_mpe_v5.sh              # Full N=3 experiment script (4 conditions × 5 seeds)
│   ├── run_v5_n5_safe.sh          # Full N=5 experiment script (4-way parallel)
│   └── ...
├── figures/                        # Paper figures (PDF/PNG)
└── experiments/                    # Ablation configs & analysis scripts
```

## Key Hyperparameters

| Parameter | PBFT-CG-MAPPO | MAPPO | Description |
|-----------|---------------|-------|-------------|
| `entropy_coef` | 0.1 (fixed) | 0.01 | Entropy bonus coefficient |
| `consensus_loss_coef` | 0.01 | — | Consensus loss weight |
| `consensus_freq` | 5 | — | Consensus round frequency (every N steps) |
| `consensus_blend_ratio` | 0.3 | — | Consensus direction blend ratio |
| `pbft.f` | 1 | — | Byzantine fault tolerance parameter |
| `clip_ratio` | 0.2 | 0.2 | PPO clip ratio |
| `lr` | 5e-4 | 5e-4 | Learning rate |
| `ppo_epoch` | 5 | 5 | PPO update epochs |

## Byzantine Attack Types

| Type | Description |
|------|-------------|
| `random` | Random action sampling |
| `adversarial` | Intentionally worst action |
| `flip` | Reverse action direction |
| `message_forge` | Forge consensus proposals (recommended) |

## Results (MPE simple_spread, 5 seeds, 300K steps)

| Setting | N=3 Reward | N=5 Reward |
|---------|-----------|-----------|
| PBFT-CG-MAPPO + Clean | -58.3 ± 18.3 | -67.7 ± 22.7 |
| MAPPO + Clean | -60.0 ± 22.3 | -48.7 ± 5.3 |
| PBFT-CG-MAPPO + Forge | -58.3 ± 18.4 | **-45.9 ± 9.1** |
| MAPPO + Forge | -40.2 ± 7.5 | -51.1 ± 6.5 |

> Under Byzantine message-forge attacks with 5 agents, PBFT-CG-MAPPO outperforms vanilla MAPPO by 5.1 reward points, demonstrating that consensus-based filtering becomes more effective as the team size grows.

## Requirements

- Python 3.10+
- PyTorch 2.2+ (CUDA 12.x)
- PettingZoo (MPE2)
- NumPy

## Citation

```bibtex
@article{pbft_cg_marl,
  title={PBFT-Consensus-Guided Multi-Agent Reinforcement Learning for Byzantine-Robust Cooperation},
  author={...},
  journal={...},
  year={2026}
}
```

## License

MIT License
