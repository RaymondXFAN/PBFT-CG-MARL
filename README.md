# PBFT-CG-MAPPO: Safe Consensus Conditioning for Byzantine-Resilient Multi-Agent Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A consensus-conditioned MARL framework that embeds PBFT (Practical Byzantine Fault Tolerance) as a per-step message filter inside the CTDE-MAPPO training loop, delivering a formal *f* < *n*/3 message-level consistency guarantee.

## Key Features

- **PBFT Consensus Layer**: Three-phase commit (pre-prepare, prepare, commit) embedded inside the MARL training loop as a per-step message filter
- **Additive Consensus Loss**: Differentiable alignment gradient toward the quorum-verified mean, providing 5.7× variance reduction (vs. 1.1× for same-form L2 penalty)
- **Entropy Floor Protection**: Prevents catastrophic policy collapse across seeds (zero catastrophic seeds with entropy_coef = 0.15)
- **Formal Byzantine Guarantee**: Provable *f* < *n*/3 message-level consistency under partial synchrony (Theorem 1)
- **Cross-Environment Validation**: Evaluated on MPE spread, SMAClite 5m_vs_6m, and VMAS UAV coverage
- **2×2 Factorial Ablation**: Decomposes filter vs. loss contributions with 3 seeds per condition

## Software & Hardware Requirements

### Software

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | Tested on 3.10 |
| PyTorch | 2.2+ | CUDA 12.1+ compatible |
| pettingzoo | 1.24+ | MPE environments |
| smac-lite | 0.1+ | SMAClite environments |
| vmas | 1.0+ | VMAS environments |
| numpy, scipy | latest | Data processing & statistics |
| matplotlib | 3.7+ | Figure generation |
| python-docx | 0.8+ | Paper generation |

```bash
pip install -r requirements.txt
```

### Hardware

| Component | Specification | Notes |
|-----------|--------------|-------|
| GPU | NVIDIA RTX 4090 (24GB) | Recommended; ~500MB VRAM per experiment |
| CPU | 16+ cores | 3-4 parallel experiments supported |
| RAM | 32GB+ | Minimal per experiment |
| Storage | 10GB+ | Code + results (data stored separately) |

Tested on AutoDL cloud GPU instances (RTX 4090, Ubuntu 22.04, CUDA 12.8).

## Data Acquisition

This project does **not** include raw experiment data in the repository due to size constraints. To reproduce results:

1. **Run experiments**: Follow the Deployment & Execution section below
2. **Pre-generated figures**: Included in `paper_figures/` for paper compilation
3. **Figure generation script**: `paper_figures/generate_figures.py` creates all 6 paper figures from the reported data

If you need access to raw experiment metrics, please open a GitHub issue.

## Software Architecture

```
PBFT-CG-MARL/
├── src/                          # Core source code
│   ├── algorithms/
│   │   ├── pbft_cg_mappo.py      # PBFT-CG-MAPPO algorithm (main contribution)
│   │   ├── mappo.py              # MAPPO baseline + L2 regularization control
│   │   ├── maddpg.py             # MADDPG baseline
│   │   ├── qmix.py               # QMIX baseline
│   │   ├── commnet.py            # CommNet baseline
│   │   └── tarmac.py             # TarMAC baseline
│   ├── consensus/
│   │   └── pbft.py               # PBFT consensus layer (3-phase commit)
│   ├── envs/
│   │   ├── mpe_wrapper.py        # MPE environment wrapper
│   │   ├── smac_wrapper.py       # SMAClite wrapper
│   │   └── vmas_wrapper.py       # VMAS wrapper
│   ├── utils/
│   │   ├── buffer.py             # On-policy replay buffer
│   │   ├── logger.py             # Metrics logging
│   │   └── metrics.py            # Metrics computation
│   └── train.py                  # Main training entry point
├── configs/
│   ├── algo/                     # Algorithm configurations (YAML)
│   │   ├── pbft_cg_mappo.yaml    # Default PBFT-CG-MAPPO config
│   │   ├── mappo.yaml            # MAPPO baseline config
│   │   ├── mappo_l2reg.yaml      # L2 regularization control config
│   │   └── ...                   # Other baselines
│   └── env/                      # Environment configurations (YAML)
│       ├── mpe_spread.yaml       # MPE spread (4 agents)
│       ├── smac_5m_vs_6m.yaml    # SMAClite scenario
│       └── vmas_uav_coverage.yaml
├── scripts/
│   ├── run_all.sh                # Full experiment suite
│   ├── run_v4_parallel.sh        # V4 parallel experiments
│   ├── run_l2reg_ablation.sh     # L2 regularization control
│   └── collect_v4_full.py        # Results extraction script
├── paper_figures/
│   ├── generate_figures.py       # Generate all 6 paper figures
│   ├── Figure1-TrainingReturnCurves.png
│   ├── Figure2-VarianceComparison.png
│   ├── Figure3-EntropyTrajectories.png
│   ├── Figure4-ByzantineComparison.png
│   ├── Figure5-AblationHeatmap.png
│   └── Figure6-SeedScatter.png
├── PBFT_CG_MAPPO_paper_v5.md     # Latest paper manuscript
└── requirements.txt
```

## Deployment & Execution

### 1. Clone and Setup

```bash
git clone https://github.com/RaymondXFAN/PBFT-CG-MARL.git
cd PBFT-CG-MARL
pip install -r requirements.txt
```

### 2. Quick Smoke Test (~5 min)

Verify the training pipeline works before committing to long experiments:

```bash
python -u src/train.py \
  --algo pbft_cg_mappo \
  --env mpe_spread \
  --seed 1 \
  --n_timesteps 5000 \
  --eval_interval 1000 \
  --eval_episodes 10 \
  --gpu 0 \
  --log_dir results/smoke_test
```

Expected output: Training logs with episode_reward, consensus_rate, entropy. Results saved to `results/smoke_test/`.

### 3. Full Experiments

#### Baseline Comparison (6 algorithms × 3–5 seeds × 500k steps)

```bash
# PBFT-CG-MAPPO clean (3 seeds)
nohup bash -c 'for seed in 1 2 3; do
  python -u src/train.py --algo pbft_cg_mappo --env mpe_spread \
    --algo_config configs/algo/pbft_cg_mappo.yaml \
    --seed $seed --n_timesteps 500000 \
    --eval_interval 10000 --eval_episodes 10 \
    --gpu 0 --log_dir results/baseline/pbft_s${seed}
done' > results/baseline/pbft.log 2>&1 &

# MAPPO clean (3 seeds) — repeat for other baselines
```

#### Byzantine Attack Experiments (3 attack types × 3 seeds)

```bash
python -u src/train.py --algo pbft_cg_mappo --env mpe_spread \
  --algo_config configs/algo/pbft_cg_mappo.yaml \
  --byzantine_n 1 --byzantine_type adversarial \
  --seed 1 --n_timesteps 500000 --gpu 0 \
  --log_dir results/byzantine/adv_s1
```

#### Ablation Study (2×2 factorial)

```bash
# Filter-off (Loss Only): consensus_threshold=0
# Loss-off (Filter Only): consensus_loss_coef=0
# Full: both enabled
# MAPPO: neither enabled (vanilla MAPPO)
```

#### L2 Regularization Control

```bash
nohup bash scripts/run_l2reg_ablation.sh 0 > results/l2reg.log 2>&1 &
```

### 4. Parallel Execution (Recommended)

3-4 experiments can run simultaneously on RTX 4090 (CPU-bound, not GPU-bound):

```bash
# Run 3 seeds in parallel
for seed in 1 2 3; do
  python -u src/train.py --algo pbft_cg_mappo --env mpe_spread \
    --seed $seed --n_timesteps 500000 --gpu 0 \
    --log_dir results/parallel/pbft_s${seed} &
done
wait
```

## Viewing Results

### Training Metrics

Results are saved as `metrics.json` in each experiment directory:

```
results/{experiment}/{algorithm}/seed_{N}/metrics.json
```

Format: WandB-style time series (keys = metric names, values = arrays).

Key metrics:
- `episode/episode_reward`: Episode return per evaluation
- `episode/consensus_rate`: PBFT consensus rate
- `train/entropy`: Policy entropy
- `train/value_loss`, `train/policy_loss`: Training losses

### Extract Summary Statistics

```bash
python scripts/collect_v4_full.py
```

Outputs per-seed and aggregate (mean ± std) tables for all experiments.

### Generate Paper Figures

```bash
cd paper_figures
python generate_figures.py
```

Generates all 6 figures (Morandi color scheme, 300 DPI, no captions) in the current directory.

## Citation

If you use this code, please cite:

```bibtex
@article{fan2025pbftcgmappo,
  title={Safe Consensus Conditioning for Byzantine-Resilient Multi-Agent Reinforcement Learning},
  author={Fan, Xiangyu and collaborators},
  journal={Under review},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
