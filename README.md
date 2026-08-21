# PBFT-Consensus-Guided Multi-Agent Reinforcement Learning

## Project Overview

PBFT-Consensus-Guided MARL (PBFT-CG-MARL) embeds a provable Byzantine fault-tolerant consensus protocol inside MAPPO training, guaranteeing f < n/3 fault tolerance while maintaining end-to-end differentiability.

## Key Innovation: Soft Consensus

We identify and resolve a critical **entropy collapse** problem in hard consensus mechanisms:

| Method | Reward | Entropy | Consensus Rate | Status |
|--------|--------|---------|---------------|--------|
| **Soft Consensus (Ours)** | **-43.2±5.5** | **0.9469** | 0.18 | ✅ Best |
| Hard Consensus (Matched) | -72.6±10.2 | 0.1158 | 0.67 | ❌ Collapsed |
| Hard Consensus (Mismatched) | -44.2±4.4 | 1.0129 | 0.08 | ⚠️ Unstable |
| Vanilla MAPPO | -49.5±7.1 | 0.1337 | 0.00 | Baseline |

**Root Cause**: Hard consensus with matched log-prob creates a self-reinforcing feedback loop → agents learn to agree → policy space collapses. Soft consensus (KL-divergence guidance) preserves both gradient consistency and policy diversity.

## Supported Environments

| Environment | Scenario | Action Type | Agents |
|-------------|----------|-------------|--------|
| MPE (PettingZoo) | simple_spread, simple_reference | Discrete | 3-4 |
| SMAClite | 5m_vs_6m, 3s5z | Discrete | 5/8 |
| VMAS | uav_coverage, formation | Continuous | 4 |
| LBF | 2s-3f-2co | Discrete | 2 |

## Quick Start

### Installation

```bash
conda create -n pbft_marl python=3.10 -y
conda activate pbft_marl
pip install -r requirements.txt
```

### Run Experiments

```bash
# Soft consensus (recommended)
python -u src/train.py \
    --algo pbft_cg_mappo \
    --env mpe_spread \
    --env_config configs/env/mpe_spread.yaml \
    --algo_config experiments/configs/fair_soft.yaml \
    --seed 1 --n_timesteps 300000 --eval_interval 10000 \
    --eval_episodes 10 --log_dir results/soft --gpu 0

# Vanilla MAPPO baseline
python -u src/train.py \
    --algo mappo \
    --env mpe_spread \
    --env_config configs/env/mpe_spread.yaml \
    --algo_config experiments/configs/fair_mappo.yaml \
    --seed 1 --n_timesteps 300000 --eval_interval 10000 \
    --eval_episodes 10 --log_dir results/mappo --gpu 0
```

## Experiment Results

### E1: Consensus Mechanism Comparison (Clean Environment)

| Method | Reward | Entropy (final) | Consensus Rate |
|--------|--------|-----------------|----------------|
| Soft Consensus | -43.2±5.5 | 0.9469 | 0.18 |
| Hard Consensus (Mismatched) | -44.2±4.4 | 1.0129 | 0.08 |
| Hard Consensus (Matched) | -72.6±10.2 | 0.1158 | 0.67 |
| MAPPO Baseline | -49.5±7.1 | 0.1337 | 0.00 |

### E2: Causal Analysis (Matched vs Mismatched Log-prob)

| Method | Reward | Entropy | Consensus Rate |
|--------|--------|---------|----------------|
| Hard + Matched | -70.8±3.6 | 0.1153 | 0.67 |
| Hard + Mismatched | -44.2±4.4 | 1.0129 | 0.08 |

**Conclusion**: Entropy collapse is caused by hard consensus constraint itself (not log-prob mismatch).

### E3: Byzantine Fault Tolerance

| Method | Reward (Adversarial) | Reward (Random) | Entropy |
|--------|---------------------|-----------------|---------|
| Hard Consensus | -43.5±3.5 | -46.6±3.8 | 0.99-1.15 |
| Soft Consensus | -39.3±6.6 | -46.1±5.9 | 0.77-1.16 |

## Citation

```bibtex
@article{pbft_cg_marl2026,
  title={PBFT-Consensus-Guided Multi-Agent Reinforcement Learning with Soft Consensus},
  author={...},
  journal={...},
  year={2026}
}
```

## License

MIT License
