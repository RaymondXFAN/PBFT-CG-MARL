# PBFT-CG-MARL

**Consensus-Guided Multi-Agent Reinforcement Learning under Byzantine Faults —
Failure-Mode Diagnosis and Design Guidelines**

This repository contains the reference implementation, configuration files, experiment
scripts, and analysis utilities for the paper:

> *Diagnosing Entropy Collapse in Consensus-Guided Multi-Agent Reinforcement Learning:
> Failure Modes and Design Guidelines* (submitted to **Applied Intelligence**).

The core question is deceptively simple: **when a Byzantine fault-tolerant consensus
protocol (PBFT) sits inside a cooperative MARL training loop, how should the agreed
action enter the learning update?** The natural answer hides a failure mode that task
reward never reveals.

---

## Table of contents

1. [Key findings](#1-key-findings)
2. [Repository layout](#2-repository-layout)
3. [Hardware & software requirements](#3-hardware--software-requirements)
4. [Installation](#4-installation)
5. [Environments](#5-environments)
6. [Algorithms](#6-algorithms)
7. [Configuration system](#7-configuration-system)
8. [Quick start](#8-quick-start)
9. [Reproducing the paper](#9-reproducing-the-paper)
10. [Outputs & how to inspect results](#10-outputs--how-to-inspect-results)
11. [Data provenance](#11-data-provenance)
12. [Troubleshooting](#12-troubleshooting)
13. [Citation](#13-citation)

---

## 1. Key findings

**The failure mode.** Replacing each agent's executed action with the PBFT-agreed action
`a*` and then computing the PPO gradient at `log π(a*|o)` (*matched* conditioning)
collapses policy entropy to near-determinism, while episode reward stays statistically
indistinguishable from the *mismatched* variant (`log π(a_i|o)`). The damage lives in the
training path, so reward-based evaluation never flags it.

**The mechanism.** An unbounded agreement–diversity loop:

```
PPO reinforces a*  →  agents learn to agree  →  consensus rate ↑  →  policy diversity ↓  →  repeat
```

The loop survives entropy-coefficient tuning, which rules out a hyperparameter explanation.

**Three design guidelines** (derived from the diagnosis and a fault-ratio sweep):

1. **Never pair action replacement with matched log-probability updates.**
2. **Prefer additive loss conditioning with zero-initialized influence.**
3. **Enforce entropy floors as defense-in-depth** (never engaged in our runs).

**Fault-ratio contingency.** The value of consensus is *conditional*: at `f/n = 1/5`
consensus buys nothing measurable, whereas at `f/n = 2/7` execution-time consensus gains
14 reward units over plain MAPPO (`p = 0.014`, `d = −1.51`).

---

## 2. Repository layout

```
PBFT-CG-MARL-release/
├── src/                        Core library
│   ├── train.py                Single-run entry point (argparse CLI)
│   ├── eval.py                 Evaluation utilities
│   ├── algorithms/             pbft_cg_mappo.py (ours), mappo, qmix, maddpg, commnet, tarmac
│   ├── consensus/              pbft.py — PBFT protocol (pre-prepare / prepare / commit)
│   ├── envs/                   ENV_REGISTRY + wrappers (MPE, SMAClite, VMAS, LBF, permafrost)
│   ├── networks/               Actor/critic backbones (GRU, MLP)
│   └── utils/                  Logging, seeding, config loading, statistics
├── configs/
│   ├── algo/                   Algorithm YAMLs (incl. n5_* review set and supp_* variants)
│   └── env/                    Environment YAMLs (mpe_spread_n5, mpe_spread_n7, vmas, lbf, …)
├── baselines/                  Independent baselines (e.g. krum_defense.py)
├── scripts/                    Batch launchers, collectors, statistical analysis
├── experiments/                Experiment design notes and phase scripts
├── figures/                    Paper figures (PDF + PNG)
├── models/                     Small reference checkpoints
├── legacy/                     Historical patch scripts (development artefacts, not needed to run)
├── paper/                      Latest manuscript (Markdown + DOCX) and its figures
│   ├── paper_v7_AI.md/.docx    Manuscript source and compiled document
│   ├── figures/                300-dpi PNG + vector PDF of every figure
│   └── figures_named/          Print-ready copies (Figure_1_Captain_Y.png …)
├── README.md                   This file
├── DEPLOY.md                   Step-by-step deployment guide (server setup)
├── requirements.txt            Python dependencies
├── CITATION.cff                Citation metadata (GitHub "Cite this repository")
├── LICENSE                     MIT licence
├── .gitignore                  Python / output / checkpoint ignores
├── EXPERIMENT_RESULTS.md       Frozen numbers from earlier development rounds
└── FIX_PLAN.md                 Development log / issue tracker (historical)
```

> **Note.** `legacy/` holds one-off patch scripts from the development history. They are
> kept for provenance only and are **not** required to reproduce any result.

---

## 3. Hardware & software requirements

### Hardware

| Item | Tested configuration | Notes |
|---|---|---|
| GPU | **NVIDIA RTX 4090 (24 GB)** | All paper results were produced on this card |
| GPU (alt) | Any CUDA GPU ≥ 8 GB | Works; reduce concurrency (see below) |
| CPU-only | Possible for tiny smoke tests | Impractically slow for full runs |

**Measured cost** (RTX 4090, MPE `simple_spread`, `n = 5`, 300K steps):
≈ **95 minutes per seed** when running 4 seeds concurrently (GPU utilisation ≈ 90–99 %).

**Concurrency guidance**

| Environment | Recommended parallel runs |
|---|---|
| MPE (`mpe_spread`, `simple_line`) | 4 (4090) / 2 (10–12 GB cards) |
| VMAS (`uav_coverage`) | 2 |
| LBF | 2 |

> ⚠️ Never launch two batch scripts at once — task-level throttling is implemented per
> script, so two instances will double the real concurrency and can trigger OOM.

### Software

| Component | Version used |
|---|---|
| OS | Linux (Ubuntu 20.04+, AutoDL container image) |
| Python | 3.10 |
| CUDA driver | 570.124.04 (shows CUDA 12.8 via `nvidia-smi`) |
| PyTorch | ≥ 2.0, built for **cu124** (12.4 ≤ 12.8, compatible) |
| Key packages | `pettingzoo[mpe]`, `vmas`, `gymnasium`, `pyyaml`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `wandb` (optional) |

Full list: [`requirements.txt`](requirements.txt).

---

## 4. Installation

```bash
# 1) Create the environment (conda recommended)
conda create -n pbft python=3.10 -y && conda activate pbft

# 2) Install PyTorch matching your CUDA driver (example: cu124)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3) Install the rest
pip install -r requirements.txt

# 4) SMAClite is not on PyPI; install from source only if you need it
# pip install git+https://github.com/ArnaudGardelle/smaclite.git

# 5) Sanity check
python -c "import torch, pettingzoo, vmas; print('OK', torch.cuda.is_available())"
```

---

## 5. Environments

Registered in `src/envs/__init__.py` (`ENV_REGISTRY`):

| Registry key | Backend | Action space | Agents |
|---|---|---|---|
| `mpe_spread` / `mpe_reference` | PettingZoo MPE | Discrete | configurable |
| `smaclite_5m_vs_6m`, `smaclite_3s5z` | SMAClite | Discrete | 5–8 |
| `vmas_uav_coverage`, `vmas_formation` | VMAS | Continuous | 4 |
| `lbf_2s3f` | Level-Based Foraging | Discrete | 2 |
| `permafrost_monitoring` | Custom (real sensor data) | Discrete | 12 |

> ⚠️ **Always pass `--env` explicitly.** The default (`simple_spread`) is *not* in
> `ENV_REGISTRY`; silently falling back makes `env_config` a no-op.

Two task variants are used in the paper:

- **MPE `simple_spread`** — cooperative coverage; primary task (30-dim observations).
- **MPE `simple_line`** — agents arrange along a line; second task, sharper contrast
  (8-dim observations).

---

## 6. Algorithms

Registered in `src/algorithms/__init__.py` (`ALGORITHM_REGISTRY`):

| `--algo` | Family | Communication |
|---|---|---|
| `pbft_cg_mappo` | Actor–Critic (PPO) | **PBFT consensus (ours)** |
| `mappo` | Actor–Critic (PPO) | none (baseline) |
| `qmix` | Value-based | none |
| `maddpg` | Actor–Critic (DDPG) | none |
| `commnet` | Actor–Critic (PPO) | implicit |
| `tarmac` | Actor–Critic (PPO) | attention |

### The two conditioning paths (the paper's central variable)

| Path | How it works | Config flag |
|---|---|---|
| **Multiplicative** (hard) | The executed action is *replaced* by `a*` | `soft_consensus_mode: false`, `conditioning_type: multiplicative` |
| **Additive** (soft) | Action untouched; a bounded KL penalty is added to the loss | `soft_consensus_mode: true`, `kl_consensus_coef: 0.1` |

…and orthogonally, the gradient target:

| Target | `matched_logprob` | Behaviour |
|---|---|---|
| **Mismatched** | `false` | Gradient at `log π(a_i)` — consistent with the executed action |
| **Matched** | `true` | Gradient at `log π(a*)` — **triggers entropy collapse when combined with replacement** |

> The 2×2 design of `{replacement, additive} × {matched, mismatched}` is exactly what
> Table 4 of the paper reports.

---

## 7. Configuration system

Two YAMLs per run: one for the algorithm, one for the environment.

```bash
--algo_config configs/algo/<file>.yaml
--env_config  configs/env/<file>.yaml
```

**Naming conventions**

| Prefix | Meaning |
|---|---|
| `n5_*` | Main-table review set at `n = 5` (E1 clean, E2 causal, E3 Byzantine) |
| `supp_*` | Supplemental batches (MAPPO+adversarial, seeds 5–8, `f = 2` at `n = 7`) |
| `pbft_alpha_*`, `pbft_kl_*` | Ablations over the additive-conditioning coefficient |

**Byzantine faults** are injected via the YAML block (and, for safety, duplicated on the
command line — both paths target the *same* agents and do not stack):

```yaml
byzantine:
  inject_at: proposal      # which phase the fault enters
  mode: adversarial        # or random
  n: 1                     # number of faulty agents
pbft_f: 1                  # top-level key (see note below)
```

> ⚠️ **`pbft_f` must be written at the top level**, not nested under `pbft:`. The config
> loader flattens with `config[k] = v`, so a nested `pbft: {f: 2}` becomes `config["f"]`
> and is silently ignored, leaving `f` at its default.

---

## 8. Quick start

A single training run:

```bash
python src/train.py \
    --algo pbft_cg_mappo \
    --env mpe_spread \
    --env_config configs/env/mpe_spread_n5.yaml \
    --algo_config configs/algo/n5_soft_fair.yaml \
    --seed 1 \
    --n_timesteps 300000 \
    --eval_interval 10000 \
    --eval_episodes 10 \
    --gpu 0 \
    --log_dir results/demo
```

Add a Byzantine agent (one adversarial faulty agent):

```bash
python src/train.py --algo mappo --env mpe_spread \
    --env_config configs/env/mpe_spread_n5.yaml \
    --algo_config configs/algo/n5_mappo_fair.yaml \
    --seed 1 --n_timesteps 300000 --gpu 0 \
    --byzantine_n 1 --byzantine_type adversarial \
    --log_dir results/demo
```

A **5-minute pre-flight check** (validates data loading, forward/backward pass, and that
`metrics.json` is actually written) is strongly recommended before any long batch:

```bash
bash scripts/verify_5min.sh 0
# expected: "预检结果: ✅ N ❌ 0"
```

---

## 9. Reproducing the paper

All batch launchers are resumable: completed seeds are skipped automatically (detected by
the presence of `metrics.json`).

### Main tables (n = 5, clean + causal + Byzantine)

```bash
bash scripts/run_n5_review.sh          # E1 (clean) + E2 (matched vs mismatched) + E3 (f = 1)
python scripts/collect_n5_results.py   # assemble the tables
```

### Supplemental batches

```bash
bash scripts/run_supplement.sh verify  # 5-minute pre-flight
bash scripts/run_supplement.sh mappo   # Batch A: MAPPO + adversarial @ n=5
bash scripts/run_supplement.sh seeds   # Batch B: seeds 5–8 @ n=5
bash scripts/run_supplement.sh f2      # Batch D: n = 7, f = 2
bash scripts/run_supplement.sh vmas    # Batch C: VMAS second environment (concurrency 2)
bash scripts/run_supplement.sh results # summarise
```

Or run the whole queue unattended:

```bash
nohup bash scripts/run_all_serial.sh > logs/supplement/serial_master.log 2>&1 &
```

### Table 4, fourth cell — Hard **Matched** at f = 2 (n = 7)

This is the run that closes the 2×2 causal design.

```bash
bash scripts/run_hard_matched.sh verify   # pre-flight (~5 min)
nohup bash scripts/run_hard_matched.sh run > logs/supplement/hard_matched_master.log 2>&1 &
bash scripts/run_hard_matched.sh status   # progress: N / 8
bash scripts/run_hard_matched.sh collect  # mean ± std over seeds
```

Expected wall-clock: ≈ **1.5 h** (8 seeds, 4-way concurrency, RTX 4090).

### Statistical analysis

```bash
python scripts/statistical_analysis.py     # Welch t, Cohen's d, Hedges' g, MWU, 95% CI
python scripts/stats_supp.py               # summary across supplemental batches
```

---

## 10. Outputs & how to inspect results

Every run writes a single JSON per seed:

```
<log_dir>/<algo>/seed_<N>/metrics.json
```

> Note the **double nesting**: `train.py` appends `<algo>/seed_<N>/` inside `--log_dir`.

**Key fields** (`key = array over the training run`):

| Key | Meaning |
|---|---|
| `episode/episode_reward` | Episode return at each evaluation point |
| `episode/episode_length` | Episode length |
| `episode/consensus_rate` | PBFT consensus success rate |
| `train/entropy` | **All-agent** mean policy entropy (mean of last 10 eval points) |
| `train/entropy_agent_i` | Per-agent entropy (only logged when instrumentation is enabled) |
| `train/policy_loss`, `train/value_loss` | PPO losses |
| `train/kl_consensus_loss` | Additive (soft) conditioning term |
| `train/entropy_floor_penalty` | Floor penalty (never engaged in reported runs) |
| `train/consensus_loss`, `train/consensus_rate` | Consensus-side training signals |
| `train/lr`, `train/entropy_coef` | Optimiser / schedule diagnostics |

**Convention used throughout the paper**: a metric for a run is the **mean over the last
10 evaluation points**, then reported as mean ± std across seeds.

Quick inspection:

```bash
find results -name metrics.json | wc -l                 # how many runs finished
python - <<'PY'
import json, glob, statistics as st
for f in sorted(glob.glob("results/**/metrics.json", recursive=True))[:3]:
    d = json.load(open(f))
    r = d["episode/episode_reward"]; e = d["train/entropy"]
    print(f, "| reward(last10)=%.2f | entropy(last10)=%.4f"
          % (sum(r[-10:])/10, sum(e[-10:])/10))
PY
```

Monitoring a live batch:

```bash
bash scripts/run_hard_matched.sh status              # finished count + GPU
tail -n 20 logs/supplement/hard_matched_master.log   # master log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

---

## 11. Data provenance

- **MPE / SMAClite / VMAS / LBF** — pure simulation, downloaded automatically by the
  respective libraries. **No external data download is required.**
- **`permafrost_monitoring`** — an optional adapter for real permafrost sensor data.
  If you use it, please cite the data sources below and export any credentials through
  **environment variables** (never commit passwords):
  - *National Tibetan Plateau / Third Pole Environment Data Center (TPDC)* —
    <http://data.tpdc.ac.cn> (DOI: `10.11888/Geocry.tpdc.271107`)
  - *GTN-P (Global Terrestrial Network for Permafrost)* — <https://data.gtn-p.org/>

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `env_config` seems ignored | You forgot `--env mpe_spread`; the default `simple_spread` is not registered |
| `f` stays at 1 despite `pbft: {f: 2}` | Put `pbft_f: 2` at the **top level** of the YAML |
| `unrecognized arguments: --byzantine_faults` | Not a valid flag; use `--byzantine_n` / `--byzantine_type` / `--byzantine_mode` |
| Training says "done" but no results | Look in `<log_dir>/<algo>/seed_<N>/metrics.json` (double nesting) |
| VMAS shape error | Must be `--env vmas_uav_coverage` (there is no `vmas_navigation`) |
| OOM with several runs | Lower concurrency (4 → 2); never run two batch scripts simultaneously |
| NaN loss | Re-run from a clean seed; a guard is included in `src/utils` |
| Re-running a batch re-trains everything | Should not happen — completed seeds are skipped via `metrics.json` check |

---

## 13. Citation

```bibtex
@article{pbftcgmarl2026,
  title   = {Diagnosing Entropy Collapse in Consensus-Guided Multi-Agent
             Reinforcement Learning: Failure Modes and Design Guidelines},
  author  = {Anonymous},
  journal = {Applied Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

---

## 14. License

Released under the **MIT License** — see [`LICENSE`](LICENSE).
Citation metadata is provided in [`CITATION.cff`](CITATION.cff), so GitHub can render a
"Cite this repository" button automatically.

---

*Maintained as the reference implementation accompanying the manuscript. See
[`DEPLOY.md`](DEPLOY.md) for a step-by-step server deployment walkthrough.*
nforcement Learning: Failure Modes and Design Guidelines},
  author  = {Anonymous},
  journal = {Applied Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

---

*Maintained as the reference implementation accompanying the manuscript. See
[`DEPLOY.md`](DEPLOY.md) for a step-by-step server deployment walkthrough.*
