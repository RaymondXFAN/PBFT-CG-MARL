# Deployment & Reproduction Guide — PBFT-CG-MARL

Everything needed to stand this repository up on a fresh GPU server, reproduce the paper's
numbers, and re-draw its figures. Written for a Linux container (an AutoDL instance is assumed;
adapt paths as needed).

**Contents**

1. [Quick start (TL;DR)](#1-quick-start-tldr)
2. [Hardware & software](#2-hardware--software)
3. [Environment setup](#3-environment-setup)
4. [Repository architecture](#4-repository-architecture)
5. [Data](#5-data)
6. [Verify the installation](#6-verify-the-installation)
7. [Running the experiments](#7-running-the-experiments)
8. [Reading the results](#8-reading-the-results)
9. [Reproducing the figures](#9-reproducing-the-figures)
10. [Monitoring & operations](#10-monitoring--operations)
11. [Troubleshooting & FAQ](#11-troubleshooting--faq)

---

## 1. Quick start (TL;DR)

```bash
# 1) unpack
mkdir -p /root/autodl-tmp && tar -xzf PBFT-CG-MARL-release.tar.gz -C /root/autodl-tmp/
cd /root/autodl-tmp/PBFT-CG-MARL-release

# 2) dependencies
pip install -r requirements.txt

# 3) ~5 min sanity check — do not skip
bash scripts/verify_5min.sh 0

# 4) one real experiment (results land under results/)
nohup bash scripts/run_n5_review.sh > logs/n5_review.log 2>&1 &

# 5) watch it
bash scripts/status_all.py 2>/dev/null || find results -name metrics.json | wc -l
```

---

## 2. Hardware & software

| Item | Value used for every reported result |
|---|---|
| GPU | **NVIDIA RTX 4090 (24 GB)** |
| Acceptable alternative | Any CUDA GPU ≥ 8 GB — reduce concurrency proportionally |
| CUDA driver | 570.x (reports CUDA 12.8) |
| PyTorch | built against **cu124** (12.4 ≤ 12.8 → compatible) |
| Python | 3.10 |
| RAM / disk | 32 GB+ / ≥ 40 GB free on the data volume |
| Image | Any recent PyTorch image — **do not reinstall CUDA or torch**, the bundled versions already match |

**Practical throughput** (4090, 4 concurrent runs):

| Workload | Wall-clock |
|---|---|
| One method, 300 K steps, 8 seeds, MPE ($n=5$) | ≈ 1.5 h |
| Main $n=5$ review set (4 methods × 8 seeds, 300 K) | ≈ 3 h |
| Same at $n=7$, $f=2$ | ≈ 1.5 h per method |
| VMAS batch (continuous control, concurrency 2) | ≈ 7–8 h |
| Any `verify_5min.sh` pre-flight | ≈ 5 min |

> 💡 **AutoDL tip.** Build the environment in *CPU (datacard-free) mode* — packages install
> cheaply — then switch to *GPU mode* to train. The data volume survives the switch.

---

## 3. Environment setup

```bash
conda create -n pbft python=3.10 -y
conda activate pbft

# PyTorch matching the driver (cu124 works with a CUDA-12.8 driver)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# everything else
pip install -r requirements.txt
# equivalently:
# pip install numpy pyyaml tqdm matplotlib seaborn pandas \
#             "pettingzoo[mpe]>=1.24.0" gymnasium vmas
```

Check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect: 2.x.x True
nvidia-smi
```

**Behind a slow link?** Use a domestic mirror:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 4. Repository architecture

```
PBFT-CG-MARL-release/
├── src/
│   ├── train.py                 ★ single-run entry point (argparse CLI)
│   ├── eval.py                  standalone evaluation helpers
│   ├── algorithms/
│   │   ├── __init__.py          ALGORITHM_REGISTRY  (name → class)
│   │   ├── pbft_cg_mappo.py     ★ OUR METHOD: consensus-guided MAPPO
│   │   ├── mappo.py             MAPPO baseline
│   │   ├── qmix.py maddpg.py commnet.py tarmac.py    other baselines
│   │   └── base.py              shared trainer interface
│   ├── consensus/
│   │   └── pbft.py              ★ PBFT protocol: pre-prepare / prepare / commit,
│   │                              leader rotation, view-change fallback
│   ├── envs/
│   │   ├── __init__.py          ENV_REGISTRY  (name → wrapper) + make_env()
│   │   ├── mpe_wrapper.py       MPE simple_spread / simple_line
│   │   ├── smaclite_wrapper.py  SMAClite (discrete)
│   │   ├── vmas_wrapper.py      VMAS (continuous)
│   │   ├── lbf_wrapper.py       LBF
│   │   ├── permafrost_env.py    real-data monitoring environment
│   │   └── base.py              BaseEnv interface
│   ├── networks/
│   │   ├── actor_critic.py      GRU / MLP actor-critic
│   │   └── mixing_net.py        QMIX mixing network
│   └── utils/
│       ├── buffer.py            rollout buffer
│       ├── logger.py            metric logging → metrics.json
│       └── metrics.py           consensus rate, message overhead, convergence, …
├── configs/
│   ├── algo/*.yaml              algorithm + PBFT settings
│   └── env/*.yaml               environment settings
├── scripts/                     batch launchers, collectors, statistics  (see §7)
│   └── figures/                 figure-generation scripts                (see §9)
├── data/                        per-seed result exports                  (see §5)
├── experiments/                 experiment-design notes and phase scripts
├── baselines/                   independent baselines (e.g. krum_defense.py)
├── models/                      small reference checkpoints
├── legacy/                      one-off development patches (provenance only)
├── README.md  README_zh.md      project overview
├── DEPLOY.md  DEPLOY_zh.md      this guide (EN / ZH)
├── requirements.txt             dependencies
├── CITATION.cff  LICENSE  .gitignore
├── EXPERIMENT_RESULTS.md        frozen numbers from earlier development rounds
└── FIX_PLAN.md                  historical development log
```

### How a run is wired together

1. `train.py` parses the CLI, then loads two YAMLs: `--algo_config` and `--env_config`.
   Keys are **flattened into one namespace**; a top-level `pbft_f` therefore wins over a nested
   `pbft: {f: …}`.
2. `ENV_REGISTRY[--env]` builds the environment wrapper; `ALGORITHM_REGISTRY[--algo]` builds the
   trainer.
3. `PBFTCGMAPPO` calls `consensus/pbft.py` every `consensus_freq` steps to obtain the agreed
   action $a^{*}$, then applies one of two conditioning paths:

   | `conditioning_type` | `soft_consensus_mode` | Behaviour |
   |---|---|---|
   | `multiplicative` | `false` | **Hard**: the executed action is *replaced* by $a^{*}$ |
   | `additive` | `true` | **Soft**: action untouched; a bounded KL term is *added* to the loss |

   A third switch, `matched_logprob`, decides whether the PPO gradient is evaluated at
   $\log\pi(a^{*})$ (matched) or at the agent's own proposal (mismatched). **The whole paper is
   about the interaction of these three switches.**
4. `utils/logger.py` writes per-episode history to
   `<log_dir>/<algo>/seed_<N>/metrics.json` — note the **double nesting**.

### Byzantine injection

Enabled by two mechanisms used together (belt and braces):

```yaml
byzantine:            # in the algo YAML
  n: 1
  mode: random        # or: adversarial
  inject_at: proposal
```

```bash
--byzantine_n 1 --byzantine_type random     # on the command line
```

---

## 5. Data

### 5.1 Simulation environments — nothing to download

`mpe_spread`, `mpe_reference`, `smaclite_*`, `vmas_*`, `lbf_2s3f` are all synthesised at runtime by
`pettingzoo` / `vmas` / `gymnasium`. There is **no external dataset** behind the main results.

### 5.2 Permafrost environment (optional, opt-in)

`permafrost_monitoring` uses real third-party observations. Credentials must go through
environment variables and never touch a file:

```bash
export TPDC_EMAIL="you@example.com"
export TPDC_PWD="..."              # never write this into any file
python scripts/download_tpdc.py
unset TPDC_PWD
```

Please cite the provider if you use it:

> Zhao, L., Zou, D.F., Hu, G.J., et al. (2021). *A synthesis dataset of permafrost thermal state
> for the Qinghai–Tibet (Xizang) Plateau, China.* Earth System Science Data.
> Data provided by the National Tibetan Plateau / Third Pole Environment Data Center
> (http://data.tpdc.ac.cn).

### 5.3 Our own results — `data/`

The numeric outcome of **every** training run (721 runs, 93 experiment directories) is shipped in
reduced form:

| File | Rows | What it is |
|---|---:|---|
| `data/seeds_summary.csv` | 721 | One row per run; headline metrics as the mean of the last 10 eval points |
| `data/per_agent_entropy.csv` | 523 | One row per (run, agent) — honest-vs-faulty entropy decomposition |
| `data/seeds_tail10.csv.gz` | 67,606 | Raw last-10 points behind every number above |
| `data/export_data.tar.gz` | — | The three files bundled |

Full column-by-column documentation: **[`data/README.md`](data/README.md)**.

Regenerate / extend it on any machine that has `results/`:

```bash
python3 scripts/export_results.py --full
```

---

## 6. Verify the installation

```bash
# registry check
python -c "import sys; sys.path.insert(0,'.'); from src.envs import ENV_REGISTRY; print(list(ENV_REGISTRY))"
python -c "import sys; sys.path.insert(0,'.'); from src.algorithms import ALGORITHM_REGISTRY; print(list(ALGORITHM_REGISTRY))"

# 1-minute smoke run
python src/train.py --algo mappo --env mpe_spread \
    --env_config configs/env/mpe_spread_n5.yaml \
    --algo_config configs/algo/n5_mappo_fair.yaml \
    --seed 1 --n_timesteps 5000 --eval_interval 2500 --eval_episodes 2 \
    --log_dir results/_smoke
find results/_smoke -name metrics.json          # a printed path = end-to-end OK

# 5-minute pre-flight (covers data loading, a real forward/backward pass, and result writing)
bash scripts/verify_5min.sh 0                   # expect: 预检结果: ✅ N ❌ 0
```

Two lines in the pre-flight output deserve attention:

- **Byzantine self-check** — must report that injection traces appear in the log.
- **`f` value check** — must read `f=2` for the $n=7$ configs. If it reads `f=1`, the top-level
  `pbft_f` key was not picked up.

---

## 7. Running the experiments

Every launcher is **resumable** (completed seeds are skipped via a `metrics.json` check) and
**throttled** (concurrency is enforced by counting live `train.py` processes).

| Paper artefact | Launcher | Config family | Scale |
|---|---|---|---|
| Table 1 — clean setting | `scripts/run_n5_review.sh` | `configs/algo/n5_*.yaml` | $n{=}5$, 8 seeds, 300 K |
| Table 2 — causal matched/mismatched | `scripts/run_mpe_matched_v2.sh` | `n5_matched_logprob.yaml` | $n{=}5$, 8 seeds |
| Table 3 — Byzantine $f{=}1$ (per-agent entropy) | `scripts/run_e3_peragent_v2.sh` | `n5_*_byz_*.yaml` | $n{=}5$, $f{=}1$, 4 seeds |
| Table 3 — 4th cell (Hard-Matched, $f{=}1$) | `scripts/run_e3_matched.sh run` | `n5_hard_matched_byz_*.yaml` | 2 conditions × 4 seeds |
| Table 4 — higher fault ratio | `scripts/run_supplement.sh f2` + `scripts/run_hard_matched.sh run` | `supp_*_byz2_adv_n7.yaml` | $n{=}7$, $f{=}2$, 8 seeds |
| Table 5 — second task (MPE simple\_line) | `scripts/run_mpe_formal.sh` | `experiments/configs/fair_*.yaml` | $n{=}5$, 5 seeds |
| Everything, unattended | `scripts/run_all_serial.sh` | — | full queue |

Typical invocations:

```bash
cd /root/autodl-tmp/PBFT-CG-MARL-release

nohup bash scripts/run_n5_review.sh      > logs/n5_review.log   2>&1 &   # ≈ 3 h
nohup bash scripts/run_supplement.sh f2  > logs/supp_f2.log     2>&1 &   # ≈ 3.5 h
nohup bash scripts/run_hard_matched.sh run > logs/hard_matched.log 2>&1 & # ≈ 1.5 h
nohup bash scripts/run_e3_matched.sh run > logs/e3_matched.log  2>&1 &   # ≈ 1.5 h
```

> ⚠️ **Run only ONE batch script at a time.** Each takes a lock at `/tmp/<name>.lock`; two
> concurrent instances defeat the throttling and can exhaust VRAM.

Single run, straight from the CLI:

```bash
python src/train.py --algo pbft_cg_mappo --env mpe_spread \
    --env_config configs/env/mpe_spread_n5.yaml \
    --algo_config configs/algo/n5_hard_matched_byz_random.yaml \
    --seed 1 --n_timesteps 300000 --eval_interval 10000 --eval_episodes 10 \
    --gpu 0 --byzantine_n 1 --byzantine_type random \
    --log_dir results/manual/E3pa_matched_rand/seed_1
```

### CLI reference (`src/train.py`)

| Flag | Default | Notes |
|---|---|---|
| `--algo` | — | key in `ALGORITHM_REGISTRY` |
| `--env` | — | key in `ENV_REGISTRY`; **always pass it explicitly** (the default is unregistered, which silently drops the env config) |
| `--algo_config` / `--env_config` | — | YAML paths |
| `--seed` | 1 | |
| `--n_timesteps` | 1 000 000 | 300 K for all paper runs |
| `--eval_interval` / `--eval_episodes` | 5000 / 10 | the paper's entropy/reward convention uses the last 10 eval points |
| `--log_dir` | `./results` | results land in `<log_dir>/<algo>/seed_<N>/` |
| `--gpu` | 0 | |
| `--byzantine_n`, `--byzantine_type`, `--byzantine_mode` | 0 | inject $n$ faulty agents of the given behaviour |

---

## 8. Reading the results

### 8.1 Where results live

```
<log_dir>/<algo>/seed_<N>/metrics.json
```

The **double nesting** (`<algo>/seed_<N>/` inside a directory already named `seed_<N>`) is
intentional and is what the collectors glob for. If a run "finished" but you see no results, check
this exact path.

### 8.2 Inside `metrics.json`

A dictionary of time series (WandB style — each key holds an array):

| Key | Meaning |
|---|---|
| `episode/episode_reward` | episode return |
| `episode/episode_length` | episode length |
| `episode/consensus_rate` | PBFT agreement rate per eval point |
| `train/entropy` | mean policy entropy over **all** agents |
| `train/entropy_agent_<i>` | per-agent entropy (only when the per-agent patch is applied) |
| `train/consensus_loss` | PBFT consensus auxiliary loss |
| `train/kl_consensus_loss` | KL term of additive conditioning |
| `train/policy_loss`, `train/value_loss` | PPO losses |
| `train/entropy_coef`, `train/lr` | schedules in effect |

### 8.3 The reporting convention

Every scalar in the paper is the **mean of the last 10 evaluation points** of a series. Cross-seed
statistics (mean ± std, Welch's $t$, Cohen's $d$, Hedges' $g$, Mann–Whitney $U$, 95 % CI) are
computed over the per-seed values, never over raw timesteps.

```bash
# reproduce a quoted number, straight from the shipped CSV
python3 - <<'EOF'
import csv, statistics
rows = [r for r in csv.DictReader(open("data/seeds_summary.csv"))
        if r["experiment"] == "e3_peragent_v2" and r["label"] == "E3pa_matched_rand"]
vals = [float(r["entropy_all"]) for r in rows if r["entropy_all"]]
print(len(vals), "seeds |", round(statistics.mean(vals), 4), "±", round(statistics.stdev(vals), 4))
EOF
```

### 8.4 Collectors

```bash
python scripts/collect_n5_results.py      # main tables
python scripts/collect_results.py         # supplemental
python scripts/statistical_analysis.py    # Welch t, Cohen's d, Hedges' g, MWU, 95 % CI
python scripts/table3_full.py             # Table 3, complete 8-row version
python scripts/e3_peragent_report.py      # honest-only entropy decomposition
python scripts/export_results.py --full   # regenerate data/ from results/
```

**All reported numbers are already in `data/`** — the collectors are for re-deriving them from
your own runs.

---

## 9. Reproducing the figures

Figure scripts live in `scripts/figures/`. They read `metrics.json` (or the exported CSVs) and
write both a **300-dpi PNG** and a **vector PDF**.

```bash
cd /root/autodl-tmp/PBFT-CG-MARL-release
python3 scripts/figures/make_figures.py      # Figures 1–5 from results/
python3 scripts/figures/make_fig4_v8.py      # Figure 4 (2×2 Byzantine panel), data inlined
```

Outputs go to `scripts/figures/out/`.

### House style (apply this if you add a figure)

- **All text ≥ 12 pt.** Nothing smaller survives two-column typesetting.
- **Morandi palette, one colour per method**, consistent across every panel:
  MAPPO `#8E7CC3` · Hard (Mismatched) `#7FAF7F` · Hard (Matched) `#C3A36B` · Soft (Ours) `#4A7C96`
- **No title or caption inside the image** — the caption is a normal, editable paragraph in the
  manuscript, with the `(a)`/`(b)` panel tags kept inside the plot.
- Export **PNG at 300 dpi** *and* **PDF** (vector) from the same code path.
- Error bars = ±1 SD over seeds; state it in the caption, not the figure.

### Figure ↔ file map

| Paper figure | Script | Data source |
|---|---|---|
| Figure 1 — conditioning paths | schematic (no data) | — |
| Figure 2 — clean performance | `make_figures.py` | `results/` (Table 1 runs) |
| Figure 3 — entropy dynamics | `make_figures.py` | `results/`, 10 K-step running mean |
| Figure 4 — Byzantine tolerance (2×2) | `make_fig4_v8.py` | values inlined from Tables 3–4 |
| Figure 5 — second task + guidelines | `make_figures.py` | `results/` (Table 5 runs) |

> The `figures_download/Figure_*_Captain_Y.png` naming used during revision is a delivery
> convention, not part of the paper.

---

## 10. Monitoring & operations

```bash
find results -name metrics.json | wc -l          # how many runs have finished
bash scripts/run_hard_matched.sh status          # per-condition progress
tail -n 30 logs/hard_matched.log                 # non-blocking log peek (never `tail -f`)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
pgrep -fa src/train.py                           # live processes
pkill -f "src/train.py"                          # stop everything
```

**Good states.** MPE at $n=5$ runs at ~50–95 % GPU utilisation and 4 concurrent processes is the
sweet spot. VMAS is best at 2 concurrent (≈90 % GPU). GPU at 0 % between eval points is normal.

**Disk.** Keep the data volume under ~85 %. Purge `logs/` and stale `results/` before a long batch.

**Shutdown vs. release.** *Shutting down* preserves the data volume; *releasing* deletes
everything. Never release a machine holding un-collected results.

**Credentials.** Passwords and tokens go through environment variables only — never in source
files, configs, shell history, or command lines.

**Cross-machine moves.** Migrate only what cannot be regenerated (code, configs, results). Model
weights and caches can be re-downloaded.

---

## 11. Troubleshooting & FAQ

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the repository root, or add it to `PYTHONPATH`. |
| `--env_config` seems to do nothing | You forgot `--env mpe_spread`. The default `simple_spread` is not in `ENV_REGISTRY`, so the config is silently ignored. |
| `f` stays at 1 for the $n{=}7$ runs | Put `pbft_f: 2` **at the top level** of the YAML, not only inside the nested `pbft:` block. |
| Training "finished" but no results | Look for `<log_dir>/<algo>/seed_<N>/metrics.json` — the double nesting catches people out. |
| OOM / GPU thrashing | Reduce concurrency; make sure only one batch script is running. |
| Byzantine injection not visible in the log | Keep **both** the YAML `byzantine:` block and the CLI flags. |
| `nvidia-smi` shows 0 % while running | Normal for MPE between eval points; confirm with `pgrep -fa src/train.py`. |
| `python3: command not found` over SSH | Non-interactive shells skip conda init. Use `bash -lc "…"` or the absolute interpreter path. |
| `ModuleNotFoundError` for a package you just installed | You are in a different conda env — `conda activate pbft` again in the new terminal. |
| Ctrl-C does not stop training | `pkill -f "src/train.py"` — background jobs survive a closed shell. |
| Quoted paths break | Quote every path: `--algo_config "configs/algo/n5_hard_matched_byz_random.yaml"`. |

---

*See [`README.md`](README.md) for the project overview and [`data/README.md`](data/README.md) for the data dictionary.*
