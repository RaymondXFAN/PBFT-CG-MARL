# PBFT-CG-MARL — Deployment and Reproduction Guide

**PBFT-CG-MAPPO / Soft Consensus** — code and experiment scripts accompanying the paper
*"Soft Consensus Mitigates Entropy Collapse in Consensus-Guided Multi-Agent Reinforcement Learning"*.

This document is written for a reader who has cloned the repository and wants to reproduce the
results, or regenerate the figures. Every command below has been executed on a single
RTX 4090 (24 GB) node running Linux with Python 3.10.

---

## 1. What is in this repository

| Item | Purpose |
|---|---|
| `src/train.py` | Training entry point (MAPPO and PBFT-CG-MAPPO) |
| `src/algorithms/` | MAPPO and PBFT-CG-MAPPO implementations |
| `src/consensus/` | PBFT consensus layer |
| `src/envs/` | Environment wrappers (MPE, VMAS, SMAClite, LBF, permafrost) |
| `src/networks/`, `src/utils/` | Actor–critic networks, buffer, logger, metrics |
| `configs/env/`, `configs/algo/` | Environment and algorithm YAML configurations |
| `scripts/` | Analysis, statistics and table-generation scripts |
| `make_figures.py` | Regenerates Figures 1–5 from raw `results/` |
| `patch_per_agent_entropy.py` | Adds per-agent entropy logging (idempotent) |
| `patch_env_line.py` | Enables `simple_line` / `simple_formation` scenarios (idempotent) |
| `run_env2_mpe.sh`, `run_n5_review.sh` | Batch launchers for the two environment studies |
| `reproducibility/aggregates.json` | Aggregated numbers behind Figures 2–5 |

Raw `results/` and `logs/` are **not** tracked in git; see Section 4.4 for how they are laid out.

---

## 2. Requirements

### 2.1 Hardware

| Item | Minimum | Used in the paper |
|---|---|---|
| GPU | 1 × 8 GB | 1 × RTX 4090 (24 GB) |
| CPU | 4 cores | 16 vCPU |
| RAM | 16 GB | 64 GB |
| Disk | 10 GB | 100 GB |

Measured wall-clock times (300K steps per run):

| Environment | Agents | Per run | Safe concurrency |
|---|---|---|---|
| `mpe_spread` | 5 | 30–45 min | 4 |
| `mpe_line` | 5 | 70–75 min | 4 |
| `vmas_uav_coverage` | 3 | ≈6 h | 2 |

Running more than four MPE jobs at once does not increase throughput and causes memory pressure.

### 2.2 Software

Two supported ways to create the environment. Pick one — do not mix them.

**Option A — Conda**

```bash
conda create -n pbftmarl python=3.10 -y
conda activate pbftmarl
```

**Option B — venv**

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
```

A venv is simply a private folder holding this project's Python packages, so installing
something here cannot break other projects. `conda` is a package manager that also manages
the Python version itself; `pip` installs packages from PyPI.

### 2.3 Install dependencies

```bash
pip install --upgrade pip

# PyTorch: pick the wheel that matches your CUDA driver.
# CUDA 12.8 driver -> cu124 wheels are compatible (12.4 <= 12.8).
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Core libraries
pip install numpy pyyaml scipy matplotlib

# Environments: MPE2 is the one used for every result in the paper.
pip install "pettingzoo>=1.24" "mpe2>=1.1"
```

Optional (not required to reproduce the paper):

```bash
pip install vmas        # continuous-action control, Sections 5.3 and 5.5
```

Lock the exact versions of a working install with:

```bash
pip freeze > requirements-lock.txt
```

---

## 3. Smoke test (≈2 minutes)

Before launching anything long, verify that data loads, the model runs forward and backward,
and results are written to disk.

```bash
cd PBFT-CG-MARL

python -u src/train.py \
  --algo pbft_cg_mappo \
  --env mpe_spread \
  --env_config configs/env/mpe_spread_n5.yaml \
  --algo_config configs/algo/n5_soft_fair.yaml \
  --seed 1 --n_timesteps 5000 --eval_interval 5000 --eval_episodes 1 \
  --gpu 0 --log_dir /tmp/smoke
```

Success looks like:

```
指标已保存到 /tmp/smoke/pbft_cg_mappo/seed_1/metrics.json
```

Note the path: results are written to **`<log_dir>/<algo>/seed_<N>/metrics.json`**, not
`<log_dir>/metrics.json`.

Then check the file:

```bash
python -c "
import json; d = json.load(open('/tmp/smoke/pbft_cg_mappo/seed_1/metrics.json'))
print(len(d), 'metrics')
print([k for k in d if 'entropy' in k])"
```

You should see `train/entropy` plus `train/entropy_agent_0 … train/entropy_agent_{n-1}`.

---

## 4. Reproducing the paper

### 4.1 Main study — MPE simple_spread (Tables 1–4)

```bash
bash scripts/run_n5_review.sh          # 9 conditions x 4 seeds x 300K steps
```

Results land in `results/n5_review/<label>/<algo>/seed_<N>/metrics.json`.
The four clean conditions are `E1_mappo`, `E1_hard`, `E1_soft`, `E2_matched`; the five
faulted conditions are `E3_{mappo,hard,soft}_{rand,adv}` plus `E3_hard_adv`.
Seeds 5–8 of the clean conditions and the `f = 2` study are launched by the scripts in
`scripts/` and land in `results/supplement/`.

Expected total: 36 + 67 runs, roughly 30 h on one 4090 at concurrency 4.

### 4.2 Per-agent entropy re-run (Table 3 and Table A.3)

Per-agent entropy logging is applied by an idempotent patch:

```bash
python patch_per_agent_entropy.py      # backs up as *.bak_peragent, safe to re-run
```

Then re-run the faulted conditions with the fault injected on the **command line**:

```bash
bash run_e3_peragent_v2.sh             # results/e3_peragent_v2/E3pa_<method>_<fault>/seed_<N>/...
```

> The fault configuration in YAML was not honoured by an earlier code revision. Always pass
> `--byzantine_n`, `--byzantine_type` and `--byzantine_mode` on the command line (Section 7).

### 4.3 Second environment — MPE simple_line (Table 5)

```bash
python patch_env_line.py               # registers mpe_line / mpe_formation (idempotent)
python patch_env_line.py --check       # optional: prints the scenario table it applies

bash run_env2_mpe.sh probe             # 5-minute pre-flight, 1 method x 1 seed x 5K steps
bash run_env2_mpe.sh run               # 30 runs, roughly 9 h at concurrency 4
```

The launcher skips runs whose `metrics.json` already exists, so re-running the same command
resumes an interrupted batch instead of duplicating work.

### 4.4 Results layout

Two directory layouts appear in the repository because the batches were launched by different
scripts. Both end in `seed_<N>/metrics.json`:

```
results/n5_review/<label>/<algo>/seed_<N>/metrics.json          # main batch
results/supplement/<label>/seed_<N>/<algo>/seed_<N>/metrics.json # supplementary batch
results/e3_peragent_v2/<label>/seed_<N>/<algo>/seed_<N>/metrics.json
results/env2_mpe/<label>/seed_<N>/<algo>/seed_<N>/metrics.json
```

`make_figures.py` and `scripts/status_all.py` handle both layouts.

### 4.5 Figures (Figures 1–5)

```bash
python make_figures.py
```

Outputs to `figures_v7/`:

| File | Content | Paper figure |
|---|---|---|
| `fig1_conditioning.pdf/.png` | Multiplicative vs additive conditioning (schematic) | Figure 1 |
| `fig2_entropy_dynamics.pdf/.png` | Entropy vs training steps, 8 seeds, 95% CI | Figure 2 |
| `fig3_clean_performance.pdf/.png` | Clean reward and final entropy | Figure 3 |
| `fig4_byzantine.pdf/.png` | `f = 1` (reward, honest entropy) and `f = 2` (reward) | Figure 4 |
| `fig5_transfer_guidelines.pdf/.png` | Second-task transfer and the three guidelines | Figure 5 |
| `aggregates.json` | Aggregated numbers behind the plots | — |

Use the PDFs for submission and the PNGs for preview. The script requires
`results/` to be present.

### 4.6 Statistics and tables

```bash
python scripts/stats_supp.py                    # Welch t, Cohen's d, Hedges' g, Mann-Whitney
python scripts/collect_results.py               # per-condition summary table
python table3_markdown.py results/e3_peragent_v2 5   # Table 3 and Table A.3 in Markdown
python scripts/status_all.py                    # inventory: seeds present, gaps, stale runs
```

`table3_markdown.py` prints all-agent entropy, honest-agent entropy, reward and consensus rate,
plus the pairwise Welch tests, as ready-to-paste Markdown.

---

## 5. Configuration reference

### 5.1 Environment configs

```yaml
# configs/env/mpe_spread_n5.yaml
scenario: simple_spread     # simple_spread | simple_line | simple_formation | ...
n_agents: 5
n_landmarks: 3
max_steps: 25
continuous_actions: false
```

### 5.2 Algorithm configs

| Config | soft | matched log-prob | blend | freq | KL coef | entropy floor | Role |
|---|---|---|---|---|---|---|---|
| `n5_mappo_fair.yaml` | — | — | — | — | — | — | MAPPO baseline |
| `n5_hard_fair.yaml` | false | false | 1.0 | 1 | 0 | 0 | Hard mismatched |
| `n5_matched_logprob.yaml` | false | true | 1.0 | 1 | 0 | 0 | Hard matched |
| `n5_soft_fair.yaml` | true | true | 0.1 | 5 | 0.1 | 0.05 | Soft Consensus |

### 5.3 Configuration keys that the code actually reads

```
soft_consensus_mode, matched_logprob, kl_consensus_coef, entropy_floor,
consensus_loss_coef, consensus_freq, consensus_blend_ratio, kl_direction, pbft_f
```

Keys such as `conditioning_type`, `lambda_kl`, `consensus_mode` (standalone), `kl_coef`,
`entropy_floor_threshold` and `entropy_floor_coef` are **not read** — setting them has no
effect and will silently change nothing. If a configuration appears to have no effect, check
this list first.

### 5.4 Command-line arguments

```
--algo          mappo | pbft_cg_mappo
--env           mpe_spread | mpe_line | mpe_reference | vmas_uav_coverage | ...
--env_config    path to the environment YAML
--algo_config   path to the algorithm YAML
--byzantine_n   number of faulty agents (default 0)
--byzantine_type  random | adversarial | silence | message_forge | message_conflict | message_silent
--byzantine_mode  random | adversarial
```

There is **no** `--byzantine_faults` argument.

---

## 6. Operational notes

**Always pass `--env` and `--algo` explicitly.** The default environment name is not in the
registry, and a missing/unknown name causes the environment config to be ignored *silently*
rather than raising an error. Batches launched without `--env` can therefore appear to run
normally while training the wrong problem.

**Run long batches under `nohup`:**

```bash
nohup bash run_env2_mpe.sh run > logs/env2_run.log 2>&1 &
tail -f logs/env2_run.log                       # monitor
find results/env2_mpe -name metrics.json | wc -l   # progress (target 30)
```

**Concurrency control.** Throttle on the number of live processes, not on a task counter:

```bash
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 10; done; }
```

Counting tasks and calling `wait` every *N* launches fails whenever each iteration starts more
than one job: the counter jumps past the modulus and concurrency runs unbounded.

**Resuming.** Batch launchers skip any seed whose `metrics.json` already exists, so an
interrupted batch is resumed by re-running the same command.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'mpe2'` | MPE2 not installed | `pip install "mpe2>=1.1"` |
| `unrecognized arguments: --byzantine_faults` | No such flag | Use `--byzantine_n` + `--byzantine_type` |
| All checks report `metrics.json` missing, but training finished | Wrong path | Look in `<log_dir>/<algo>/seed_<N>/metrics.json` |
| Runs ignore the environment config | `--env` omitted or not in the registry | Pass a registered `--env` explicitly |
| All runs use the same algorithm | `--algo` omitted | Pass `--algo` explicitly |
| A config value has no effect | Dead key | See Section 5.3 |
| `TypeError: unexpected keyword argument 'N'` | Scenario has a fixed agent count | Only `simple_spread`, `simple_adversary`, `simple_line`, `simple_formation` accept `N`; `simple_reference`, `simple_tag`, `simple_world_comm` do not |
| Three fault metrics are all zero | Exception swallowed inside `loss_dict.get(...)`, or a float/tensor type mix broke the graph | Index with `loss_dict["key"]` so exceptions surface |
| Per-agent keys absent from `metrics.json` | Patch not applied | `python patch_per_agent_entropy.py` |
| A run wrote a very small `metrics.json` | Process died mid-run | Re-launch; the launcher overwrites |
| GPU idle while jobs are "running" | Fewer live processes than expected | `ps aux \| grep "[t]rain.py" \| wc -l` |

---

## 8. Package versions used for the published results

The exact environment is captured by `requirements-lock.txt` (`pip freeze` on the node that
produced the results). Key components:

```
python 3.10
torch 2.x (CUDA 12.4 wheels, driver 570.124.04)
pettingzoo >= 1.24
mpe2 >= 1.1
numpy, pyyaml, scipy, matplotlib
```

SMAClite and the Level-Based Foraging package are **not** dependencies: neither was usable in
our setting (the first is not distributed on PyPI and its repository was unreachable from our
node; the second exposes an interface incompatible with the parallel wrapper). The second task
in Table 5 is `mpe_line`, shipped with MPE2.

---

## 9. Citation

If you use this code, please cite the paper (reference to be added on publication) and the
environments:

- MAPPO — Yu et al., *The surprising effectiveness of PPO in cooperative multi-agent games*, NeurIPS 2022
- MPE / particle environments — Mordatch & Abbeel, AAAI 2018
- PBFT — Castro & Liskov, OSDI 1999

## 10. License

Released under the MIT License (see `LICENSE`). The environment wrappers under
`src/envs/` include code adapted from PettingZoo and MPE2, which retain their own licenses.
