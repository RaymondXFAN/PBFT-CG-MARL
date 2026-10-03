# `data/` — Per-seed experimental results

This folder holds the **numeric outcome of every training run** reported in the manuscript,
reduced to a small, reviewable form. It is the machine-readable counterpart of the tables and
figures in the paper.

Raw `metrics.json` files (≈1 MB each, 700+ of them) are deliberately **not** shipped — they are
fully regenerable from the code, and this reduced form is what reviewers actually need to
re-check a number.

---

## 1. Files

| File | Rows | Size | Contents |
|---|---:|---:|---|
| `seeds_summary.csv` | 721 | 136 KB | **One row per run.** All headline metrics, as the mean of the last 10 evaluation points. |
| `per_agent_entropy.csv` | 523 | 28 KB | One row per **(run, agent)**. Only for runs that were instrumented with per-agent logging. |
| `seeds_tail10.csv.gz` | 67,606 | 544 KB | One row per **(run, metric, timestep)** — the raw last-10 points behind every number above. Gzipped. |
| `export_data.tar.gz` | — | 523 KB | The three files above, bundled (this is the artefact to circulate). |

---

## 2. Column reference — `seeds_summary.csv`

| Column | Meaning |
|---|---|
| `experiment` | Top-level results directory, e.g. `e3_peragent_v2`, `supplement`, `review`. Groups runs by campaign. |
| `label` | Sub-directory identifying the condition, e.g. `E3pa_matched_rand`, `E3f2_hard_matched`. |
| `seed` | Random seed. |
| `algo` | Algorithm key: `pbft_cg_mappo` (our method), `mappo`, `qmix`, … |
| `n_episodes` | Length of the logged series (episodes, not timesteps). |
| `episode_reward` | Episode reward ↑ — **higher is better**. |
| `episode_length` | Episode length (steps). |
| `consensus_rate` | Fraction of steps on which PBFT reached agreement. `—`/blank for non-consensus baselines. |
| `entropy_all` | Policy entropy averaged over **all** $n$ agents, including faulty ones. Nats. |
| `entropy_honest_only` | Policy entropy averaged over the **honest agents only** (index ≥ number of injected faults). Blank when per-agent logging is absent. |
| `consensus_loss` | PBFT consensus auxiliary loss. |
| `kl_consensus_loss` | KL term of additive conditioning (0 for hard/multiplicative variants). |
| `policy_loss`, `value_loss` | PPO losses. |
| `entropy_coef` | Entropy bonus coefficient in effect. |
| `entropy_floor_pen` | Entropy-floor penalty term. |
| `lr` | Learning rate. |
| `n_agents_logged` | How many agents were logged individually (0 = no per-agent instrumentation). |
| `path` | Original `metrics.json` path on the training machine — the double nesting `<log_dir>/<algo>/seed_<N>/` is normal. |

### `per_agent_entropy.csv`

`experiment, label, seed, algo, agent_index, entropy` — `agent_index` 0 is the **faulty** agent
whenever Byzantine injection was active (faulty agents are injected first), and
`agent_index ≥ f` are honest. `entropy_honest_only` in the summary table is the mean over those.

---

## 3. Metric convention (important)

Every scalar in `seeds_summary.csv` is the **mean of the last 10 evaluation points** of that
series — the same convention used throughout the paper. Cross-seed statistics (mean ± std,
Welch's $t$, Cohen's $d$, Hedges' $g$, Mann–Whitney $U$) are then computed over the per-seed
values, **not** over raw timesteps.

To reproduce a quoted number: take the rows with the matching `experiment`+`label`, collect one
value per seed, and average.

```bash
# example: mean entropy over all seeds of one condition
python3 - <<'EOF'
import csv, statistics
rows = [r for r in csv.DictReader(open("data/seeds_summary.csv"))
        if r["experiment"] == "e3_peragent_v2" and r["label"] == "E3pa_matched_rand"]
vals = [float(r["entropy_all"]) for r in rows if r["entropy_all"]]
print(len(vals), "seeds | mean =", round(statistics.mean(vals), 4),
      "| std =", round(statistics.stdev(vals), 4))
EOF
```

---

## 4. How it was produced

```bash
python3 scripts/export_results.py --full     # run from the repository root, on the training machine
```

The exporter walks `results/**/metrics.json`, extracts the head-line metrics, and writes this
folder. It is idempotent and dependency-free (standard library only).

---

## 5. Coverage

| | |
|---|---:|
| Runs | **721** |
| Distinct `experiment` directories | **93** |
| Distinct (`experiment`, `label`) pairs | **400** |
| Runs with per-agent logging | **110** |

The archive contains the full development history — early prototypes and abandoned variants
included — so that every intermediate claim in the paper's revision trail can be traced back to
its source run. The subsets that feed the manuscript's final tables are listed in
[`../DEPLOY.md`](../DEPLOY.md) §7.
