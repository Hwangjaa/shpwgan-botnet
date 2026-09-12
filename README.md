# shapwgan-ids

Experiment pipeline for the thesis **"Adaptive SHAP-WGAN closed-loop feedback for
adversarial traffic generation against IoT IDS"** (N-BaIoT, flow-statistical features).

Perspective: evaluator/attacker. The deliverable is a measurement of IDS robustness
under feature-aware, adaptively generated perturbations -- not a new IDS.

## Components (mapped to BAB III of the thesis)

| Thesis component | Code location |
| --- | --- |
| N-BaIoT ingestion, labeling, splits | `src/shapwgan_ids/data/` |
| SHAP feature ranking -> Top-K immutable mask | `src/shapwgan_ids/shap/` |
| Surrogate classifier (black-box confidence feedback) | `src/shapwgan_ids/models/` |
| WGAN-GP perturbation generator | `src/shapwgan_ids/models/` |
| IDS target oracles (XGBoost, 1D-CNN) | `src/shapwgan_ids/models/` |
| Closed-loop (retrain -> adapt -> co-evolve cycles) | `src/shapwgan_ids/loop/` |
| Realism + attack-success + robustness metrics | `src/shapwgan_ids/eval/` |

## Environment

* WSL2 Ubuntu, project lives on the Linux filesystem (`~/dev/shapwgan-ids`) -- never on
  `/mnt/c` or OneDrive, because training reads/writes thousands of small files and 9p
  mounts are slow.
* Python 3.12 managed by `uv` (not the system python; Ubuntu 26.04 blocks global pip).
* GPU: RTX 3060 Laptop 6 GB via CUDA passthrough. PyTorch is pinned to `2.9.1+cu126`
  (CUDA 12.x wheel) so it stays compatible with the installed driver branch.

## Setup

```bash
uv sync                 # create/refresh .venv from pyproject + uv.lock
uv run swg info         # env + GPU + dataset sanity report
cp .env.example .env    # optional local path overrides
```

Open in VS Code from WSL so the interpreter and terminal resolve to the Linux env:

```bash
code ~/dev/shapwgan-ids
```

## Dataset

The corpus is used from a **local working copy on ext4** (`data/raw/`, git-ignored,
8.1 GB / 7.6 GiB total). The OneDrive original stays read-only and untouched; point
`SHAPWGAN_DATASET__RAW_DIR` (see `.env.example`) back at it if the copy is deleted.

```
data/raw/N-BaIoT/              # full corpus: 9 devices, 89 (device, attack) strata,
                               # 7,062,606 rows x 115 features, 8.1 GB / 7.6 GiB
data/raw/N-BaIoT_10Percent/    # 10% sample of devices 1-2, quick iteration
```

Label comes from the filename (`<device>.<family>.<attack>.csv`, `benign` = normal);
there is no label column. Reading CSVs straight off the OneDrive/9p mount is roughly an
order of magnitude slower than parquet on ext4, hence the copy + caches.

### Parquet cache (whole profile)

```bash
uv run swg prepare-data --profile full       # 7.06 M rows, slower first run
uv run swg prepare-data --profile sampled    # N-BaIoT_10Percent
uv run swg data-report --profile sampled     # composition, duplicates, split sanity
```

### Working subsets (what experiments actually train on)

The full corpus is too big for a closed-loop experiment: every cycle retrains the IDS
oracles and the generator. `make-subset` draws **distinct** flow vectors per
`(device, attack)` file until `per_stratum_cap` is met, so

* all 9 devices and all 11 attack kinds stay represented (nothing silently disappears),
* no single stratum can dominate (`mirai.udp` alone is 17% of the raw corpus),
* the result is reproducible and documented in a manifest with the source fingerprint.

The cap counts *distinct* vectors, not raw rows, because N-BaIoT repeats records
heavily: 5,215,807 of 7,062,606 rows are unique overall, and some captures are almost a
single repeated flow (`1.gafgyt.tcp.csv`: 94 distinct vectors in 100,313 rows; one vector
occurs 92,013 times). Duplicates are dropped **globally** before splitting, which also
removes vectors shared *between devices* -- otherwise the same flow could be trained on
from device 1 and tested on from device 6.

| subset | cap (distinct/stratum) | rows kept | train / val / test | RAM (float32) | intended use |
| --- | --- | --- | --- | --- | --- |
| `laptop` | 4,000 | 263,441 | 184,408 / 26,344 / 52,689 | 134 MB | main experiments: closed loop, WGAN-GP, SHAP |
| `smoke` | 500 | 35,537 | 24,875 / 3,554 / 7,108 | 18 MB | unit tests, SHAP debugging, quick smoke runs |

```bash
uv run swg make-subset --subset laptop       # -> data/processed/laptop/{train,val,test}.parquet
uv run swg subset-report --subset laptop     # re-read from disk and verify against manifest
uv run swg make-subset --subset smoke
make bench-laptop                            # capacity check: XGBoost fit + CNN epoch timings
```

Measured on this laptop (RTX 3060 6 GB, 16 cores, WSL): parquet load 0.2 s,
XGBoost 200 trees on 184,408 rows in 3.8 s, 1D-CNN epoch 2-5 s at 625 MB VRAM. One closed
loop re-fits the oracles several times, so the 263k-row subset leaves a wide margin; VRAM,
not dataset size, is the binding constraint when the cap is raised.

Every build writes `data/processed/<subset>/manifest.json`: caps, seed, source
fingerprint, per-stratum draw statistics, row counts per split, and the
`disjoint_and_complete` invariant. The split is stratified on `(device, attack)`, so each
partition sees all 9 devices and all 11 kinds.

Two honesty notes carried in the manifest:

* 63 of 89 strata reach the 4,000-vector cap; the other 26 are duplicate-degenerate
  (`gafgyt.tcp`/`gafgyt.udp` on every device, and `gafgyt.junk`/`gafgyt.scan` on devices
  6-9). Those two kinds therefore contribute only ~90 rows each to the `laptop` subset
  (85 and 95 rows) -- enough to remain represented, not enough for per-kind statistics.
* attack traffic is 86.3% of the subset, matching the real N-BaIoT imbalance; class
  balancing, if wanted, is a *training-time* decision (weights / resampling), not a
  property of the frozen subset.

## Roadmap

1. [x] Repo scaffold: uv env, config layers, local dataset copy, ingestion + parquet
       cache, laptop-sized stratified subsets (train/val/test parquet + manifest)
2. [ ] Surrogate classifier + SHAP feature ranking, Top-K immutable mask
3. [ ] IDS oracle training (XGBoost, 1D-CNN) + clean baseline metrics
4. [ ] WGAN-GP perturbation generator with mask constraint
5. [ ] Closed-loop driver: retrain detector -> adapt generator -> multi-cycle co-evolution
6. [ ] Evaluation: realism (distance/statistics), attack success rate, robustness curves
7. [ ] Export thesis tables/figures from `artifacts/`

> Thesis note: any numbers produced before step 6 are development placeholders and must
> never be reported as research results.

## Common commands

```bash
uv run swg info                # environment + dataset report
uv run swg prepare-data        # build parquet cache
uv run swg make-subset         # materialise the laptop subset (train/val/test parquet)
uv run swg subset-report        # re-read a built subset from disk and verify
uv run python scripts/04_bench_subset.py --subset laptop   # capacity check (XGBoost + CNN)
uv run pytest -q               # fast unit tests
uv run ruff check . && uv run ruff format --check .
make help                      # shortcuts
```
