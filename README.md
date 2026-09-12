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

Raw N-BaIoT CSVs are **not** versioned here. Default location (override in
`.env` or `configs/data.yaml`):

```
/mnt/c/Users/Hwangja/OneDrive - Bina Nusantara/S2 - Cyber Security/Thesis/Code/N-BaIoT_10Percent
```

`Thesis/Code/Dataset/N-BaIoT` holds the full 9-device corpus (85 CSVs) if the sampled
subset is not enough. Feature set: 115 flow statistics, one row per 115-dim flow vector,
label derived from the filename (`<device>.<family>.<attack>.csv`, `benign` = normal).

```bash
uv run swg prepare-data --profile sampled    # CSV -> parquet cache in data/interim/
uv run swg prepare-data --profile full       # 9 devices, slower first run
```

## Roadmap

1. [x] Repo scaffold: uv env, config layers, data ingestion + parquet cache
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
uv run pytest -q               # fast unit tests
uv run ruff check . && uv run ruff format --check .
make help                      # shortcuts
```
