#!/usr/bin/env python3
"""Capacity check: does one subset fit the laptop's RAM/VRAM in a sane amount of time?

This is a *feasibility probe*, not a thesis result: it answers "how long does one XGBoost
fit / one CNN epoch take on this subset on this machine?", so the subset size can be
justified instead of guessed.

    uv run python scripts/04_bench_subset.py --subset laptop
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from shapwgan_ids.config import DEFAULT_CONFIG_FILES, load_config
from shapwgan_ids.data.loader import META_COLUMNS
from shapwgan_ids.data.subsets import load_subset, subset_summary
from shapwgan_ids.paths import CONFIG_DIR, PROJECT_ROOT, artifacts_dir, subset_dir


def _features(frame: pd.DataFrame) -> np.ndarray:
    cols = [c for c in frame.columns if c not in META_COLUMNS]
    return np.ascontiguousarray(frame[cols].to_numpy(dtype=np.float32))


def bench_xgboost(train: pd.DataFrame, test: pd.DataFrame, n_estimators: int, jobs: int) -> dict[str, float]:
    import xgboost as xgb
    from sklearn.metrics import f1_score, roc_auc_score

    x_train, y_train = _features(train), train["label"].to_numpy()
    x_test, y_test = _features(test), test["label"].to_numpy()
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.3,
        tree_method="hist",
        n_jobs=jobs,
        eval_metric="logloss",
    )
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    proba = model.predict_proba(x_test)[:, 1]
    predict_s = time.perf_counter() - start
    return {
        "n_estimators": n_estimators,
        "fit_seconds": round(fit_s, 1),
        "predict_seconds": round(predict_s, 2),
        "test_macro_f1": round(float(f1_score(y_test, proba > 0.5, average="macro")), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }


def bench_cnn(train: pd.DataFrame, test: pd.DataFrame, batch_size: int, epochs: int) -> dict[str, float]:
    import torch
    from torch import nn

    if not torch.cuda.is_available():  # pragma: no cover - depends on the host
        raise RuntimeError("CUDA is not available; the CNN probe needs the GPU")

    device = torch.device("cuda")
    x_train = torch.from_numpy(_features(train)).unsqueeze(1)
    y_train = torch.from_numpy(train["label"].to_numpy().astype(np.float32))
    x_test = torch.from_numpy(_features(test)).unsqueeze(1)

    model = nn.Sequential(
        nn.Conv1d(1, 32, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, 5, padding=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(64, 1),
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    torch.cuda.reset_peak_memory_stats()
    epoch_seconds: list[float] = []
    model.train()
    for _ in range(epochs):
        started = time.perf_counter()
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            optimiser.step()
        torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - started)

    model.eval()
    with torch.no_grad():
        scores = []
        for chunk in x_test.split(16_384):
            scores.append(torch.sigmoid(model(chunk.to(device)).squeeze(1)).cpu())
        proba = torch.cat(scores).numpy()
    from sklearn.metrics import f1_score, roc_auc_score

    y_test = test["label"].to_numpy()
    return {
        "batch_size": batch_size,
        "steps_per_epoch": len(loader),
        "epoch_seconds_mean": round(float(np.mean(epoch_seconds)), 1),
        "epoch_seconds": [round(s, 1) for s in epoch_seconds],
        "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        "gpu_name": torch.cuda.get_device_name(0),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "test_macro_f1": round(float(f1_score(y_test, proba > 0.5, average="macro")), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="laptop")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--jobs", type=int, default=0, help="XGBoost threads (0 = all cores)")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(CONFIG_DIR, DEFAULT_CONFIG_FILES, dotenv=PROJECT_ROOT / ".env")
    manifest_path = subset_dir(cfg, args.subset) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    print(subset_summary(manifest))

    print("\nloading splits ...")
    start = time.perf_counter()
    train = load_subset(cfg, args.subset, "train")
    test = load_subset(cfg, args.subset, "test")
    load_s = time.perf_counter() - start
    print(f"  train={train.shape} test={test.shape} in {load_s:.1f}s")

    summary: dict[str, object] = {
        "subset": args.subset,
        "train_rows": len(train),
        "test_rows": len(test),
        "features": int(_features(train).shape[1]),
        "rows_in_memory_mb": round((train.memory_usage(deep=True).sum() + test.memory_usage(deep=True).sum()) / 1e6, 1),
        "load_seconds": round(load_s, 1),
    }
    summary["xgboost"] = bench_xgboost(train, test, args.trees, args.jobs)
    print(f"  xgboost: {summary['xgboost']}")
    summary["cnn"] = bench_cnn(train, test, args.batch_size, args.epochs)
    print(f"  cnn: {summary['cnn']}")

    out = args.json or (artifacts_dir(cfg) / "data" / f"bench_{args.subset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
