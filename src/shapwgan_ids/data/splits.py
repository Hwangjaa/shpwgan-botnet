"""Train/val/test splitting for N-BaIoT.

Two split strategies are needed by the thesis:

* :func:`stratified_split` -- standard train/val/test, stratified on (device, attack) so
  every attack family and device appears in each partition.
* :func:`leave_one_attack_out` -- hold out entire attack types to measure how an IDS
  trained on known attacks behaves against attacks it has never seen (robustness
  generalisation, used for the closed-loop scenarios).

Splitting must happen *after* :func:`~shapwgan_ids.data.loader.drop_leaky_duplicates`,
otherwise burst duplicates leak between partitions and inflate accuracy.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def _strata_key(frame: pd.DataFrame, columns: Sequence[str], min_count: int) -> pd.Series:
    """Stratification key; rare strata fall back to the coarse family column.

    sklearn refuses strata with fewer than 2 members, and an attack with 3 rows in the
    corpus should not break the split -- it is merged into its family bucket instead.
    """
    key = frame[list(columns)].astype(str).agg("|".join, axis=1)
    counts = key.value_counts()
    rare = counts[counts < min_count].index
    if len(rare):
        log.warning("merging %d rare strata into coarser buckets: %s", len(rare), list(rare)[:5])
        fallback = frame["attack_family"].astype(str)
        key = key.where(~key.isin(rare), "other|" + fallback)
    return key


def stratified_split(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify_by: Sequence[str] = ("device", "attack"),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Return ``{"train", "val", "test"}`` frames with reproducible stratification."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0, 1)")

    key = _strata_key(frame, stratify_by, min_count=2)
    train_val, test = train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=key, shuffle=True
    )
    out = {"test": test.reset_index(drop=True)}
    if val_size <= 0:
        out["train"] = train_val.reset_index(drop=True)
        out["val"] = train_val.iloc[0:0].reset_index(drop=True)
    else:
        rel_val = val_size / (1.0 - test_size)
        key_tv = _strata_key(train_val, stratify_by, min_count=2)
        train, val = train_test_split(
            train_val, test_size=rel_val, random_state=seed, stratify=key_tv, shuffle=True
        )
        out["train"] = train.reset_index(drop=True)
        out["val"] = val.reset_index(drop=True)

    for name in ("train", "val", "test"):
        part = out[name]
        n_attack = int((part["label"] == 1).sum())
        log.info("split %-5s rows=%7d attacks=%6d (%.1f%%)", name, len(part), n_attack, 100 * n_attack / max(len(part), 1))
    return out


def leave_one_attack_out(
    frame: pd.DataFrame,
    held_out_attacks: Sequence[str],
    normal_share: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split normal traffic proportionally and move held-out attacks entirely to test."""
    held = set(held_out_attacks)
    known = frame[~frame["attack"].isin(held)]
    unseen = frame[frame["attack"].isin(held)]
    if unseen.empty:
        raise ValueError(f"none of the held-out attacks {sorted(held)} exist in the corpus")

    rng = np.random.default_rng(seed)
    normal = known[known["label"] == 0]
    n_test_normal = int(round(len(normal) * normal_share))
    idx = rng.permutation(len(normal))
    normal_test = normal.iloc[idx[:n_test_normal]]
    normal_train = normal.iloc[idx[n_test_normal:]]

    train = pd.concat([normal_train, known[known["label"] == 1]], axis=0).reset_index(drop=True)
    test = pd.concat([normal_test, unseen], axis=0).reset_index(drop=True)
    log.info(
        "leave-one-attack-out: train=%d rows, test=%d rows (unseen attacks: %s)",
        len(train),
        len(test),
        sorted(held),
    )
    return train, test
