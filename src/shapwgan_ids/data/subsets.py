"""Laptop-sized subsets of N-BaIoT, materialised as train/val/test parquet.

The raw corpus (9 devices, 89 strata, ~7.06 M rows, 7.6 GB of CSV) is far too large for a
closed-loop experiment: every cycle retrains the IDS oracles and the generator, so the
working set has to be small enough that one cycle is minutes, not hours -- while still
covering every device and every attack kind, which is what the thesis claims generalises.

How the working set is built:

* **Per-stratum capping on *distinct* vectors.** ``per_stratum_cap`` is the number of
  *unique* flow vectors kept per ``(device, attack)`` stratum, not the number of rows read.
  N-BaIoT repeats near-identical records inside each capture (39% of the rows in the first
  712k-row draw were exact repeats), so a nominal row cap silently shrank some strata --
  and one stratum disappeared completely. The builder therefore keeps drawing extra rounds
  from a stratum until it has ``per_stratum_cap`` distinct vectors, or the source file runs
  out of unseen rows.
* **Global duplicate removal.** Duplicates are removed against *all* strata already kept,
  not just inside a stratum, so the same vector cannot appear both in train (device 1) and
  test (device 6). Files are processed in sorted name order, so the stratum that claims a
  shared vector is deterministic.
* **Honest reporting.** A stratum that cannot reach the cap (its remaining rows are copies
  of another stratum's) is listed in ``strata_below_target`` instead of being quietly
  padded or dropped.
"""

from __future__ import annotations

import json
import logging
import platform
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from ..config import Config
from ..paths import dataset_dir, subset_dir
from .loader import (
    LABEL_ATTACK,
    LABEL_NORMAL,
    META_COLUMNS,
    DatasetFile,
    discover_files,
    read_dataset_file,
    source_fingerprint,
)
from .splits import stratified_split

log = logging.getLogger(__name__)

SUBSET_ORDER = ("train", "val", "test")
# Bump whenever the sampling logic changes, so stale subsets are rebuilt instead of reused.
BUILDER_VERSION = 2


@dataclass(frozen=True)
class SubsetSpec:
    """Everything that defines a subset, so a rebuild is byte-for-byte reproducible."""

    name: str
    profile: str
    per_stratum_cap: int
    dedup: bool
    test_size: float
    val_size: float
    seed: int
    stratify_by: tuple[str, ...]


def subset_spec(cfg: Config, name: str) -> SubsetSpec:
    """Read ``dataset.subsets.<name>`` and fall back to the global split settings."""
    subsets = cfg.get_path("dataset.subsets", {}) or {}
    if name not in subsets:
        known = ", ".join(sorted(subsets)) or "<none>"
        raise KeyError(f"unknown subset {name!r} (configured: {known})")
    raw = subsets[name] or {}
    return SubsetSpec(
        name=name,
        profile=str(raw.get("profile", "full")),
        per_stratum_cap=int(raw.get("per_stratum_cap", 8000)),
        dedup=bool(raw.get("dedup", True)),
        test_size=float(raw.get("test_size", cfg.get_path("splits.test_size", 0.2))),
        val_size=float(raw.get("val_size", cfg.get_path("splits.val_size", 0.1))),
        seed=int(raw.get("seed", cfg.get_path("splits.seed", cfg.get_path("seed", 42)))),
        stratify_by=tuple(raw.get("stratify_by", cfg.get_path("splits.stratify_by", ["device", "attack"]))),
    )


def _rng(seed: int, round_no: int, entry: DatasetFile) -> np.random.Generator:
    """Per-file, per-round RNG: reproducible and independent of iteration order.

    ``crc32`` rather than :func:`hash`, because Python string hashing is salted per process.
    """
    return np.random.default_rng([seed, round_no, entry.device, zlib.crc32(entry.attack.encode())])


def _draw_rows(frame: pd.DataFrame, cap: int, seed: int, round_no: int, entry: DatasetFile) -> pd.DataFrame:
    """Draw up to ``cap`` rows at uniform random positions (whole frame when smaller)."""
    if len(frame) <= cap:
        return frame
    rng = _rng(seed, round_no, entry)
    picked = np.sort(rng.choice(len(frame), size=cap, replace=False))
    return frame.iloc[picked]


class _SeenVectors:
    """Sorted-array set of 64-bit row hashes, used to test 'have I kept this vector yet?'."""

    def __init__(self) -> None:
        self._array = np.empty(0, dtype=np.uint64)

    @property
    def size(self) -> int:
        return int(self._array.size)

    def add(self, hashes: np.ndarray) -> np.ndarray:
        """Record ``hashes`` and return a mask of the entries that were not seen before."""
        values = np.asarray(hashes, dtype=np.uint64)
        if self._array.size:
            position = np.clip(np.searchsorted(self._array, values), 0, self._array.size - 1)
            already = self._array[position] == values
        else:
            already = np.zeros(values.shape, dtype=bool)
        fresh = ~already
        if fresh.any():
            self._array = np.union1d(self._array, values[fresh])
        return fresh


def _row_hashes(frame: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    # hash_pandas_object is public but its decorators defeat type checkers -> ignore[call-arg]
    hashes = hash_pandas_object(frame[list(feature_cols)], index=False)  # type: ignore[call-arg]
    return np.asarray(hashes, dtype=np.uint64)


def sample_strata(
    files: Sequence[DatasetFile],
    per_stratum_cap: int,
    seed: int,
    dtype: str = "float32",
) -> pd.DataFrame:
    """One uniform-random draw of at most ``per_stratum_cap`` rows per stratum file.

    Strata smaller than the cap are kept whole -- the point is to lift small classes up to
    the cap, not to thin them. No duplicate bookkeeping happens here; see
    :func:`draw_unique_sample` for the builder used by the real subsets.
    """
    if per_stratum_cap < 1:
        raise ValueError("per_stratum_cap must be >= 1")

    parts = [first_draw(entry, per_stratum_cap, seed, dtype=dtype) for entry in sorted(files, key=lambda f: f.path.name)]
    out = pd.concat(parts, axis=0, ignore_index=True)
    log.info(
        "subset drawn: %d rows x %d features from %d strata (cap=%d rows/stratum, no dedup)",
        len(out),
        out.shape[1] - len(META_COLUMNS),
        len(files),
        per_stratum_cap,
    )
    return out


def first_draw(entry: DatasetFile, per_stratum_cap: int, seed: int, dtype: str = "float32") -> pd.DataFrame:
    """Round-0 draw for one stratum (public so callers can reproduce a single file)."""
    frame = read_dataset_file(entry, dtype=dtype)
    return _draw_rows(frame, per_stratum_cap, seed, 0, entry)


def draw_unique_sample(
    files: Sequence[DatasetFile],
    per_stratum_cap: int,
    seed: int,
    dtype: str = "float32",
    max_rounds: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep up to ``per_stratum_cap`` *distinct* vectors per stratum, deduplicated globally.

    Returns ``(frame, stats)`` where ``stats`` records, per stratum, how many rows were
    sampled, how many distinct vectors survived, the rounds used, and everything that could
    not reach the cap.
    """
    if per_stratum_cap < 1:
        raise ValueError("per_stratum_cap must be >= 1")

    ordered = sorted(files, key=lambda f: f.path.name)
    feature_cols = sorted(frame_columns(ordered[0]))
    seen = _SeenVectors()
    kept_parts: list[pd.DataFrame] = []
    per_stratum: dict[str, dict[str, Any]] = {}

    for entry in ordered:
        frame = read_dataset_file(entry, dtype=dtype)
        parts: list[pd.DataFrame] = []
        kept = 0
        sampled = 0
        rounds = 0
        empty_rounds = 0
        exhausted = len(frame) <= per_stratum_cap

        while kept < per_stratum_cap and rounds < max_rounds:
            rounds += 1
            drawn = _draw_rows(frame, per_stratum_cap, seed, rounds, entry)
            sampled += len(drawn)
            hashes = _row_hashes(drawn, feature_cols)
            fresh_mask = seen.add(hashes)
            if not fresh_mask.any():
                empty_rounds += 1
                if exhausted or empty_rounds >= 2:
                    break
                continue
            # a single round can draw two copies of the same vector: keep the first of each
            positions = np.flatnonzero(fresh_mask)
            _, first_of_each_hash = np.unique(hashes[positions], return_index=True)
            positions = positions[np.sort(first_of_each_hash)][: per_stratum_cap - kept]
            fresh = drawn.iloc[positions]
            parts.append(fresh)
            kept += len(fresh)
            if exhausted:
                break

        per_stratum[entry.path.name] = {
            "device": entry.device,
            "attack": entry.attack,
            "source_rows": int(len(frame)),
            "rows_sampled": int(sampled),
            "distinct_kept": int(kept),
            "rounds": int(rounds),
            "target_met": bool(kept >= per_stratum_cap),
        }
        if parts:
            kept_parts.append(pd.concat(parts, axis=0))

    frame = pd.concat(kept_parts, axis=0, ignore_index=True)
    if len(ordered) != len(per_stratum):
        raise AssertionError("stratum bookkeeping lost a file")

    feature_cols = [c for c in frame.columns if c not in META_COLUMNS]
    rows_sampled = int(sum(s["rows_sampled"] for s in per_stratum.values()))
    before = len(frame)
    frame = frame.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
    if len(frame) != before:  # only reachable through a 64-bit hash collision
        log.warning("exact dedup removed %d rows the hash set considered distinct", before - len(frame))

    below = sorted(name for name, s in per_stratum.items() if not s["target_met"])
    stats: dict[str, Any] = {
        "builder_version": BUILDER_VERSION,
        "per_stratum_cap": per_stratum_cap,
        "rows_sampled": rows_sampled,
        "rows_unique": int(len(frame)),
        "strata": len(per_stratum),
        "strata_at_target": int(len(per_stratum) - len(below)),
        "strata_below_target": below,
        "per_stratum": per_stratum,
    }
    log.info(
        "subset drawn: %d rows sampled -> %d distinct vectors x %d features from %d strata (target=%d distinct/stratum)",
        rows_sampled,
        len(frame),
        frame.shape[1] - len(META_COLUMNS),
        len(per_stratum),
        per_stratum_cap,
    )
    log.info(
        "strata at target: %d/%d%s",
        stats["strata_at_target"],
        stats["strata"],
        f" | below target: {below}" if below else "",
    )
    return frame, stats


def frame_columns(entry: DatasetFile) -> list[str]:
    """Feature columns of a stratum file, read from the header only."""
    with entry.path.open() as handle:
        header = handle.readline().strip().split(",")
    return [c for c in header if c not in META_COLUMNS]


def save_subset(out_dir: Path, parts: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> dict[str, Any]:
    """Write ``train/val/test`` parquet next to the manifest; returns the manifest payload."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, Any]] = {}
    for name in SUBSET_ORDER:
        part = parts[name]
        path = out_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        written[name] = {
            "path": str(path),
            "rows": int(len(part)),
            "normal_rows": int((part["label"] == LABEL_NORMAL).sum()),
            "attack_rows": int((part["label"] == LABEL_ATTACK).sum()),
            "devices": sorted(part["device"].astype(int).unique().tolist()),
            "attack_kinds": int(part["attack"].nunique()),
            "size_mb": round(path.stat().st_size / 1e6, 1),
        }
    payload = {**manifest, "frames": written}
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2))
    return payload


def build_subset(
    cfg: Config,
    name: str,
    force: bool = False,
    base: Path | None = None,
) -> dict[str, Any]:
    """Build (or reuse) the subset ``name``; returns the manifest as a dict."""
    spec = subset_spec(cfg, name)
    folder = dataset_dir(cfg, spec.profile, base)
    out_dir = subset_dir(cfg, name, base)
    manifest_path = out_dir / "manifest.json"

    if not folder.is_dir():
        from ..paths import describe_missing_dataset

        raise FileNotFoundError(describe_missing_dataset(cfg, base))

    files = discover_files(folder)
    fingerprint = source_fingerprint(files)
    if manifest_path.is_file() and not force:
        meta = json.loads(manifest_path.read_text())
        if (
            meta.get("fingerprint") == fingerprint
            and meta.get("spec", {}).get("per_stratum_cap") == spec.per_stratum_cap
            and meta.get("builder_version") == BUILDER_VERSION
        ):
            log.info("subset %r is up to date (%s)", name, manifest_path)
            return meta
        log.info("subset %r stale, built with a different cap, or older builder -> regenerating", name)

    frame, stats = draw_unique_sample(
        files,
        spec.per_stratum_cap,
        spec.seed,
        dtype=str(cfg.get_path("features.dtype", "float32")),
    )
    rows_drawn = int(stats["rows_sampled"])
    if spec.dedup is False and stats["rows_unique"] != len(frame):
        raise AssertionError("dedup=False is not supported by draw_unique_sample")

    parts = stratified_split(
        frame,
        test_size=spec.test_size,
        val_size=spec.val_size,
        stratify_by=spec.stratify_by,
        seed=spec.seed,
    )

    def counts(part: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(part)),
            "normal_rows": int((part["label"] == LABEL_NORMAL).sum()),
            "attack_rows": int((part["label"] == LABEL_ATTACK).sum()),
            "devices": sorted(part["device"].astype(int).unique().tolist()),
            "attacks": {str(k): int(v) for k, v in part["attack"].value_counts().sort_index().items()},
        }

    manifest: dict[str, Any] = {
        "subset": name,
        "builder_version": BUILDER_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": f"{platform.node()} ({platform.platform()})",
        "spec": {
            "profile": spec.profile,
            "source_folder": str(folder),
            "per_stratum_cap": spec.per_stratum_cap,
            "dedup": spec.dedup,
            "test_size": spec.test_size,
            "val_size": spec.val_size,
            "stratify_by": list(spec.stratify_by),
            "seed": spec.seed,
        },
        "fingerprint": fingerprint,
        "n_files": len(files),
        "features": len(part_feature_columns(frame)),
        "rows_drawn": rows_drawn,
        "rows_after_dedup": int(stats["rows_unique"]),
        "strata_at_target": stats["strata_at_target"],
        "strata_below_target": stats["strata_below_target"],
        "per_stratum": stats["per_stratum"],
        "memory_mb": round(frame.memory_usage(deep=False).sum() / 1e6, 1),
        "split": {name_: counts(part) for name_, part in parts.items()},
        "disjoint_and_complete": bool(
            sum(len(p) for p in parts.values()) == len(frame)
            and not (
                set(parts["train"].index) & set(parts["val"].index)
                or set(parts["train"].index) & set(parts["test"].index)
                or set(parts["val"].index) & set(parts["test"].index)
            )
        ),
    }
    payload = save_subset(out_dir, parts, manifest)
    log.info("subset %r written to %s", name, out_dir)
    log.info("data card:\n%s", subset_summary(payload))
    return payload


def part_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in META_COLUMNS]


def load_subset(cfg: Config, name: str, split: str = "train", base: Path | None = None) -> pd.DataFrame:
    """Read back one materialised split of a subset."""
    if split not in SUBSET_ORDER:
        raise ValueError(f"split must be one of {SUBSET_ORDER}, got {split!r}")
    path = subset_dir(cfg, name, base) / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"subset {name!r} has no {split} split at {path}; run: uv run swg make-subset --subset {name}")
    return pd.read_parquet(path)


def subset_summary(manifest: dict[str, Any]) -> str:
    """One-screen data card for the console and the thesis appendix."""
    spec = manifest["spec"]
    lines = [
        f"subset        : {manifest['subset']} (builder v{manifest.get('builder_version', '?')}, profile={spec['profile']}, "
        f"target={spec['per_stratum_cap']} distinct/stratum, seed={spec['seed']})",
        f"source        : {spec['source_folder']}",
        f"files / feats : {manifest['n_files']} strata, {manifest['features']} features",
        f"rows          : sampled={manifest['rows_drawn']} -> distinct={manifest['rows_after_dedup']}",
        f"strata        : at_target={manifest['strata_at_target']}/{manifest['n_files']}"
        + (f" | below_target={len(manifest['strata_below_target'])}" if manifest["strata_below_target"] else ""),
        f"in-memory     : {manifest['memory_mb']} MB (float32, features only)",
        f"invariants    : disjoint_and_complete={manifest['disjoint_and_complete']}",
    ]
    for name, part in manifest["frames"].items():
        lines.append(
            f"  {name:<5} rows={part['rows']:<7} attack={part['attack_rows']:<7} "
            f"devices={len(part['devices'])} kinds={part['attack_kinds']} "
            f"size={part['size_mb']} MB  {part['path']}"
        )
    return "\n".join(lines)
