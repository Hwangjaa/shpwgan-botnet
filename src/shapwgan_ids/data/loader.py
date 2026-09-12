"""N-BaIoT CSV discovery + labelled loading.

N-BaIoT ships one CSV per (device, attack) pair and no label column: the filename is
the label. Layout::

    <device>.<family>.<attack>.csv    e.g. 1.mirai.syn.csv, 2.gafgyt.junk.csv
    <device>.<benign>.csv             e.g. 1.benign.csv

Every row is one flow record of 115 numeric flow-statistical features spanning five
time windows (L5 ... L0.01) over three statistics (weight/mean/variance) plus
correlation/spectrum blocks; the exact feature list is in the dataset's features.csv.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ..config import Config
from ..paths import dataset_dir, interim_dir

log = logging.getLogger(__name__)

FILENAME_RE = re.compile(r"^(?P<device>\d+)\.(?P<rest>.+)\.csv$")

META_COLUMNS = ("device", "family", "attack", "attack_family", "label")
# label convention: 0 = normal traffic, 1 = attack traffic
LABEL_NORMAL, LABEL_ATTACK = 0, 1


@dataclass(frozen=True)
class DatasetFile:
    """One CSV of the corpus with its label decoded from the filename."""

    path: Path
    device: int
    family: str      # "benign" | "mirai" | "gafgyt" | ...
    attack: str      # "benign" | "mirai.syn" | "gafgyt.combo" | ...
    is_attack: bool

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def label(self) -> int:
        return LABEL_ATTACK if self.is_attack else LABEL_NORMAL

    @property
    def attack_family(self) -> str:
        """Coarse family used for stratified splits: 'benign' or the botnet name."""
        return self.family


def parse_stem(stem: str) -> tuple[int, str, str, bool]:
    """Decode ``'1.mirai.syn'`` -> (device=1, family='mirai', attack='mirai.syn', True)."""
    match = FILENAME_RE.match(f"{stem}.csv")
    if not match:
        raise ValueError(f"unexpected N-BaIoT filename: {stem!r} (expected <device>.<label>.csv)")
    device = int(match.group("device"))
    rest = match.group("rest")
    if rest == "benign":
        return device, "benign", "benign", False
    family = rest.split(".")[0]
    return device, family, rest, True


def discover_files(
    folder: Path,
    devices: Sequence[int] | None = None,
    attacks: Sequence[str] | None = None,
    require_features_csv: bool = True,
) -> list[DatasetFile]:
    """List usable CSVs in ``folder`` (skips the dataset's metadata CSVs)."""
    if not folder.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {folder}")

    meta_names = set()
    if require_features_csv:
        meta_names = {"features.csv", "data_summary.csv", "device_info.csv"}

    files: list[DatasetFile] = []
    for path in sorted(folder.glob("*.csv")):
        if path.name in meta_names:
            continue
        try:
            device, family, attack, is_attack = parse_stem(path.stem)
        except ValueError:
            log.debug("skipping non-corpus csv: %s", path.name)
            continue
        if devices and device not in set(devices):
            continue
        if attacks and attack not in set(attacks):
            continue
        files.append(DatasetFile(path=path, device=device, family=family, attack=attack, is_attack=is_attack))

    if not files:
        raise FileNotFoundError(f"no N-BaIoT CSVs matched in {folder}")
    return files


def read_dataset_file(
    entry: DatasetFile,
    max_rows: int | None = None,
    dtype: str = "float32",
) -> pd.DataFrame:
    """Read one CSV and attach its label columns (device/family/attack/label)."""
    frame = pd.read_csv(entry.path, nrows=max_rows, dtype=dtype, engine="c")
    if frame.empty:
        raise ValueError(f"empty dataset file: {entry.path}")
    frame["device"] = entry.device
    frame["family"] = entry.family
    frame["attack"] = entry.attack
    frame["attack_family"] = entry.attack_family
    frame["label"] = entry.label
    return frame


def build_frame(
    files: Sequence[DatasetFile],
    max_rows: int | None = None,
    dtype: str = "float32",
    validate_features: int | None = None,
) -> pd.DataFrame:
    """Concatenate several CSVs into one labelled frame (feature column order kept)."""
    parts = []
    for entry in files:
        frame = read_dataset_file(entry, max_rows=max_rows, dtype=dtype)
        if validate_features is not None and frame.shape[1] - len(META_COLUMNS) != validate_features:
            raise ValueError(
                f"{entry.path.name}: expected {validate_features} feature columns, "
                f"got {frame.shape[1] - len(META_COLUMNS)}"
            )
        parts.append(frame)
        log.debug("loaded %s (%d rows)", entry.path.name, len(frame))
    out = pd.concat(parts, axis=0, ignore_index=True)
    log.info("corpus assembled: %d rows x %d features", len(out), out.shape[1] - len(META_COLUMNS))
    return out


def source_fingerprint(files: Sequence[DatasetFile]) -> str:
    """Stable hash of (name, size, mtime) so a stale cache is detected, not trusted."""
    import hashlib

    digest = hashlib.sha256()
    for entry in sorted(files, key=lambda f: f.path.name):
        stat = entry.path.stat()
        digest.update(f"{entry.path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def cache_paths(cfg: Config, profile: str, base: Path | None = None) -> tuple[Path, Path]:
    folder = interim_dir(cfg, base)
    key = f"nbaiot_{profile}"
    return folder / f"{key}.parquet", folder / f"{key}.manifest.json"


def load_corpus(
    cfg: Config,
    profile: str | None = None,
    force: bool = False,
    base: Path | None = None,
) -> pd.DataFrame:
    """Return the labelled corpus for ``profile``, using the parquet cache when fresh.

    The cache is stored on the Linux filesystem on purpose: reading 130 MB of CSV from
    a 9p/OneDrive mount is an order of magnitude slower than parquet from ext4.
    """
    profile = profile or cfg.get_path("dataset.default_profile", "sampled")
    folder = dataset_dir(cfg, profile, base)
    if not folder.is_dir():
        from ..paths import describe_missing_dataset

        raise FileNotFoundError(describe_missing_dataset(cfg, base))

    files = discover_files(folder, devices=None, attacks=None)
    fingerprint = source_fingerprint(files)
    parquet, manifest = cache_paths(cfg, profile, base)

    if parquet.is_file() and manifest.is_file() and not force:
        meta = json.loads(manifest.read_text())
        if meta.get("fingerprint") == fingerprint:
            log.info("cache hit: %s (%d files, %d rows)", parquet.name, meta["n_files"], meta["n_rows"])
            return pd.read_parquet(parquet)
        log.info("cache stale (source changed) -> rebuilding")

    frame = build_frame(
        files,
        max_rows=cfg.get_path("dataset.max_rows_per_file"),
        validate_features=cfg.get_path("features.expected_count"),
    )
    parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet, index=False)
    manifest.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "n_files": len(files),
                "n_rows": int(len(frame)),
                "n_features": int(frame.shape[1] - len(META_COLUMNS)),
                "files": sorted(f.path.name for f in files),
            },
            indent=2,
        )
    )
    log.info("cached %d rows -> %s", len(frame), parquet)
    return frame


def corpus_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Row/attack composition used by ``swg info`` and thesis tables."""
    by_attack = frame.groupby("attack", observed=True).size().sort_values(ascending=False)
    by_device = frame.groupby("device", observed=True).size().sort_values(ascending=False)
    return {
        "rows": int(len(frame)),
        "features": int(frame.shape[1] - len(META_COLUMNS)),
        "normal_rows": int((frame["label"] == LABEL_NORMAL).sum()),
        "attack_rows": int((frame["label"] == LABEL_ATTACK).sum()),
        "devices": [int(d) for d in by_device.index],
        "attacks": {str(k): int(v) for k, v in by_attack.items()},
    }


def iter_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Feature columns in file order (all columns that are not label metadata)."""
    return [c for c in frame.columns if c not in META_COLUMNS]


def feature_matrix(frame: pd.DataFrame) -> "pd.DataFrame":
    return frame[iter_feature_columns(frame)]


def drop_leaky_duplicates(frame: pd.DataFrame, subset: Iterable[str] | None = None) -> pd.DataFrame:
    """Drop exact duplicate flow vectors (N-BaIoT has many repeated records per burst).

    Duplicates across train/test inflate IDS accuracy, so split *after* deduplication.
    """
    cols = list(subset) if subset is not None else iter_feature_columns(frame)
    before = len(frame)
    out = frame.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)
    log.info("deduplicated flows: %d -> %d rows (-%.1f%%)", before, len(out), 100 * (1 - len(out) / before))
    return out
