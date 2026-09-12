"""Project path resolution. Everything is derived from the repo root or the config."""

from __future__ import annotations

from pathlib import Path

from .config import Config

# src/shapwgan_ids/paths.py -> repo root is two levels up
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_path(value: str | Path, base: Path | None = None) -> Path:
    """Resolve ``value``; relative paths are anchored at the repo root (or ``base``)."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base or PROJECT_ROOT) / path


def artifacts_dir(cfg: Config, base: Path | None = None, create: bool = True) -> Path:
    path = as_path(cfg.get_path("paths.artifacts_dir", "artifacts"), base)
    return ensure_dir(path) if create else path


def raw_data_dir(cfg: Config, base: Path | None = None) -> Path:
    return as_path(cfg.require("dataset.raw_dir"), base)


def profile_subdir(cfg: Config, profile: str | None = None) -> str:
    profiles = cfg.get_path("dataset.profiles", {}) or {}
    name = profile or cfg.get_path("dataset.default_profile")
    if name not in profiles:
        known = ", ".join(sorted(profiles)) or "<none>"
        raise KeyError(f"unknown dataset profile {name!r} (known: {known})")
    subdir = (profiles[name] or {}).get("subdir")
    if not subdir:
        raise KeyError(f"dataset profile {name!r} has no 'subdir' entry")
    return str(subdir)


def dataset_dir(cfg: Config, profile: str | None = None, base: Path | None = None) -> Path:
    """Absolute path of the CSV folder for ``profile`` (e.g. .../N-BaIoT_10Percent)."""
    return raw_data_dir(cfg, base) / profile_subdir(cfg, profile)


def interim_dir(cfg: Config, base: Path | None = None, create: bool = True) -> Path:
    path = as_path(cfg.get_path("dataset.interim_dir", "data/interim"), base)
    return ensure_dir(path) if create else path


def processed_dir(cfg: Config, base: Path | None = None, create: bool = True) -> Path:
    """Folder holding materialised subsets (train/val/test parquet)."""
    path = as_path(cfg.get_path("dataset.processed_dir", "data/processed"), base)
    return ensure_dir(path) if create else path


def subset_dir(cfg: Config, name: str, base: Path | None = None, create: bool = True) -> Path:
    return processed_dir(cfg, base, create=create) / name


def describe_missing_dataset(cfg: Config, base: Path | None = None) -> str:
    """Human-readable diagnosis when the configured dataset folder is absent."""
    raw = raw_data_dir(cfg, base)
    lines = [f"dataset root not found: {raw}"]
    if raw.parent.is_dir():
        siblings = sorted(p.name for p in raw.parent.iterdir() if p.is_dir())
        lines.append(f"folders available next to it: {siblings[:10]}")
    lines.append("set SHAPWGAN_DATASET__RAW_DIR in .env (or dataset.raw_dir in configs/data.yaml)")
    return "\n".join(lines)
