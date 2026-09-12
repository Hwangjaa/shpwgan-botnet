"""``swg`` command line entry point.

Implemented today: environment/dataset inspection and the parquet cache build.
The modelling subcommands are registered now so the interface is stable, but they exit
with a clear roadmap message until their stage is implemented -- no silent no-ops.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import DEFAULT_CONFIG_FILES, load_config
from .logging_utils import get_logger, setup_logging
from .paths import CONFIG_DIR, PROJECT_ROOT, dataset_dir

log = get_logger("swg")

ROADMAP_STATUS: dict[str, tuple[str, str]] = {
    "train-surrogate": ("step 2", "surrogate classifier used as SHAP/feedback black box"),
    "shap-rank": ("step 2", "SHAP importance ranking + Top-K immutable mask artifact"),
    "train-ids": ("step 3", "IDS oracles (XGBoost + 1D-CNN) and clean baseline metrics"),
    "train-wgan": ("step 4", "WGAN-GP perturbation generator with mask constraint"),
    "run-loop": ("step 5", "closed-loop retrain -> adapt -> co-evolve cycles"),
    "evaluate": ("step 6", "realism, attack-success and robustness metrics + thesis tables"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swg",
        description="Adaptive SHAP-WGAN closed-loop pipeline (N-BaIoT / IDS robustness).",
    )
    parser.add_argument("--version", action="version", version=f"shapwgan-ids {__version__}")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR, help="folder with the YAML layers")
    parser.add_argument("--config", type=Path, default=None, help="extra user config that overrides the layers")
    parser.add_argument("--log-level", default="INFO", help="DEBUG | INFO | WARNING | ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="environment, GPU and dataset report")
    p_info.add_argument("--profile", default=None, help="dataset profile to report on")
    p_info.add_argument("--json", type=Path, default=None, help="also write the report as JSON")

    p_prep = sub.add_parser("prepare-data", help="build/refresh the parquet cache for a profile")
    p_prep.add_argument("--profile", default=None)
    p_prep.add_argument("--force", action="store_true", help="ignore a fresh cache and rebuild")
    p_prep.add_argument("--json", type=Path, default=None, help="write the corpus summary as JSON")

    p_report = sub.add_parser("data-report", help="dataset characterisation + split sanity check")
    p_report.add_argument("--profile", default=None)
    p_report.add_argument("--force", action="store_true", help="rebuild the cache before reporting")
    p_report.add_argument("--json", type=Path, default=None, help="write the report as JSON")

    p_subset = sub.add_parser("make-subset", help="materialise a capped subset (train/val/test parquet)")
    p_subset.add_argument("--subset", default="laptop", help="name from dataset.subsets in configs/data.yaml")
    p_subset.add_argument("--force", action="store_true", help="rebuild even if the manifest is up to date")
    p_subset.add_argument("--json", type=Path, default=None, help="write the subset manifest as JSON")

    p_subinfo = sub.add_parser("subset-report", help="show a materialised subset and verify it on disk")
    p_subinfo.add_argument("--subset", default="laptop")
    p_subinfo.add_argument("--json", type=Path, default=None, help="write the verification report as JSON")

    for name, (_, description) in ROADMAP_STATUS.items():
        sub.add_parser(name, help=f"[not implemented yet] {description}")

    return parser


def _config(args: argparse.Namespace):
    cfg = load_config(
        Path(args.config_dir), DEFAULT_CONFIG_FILES, user_config=args.config, dotenv=PROJECT_ROOT / ".env"
    )
    setup_logging(
        args.log_level or cfg.get_path("logging.level", "INFO"),
        cfg.get_path("logging.format", "%(asctime)s %(levelname)-7s %(name)s: %(message)s"),
    )
    return cfg


def _torch_report() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    report: dict[str, object] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        report["vram_gb"] = round(props.total_memory / 1024**3, 2)
        report["capability"] = f"{props.major}.{props.minor}"
    return report


def _dataset_report(cfg, profile: str | None) -> dict[str, object]:
    from .data.loader import cache_paths

    try:
        folder = dataset_dir(cfg, profile)
    except KeyError as exc:
        return {"error": str(exc)}
    parquet, manifest = cache_paths(cfg, profile or cfg.get_path("dataset.default_profile"))

    report: dict[str, object] = {
        "profile": profile or cfg.get_path("dataset.default_profile"),
        "folder": str(folder),
        "exists": folder.is_dir(),
    }
    if folder.is_dir():
        csvs = [p for p in folder.glob("*.csv") if p.stem.split(".")[0].isdigit()]
        report["csv_files"] = len(csvs)
        report["csv_size_mb"] = round(sum(p.stat().st_size for p in csvs) / 1e6, 1)
    report["cache"] = str(parquet)
    report["cache_present"] = parquet.is_file()
    if manifest.is_file():
        report["cache_manifest"] = json.loads(manifest.read_text())
    return report


def cmd_info(args: argparse.Namespace, cfg) -> int:
    from .seeding import resolve_device

    report = {
        "project_root": str(PROJECT_ROOT),
        "config_dir": str(Path(args.config_dir)),
        "config_files": [n for n in DEFAULT_CONFIG_FILES if (Path(args.config_dir) / n).is_file()],
        "python": {"version": platform.python_version(), "executable": sys.executable, "prefix": sys.prefix},
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "torch": _torch_report(),
        "resolved_device": resolve_device(cfg.get_path("device", "auto")),
        "seed": cfg.get_path("seed"),
        "dataset": _dataset_report(cfg, args.profile),
        "artifacts_dir": str(cfg.get_path("paths.artifacts_dir")),
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2))

    torch_info = report["torch"]
    print(f"project        : {report['project_root']}")
    print(f"python         : {report['python']['version']} ({report['python']['executable']})")
    print(f"platform       : {report['platform']}")
    if torch_info.get("installed"):
        gpu = torch_info.get("device_name", "-") if torch_info.get("cuda_available") else "no CUDA device"
        print(f"torch          : {torch_info['version']} (cuda build {torch_info['cuda_build']}) -> {gpu}")
    else:
        print("torch          : NOT installed (run: uv sync)")
    print(f"device (config): {report['resolved_device']}   seed: {report['seed']}")
    ds = report["dataset"]
    print(f"dataset        : profile={ds.get('profile')} on_disk={ds.get('exists')}")
    print(f"                 {ds.get('folder')}")
    if ds.get("csv_files"):
        print(f"                 {ds['csv_files']} CSVs, {ds['csv_size_mb']} MB")
    if ds.get("cache_present"):
        manifest = ds.get("cache_manifest", {})
        print(f"                 cache: {ds['cache']} ({manifest.get('n_rows', '?')} rows)")
    else:
        print("                 cache: not built yet (uv run swg prepare-data)")
    if args.json:
        print(f"report written : {args.json}")
    return 0


def cmd_prepare_data(args: argparse.Namespace, cfg) -> int:
    from .data.loader import corpus_summary, load_corpus

    frame = load_corpus(cfg, profile=args.profile, force=args.force)
    summary = corpus_summary(frame)
    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"summary written: {args.json}")
    return 0


def cmd_data_report(args: argparse.Namespace, cfg) -> int:
    """Dataset characterisation: composition, duplicate flows and split sanity check."""
    from .data.loader import corpus_summary, iter_feature_columns, load_corpus
    from .data.splits import leave_one_attack_out, stratified_split

    frame = load_corpus(cfg, profile=args.profile, force=args.force)
    feats = iter_feature_columns(frame)
    dup_rows = int(frame.duplicated(subset=feats, keep="first").sum())

    test_size = float(cfg.get_path("splits.test_size", 0.2))
    val_size = float(cfg.get_path("splits.val_size", 0.1))
    stratify_by = tuple(cfg.get_path("splits.stratify_by", ["device", "attack"]))
    seed = int(cfg.get_path("splits.seed", cfg.get_path("seed", 42)))

    parts = stratified_split(frame, test_size=test_size, val_size=val_size, stratify_by=stratify_by, seed=seed)
    index_sets = [set(p.index) for p in parts.values()]
    disjoint = not (index_sets[0] & index_sets[1] or index_sets[0] & index_sets[2] or index_sets[1] & index_sets[2])
    covered = sum(len(p) for p in parts.values()) == len(frame)

    report: dict[str, object] = {
        **corpus_summary(frame),
        "duplicate_flows": dup_rows,
        "duplicate_share_pct": round(100 * dup_rows / max(len(frame), 1), 2),
        "split": {
            name: {
                "rows": len(part),
                "normal_rows": int((part["label"] == 0).sum()),
                "attack_rows": int((part["label"] == 1).sum()),
                "attacks": sorted(part["attack"].unique().tolist()),
            }
            for name, part in parts.items()
        },
        "split_disjoint": disjoint,
        "split_covers_all_rows": covered,
        "split_config": {"test_size": test_size, "val_size": val_size, "stratify_by": list(stratify_by), "seed": seed},
    }

    holdout = list(cfg.get_path("splits.attack_holdout", []) or [])
    if holdout:
        train, test = leave_one_attack_out(frame, holdout, seed=seed)
        report["leave_one_attack_out"] = {
            "held_out": holdout,
            "train_rows": len(train),
            "test_rows": len(test),
            "unseen_in_train": sorted(set(test["attack"]) & set(train["attack"]) - set(holdout)),
        }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2))

    print(f"corpus         : {report['rows']} rows x {report['features']} features")
    print(f"label balance  : normal={report['normal_rows']} attack={report['attack_rows']}")
    print(f"devices        : {report['devices']}")
    print(f"duplicate flows: {dup_rows} ({report['duplicate_share_pct']}%) -- drop before splitting")
    print("attacks:")
    for name, count in report["attacks"].items():
        print(f"  {name:<16} {count:>7}")
    print(f"split          : disjoint={disjoint} covers_all_rows={covered}")
    for name, part in report["split"].items():
        print(f"  {name:<5} rows={part['rows']:<7} attacks={part['attack_rows']:<7} kinds={len(part['attacks'])}")
    if holdout:
        lo = report["leave_one_attack_out"]
        print(f"leave-one-out  : held_out={lo['held_out']} train={lo['train_rows']} test={lo['test_rows']}")
    if args.json:
        print(f"report written : {args.json}")
    return 0 if (disjoint and covered) else 1


def cmd_make_subset(args: argparse.Namespace, cfg) -> int:
    """Draw a stratified-capped subset of a profile and freeze it as train/val/test parquet."""
    from .data.subsets import build_subset, subset_summary

    manifest = build_subset(cfg, args.subset, force=args.force)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(manifest, indent=2))
    print(subset_summary(manifest))
    if args.json:
        print(f"manifest written: {args.json}")
    return 0


def cmd_subset_report(args: argparse.Namespace, cfg) -> int:
    """Re-read a subset from disk and check it against its own manifest."""
    from .data.subsets import load_subset, subset_dir, subset_summary

    folder = subset_dir(cfg, args.subset, create=False)
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        print(f"subset {args.subset!r} not built yet ({manifest_path} missing).")
        print(f"run: uv run swg make-subset --subset {args.subset}")
        return 1

    manifest = json.loads(manifest_path.read_text())
    print(subset_summary(manifest))

    checks: dict[str, dict[str, Any]] = {}
    ok = True
    for split in ("train", "val", "test"):
        part = load_subset(cfg, args.subset, split)
        claimed = manifest["frames"][split]["rows"]
        n_feat = int(sum(1 for c in part.columns if c not in ("device", "family", "attack", "attack_family", "label")))
        checks[split] = {
            "rows_on_disk": len(part),
            "rows_in_manifest": int(claimed),
            "features": n_feat,
            "normal": int((part["label"] == 0).sum()),
            "attack": int((part["label"] == 1).sum()),
            "devices": sorted(part["device"].astype(int).unique().tolist()),
            "attack_kinds": int(part["attack"].nunique()),
        }
        ok &= len(part) == int(claimed)

    total = sum(c["rows_on_disk"] for c in checks.values())
    ok &= total == int(manifest["rows_after_dedup"])
    print(f"verify         : rows_on_disk_total={total} manifest={manifest['rows_after_dedup']} matches={ok}")
    for split, info in checks.items():
        print(
            f"  {split:<5} rows={info['rows_on_disk']:<7} feats={info['features']} "
            f"normal={info['normal']:<6} attack={info['attack']:<7} devices={len(info['devices'])} kinds={info['attack_kinds']}"
        )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"manifest": manifest, "verification": checks, "matches": ok}, indent=2))
        print(f"report written : {args.json}")
    return 0 if ok else 1


def cmd_not_implemented(args: argparse.Namespace, cfg) -> int:
    step, description = ROADMAP_STATUS[args.command]
    print(f"'{args.command}' is planned for {step}: {description}.")
    print("Nothing was executed and no placeholder numbers were produced.")
    print("See README.md -> Roadmap for the current stage.")
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config(args)
    if args.command == "info":
        return cmd_info(args, cfg)
    if args.command == "prepare-data":
        return cmd_prepare_data(args, cfg)
    if args.command == "data-report":
        return cmd_data_report(args, cfg)
    if args.command == "make-subset":
        return cmd_make_subset(args, cfg)
    if args.command == "subset-report":
        return cmd_subset_report(args, cfg)
    return cmd_not_implemented(args, cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
