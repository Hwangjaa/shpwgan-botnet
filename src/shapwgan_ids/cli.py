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

from . import __version__
from .config import DEFAULT_CONFIG_FILES, load_config
from .logging_utils import get_logger, setup_logging
from .paths import CONFIG_DIR, PROJECT_ROOT, dataset_dir, interim_dir

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

    for name, (_, description) in ROADMAP_STATUS.items():
        sub.add_parser(name, help=f"[not implemented yet] {description}")

    return parser


def _config(args: argparse.Namespace):
    cfg = load_config(Path(args.config_dir), DEFAULT_CONFIG_FILES, user_config=args.config, dotenv=PROJECT_ROOT / ".env")
    setup_logging(args.log_level or cfg.get_path("logging.level", "INFO"), cfg.get_path("logging.format", "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
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
    print(f"dataset        : profile={ds.get('profile')} every_path_ok={ds.get('exists')}")
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
    return cmd_not_implemented(args, cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
