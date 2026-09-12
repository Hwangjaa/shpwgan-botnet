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

import numpy as np
import torch

from . import __version__
from .config import DEFAULT_CONFIG_FILES, load_config
from .logging_utils import get_logger, setup_logging
from .models.surrogate import SurrogateModel
from .paths import CONFIG_DIR, PROJECT_ROOT, dataset_dir
from .seeding import resolve_device

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

    p_train_surrogate = sub.add_parser("train-surrogate", help="train surrogate classifier on a subset")
    p_train_surrogate.add_argument("--subset", default="laptop", help="subset name from configs/data.yaml")
    p_train_surrogate.add_argument("--output", type=Path, default=None, help="path to write trained surrogate")

    p_shap_rank = sub.add_parser("shap-rank", help="compute SHAP ranking and feature mask from a surrogate")
    p_shap_rank.add_argument("--subset", default="laptop")
    p_shap_rank.add_argument("--surrogate", type=Path, required=True, help="path to surrogate model")
    p_shap_rank.add_argument("--output", type=Path, default=None, help="path to write mask JSON")
    p_shap_rank.add_argument("--coverage", type=float, default=None, help="cumulative importance threshold (Eq 2.9)")

    p_train_ids = sub.add_parser("train-ids", help="train IDS oracles (XGBoost + 1D-CNN) on a subset")
    p_train_ids.add_argument("--subset", default="laptop")
    p_train_ids.add_argument("--output-dir", type=Path, default=None)

    p_train_wgan = sub.add_parser("train-wgan", help="train WGAN-GP perturbation generator")
    p_train_wgan.add_argument("--subset", default="laptop")
    p_train_wgan.add_argument("--mask", type=Path, required=True, help="path to feature mask JSON")
    p_train_wgan.add_argument("--epochs", type=int, default=None)
    p_train_wgan.add_argument("--output", type=Path, default=None, help="path to write generator checkpoint")

    p_run_loop = sub.add_parser("run-loop", help="run one closed-loop co-evolution cycle")
    p_run_loop.add_argument("--subset", default="laptop")
    p_run_loop.add_argument("--cycles", type=int, default=None)
    p_run_loop.add_argument("--output-dir", type=Path, default=None)

    p_evaluate = sub.add_parser("evaluate", help="evaluate clean IDS baselines and adversarial robustness")
    p_evaluate.add_argument("--subset", default="laptop")
    p_evaluate.add_argument("--ids-dir", type=Path, default=None, help="directory with trained IDS oracles")
    p_evaluate.add_argument("--json", type=Path, default=None)

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


def _load_split_arrays(cfg, subset: str, split: str):
    """Load a subset split and return (X, y, feature_names)."""
    from .data.subsets import load_subset

    frame = load_subset(cfg, subset, split)
    meta_cols = {"device", "family", "attack", "attack_family", "label"}
    feature_cols = [c for c in frame.columns if c not in meta_cols]
    x = frame[feature_cols].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)
    return x, y, feature_cols


def cmd_train_surrogate(args: argparse.Namespace, cfg) -> int:
    from .models.surrogate import train_surrogate

    x_train, y_train, _feature_cols = _load_split_arrays(cfg, args.subset, "train")
    surrogate = train_surrogate(x_train, y_train, backend="xgboost")
    out = args.output or Path(cfg.get_path("paths.artifacts_dir")) / "surrogate" / f"{args.subset}_surrogate.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    surrogate.save(out)
    print(f"surrogate saved: {out}")
    return 0


def cmd_shap_rank(args: argparse.Namespace, cfg) -> int:
    import shap

    from .shap.mask import build_mask
    from .shap.ranking import aggregate_shap

    x_train, _y_train, feature_cols = _load_split_arrays(cfg, args.subset, "train")
    surrogate = SurrogateModel.load(args.surrogate)

    bg_size = min(cfg.get_path("shap.background_samples", 200), len(x_train))
    rng = np.random.default_rng(cfg.get_path("seed", 42))
    bg = x_train[rng.choice(len(x_train), size=bg_size, replace=False)]

    explainer = shap.TreeExplainer(surrogate.model)
    shap_values = explainer.shap_values(bg)
    importance = aggregate_shap(shap_values)

    out = args.output or Path(cfg.get_path("paths.artifacts_dir")) / "masks" / f"{args.subset}_mask.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    mask = build_mask(
        importance,
        feature_cols,
        top_k=cfg.get_path("shap.mask.top_k", 30),
        coverage=args.coverage,
        mode=cfg.get_path("shap.mask.mode", "immutable_topk"),
    )
    mask.save(out)
    print(f"mask saved     : {out}")
    print(f"immutable      : {mask.n_immutable}/{mask.n_features}")
    return 0


def cmd_train_ids(args: argparse.Namespace, cfg) -> int:
    from .eval.attack_success import evaluate_oracle
    from .models.ids_cnn import train_ids_cnn
    from .models.ids_xgb import train_ids_xgb

    x_train, y_train, _feature_cols = _load_split_arrays(cfg, args.subset, "train")
    x_val, y_val, _feature_cols = _load_split_arrays(cfg, args.subset, "val")
    out_dir = args.output_dir or Path(cfg.get_path("paths.artifacts_dir")) / "ids"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids_cfg = cfg.get_path("ids", {})
    xgb = train_ids_xgb(x_train, y_train, params=ids_cfg.get("xgboost"))
    xgb_path = out_dir / f"{args.subset}_ids_xgb.json"
    xgb.save_model(str(xgb_path))
    xgb_metrics = evaluate_oracle(xgb, x_val, y_val)
    print(f"XGBoost IDS    : {xgb_path} -> {xgb_metrics}")

    device = resolve_device(cfg.get_path("device", "auto"))
    cnn = train_ids_cnn(x_train, y_train, x_val=x_val, y_val=y_val, params=ids_cfg.get("cnn"), device=torch.device(device))
    cnn_path = out_dir / f"{args.subset}_ids_cnn.pt"
    torch.save(cnn.state_dict(), cnn_path)
    cnn_metrics = evaluate_oracle(cnn, x_val, y_val)
    print(f"CNN IDS        : {cnn_path} -> {cnn_metrics}")
    return 0


def cmd_train_wgan(args: argparse.Namespace, cfg) -> int:
    from .models.wgan import WGAN_GP
    from .shap.mask import FeatureMask

    x_benign_train, _y_benign, _feature_cols = _load_split_arrays(cfg, args.subset, "train")
    x_malicious_train, _y_mal, _feature_cols = _load_split_arrays(cfg, args.subset, "train")
    # Keep only benign for critic "real" samples and malicious for generator conditioning
    x_benign = x_benign_train[_y_benign == 0]
    x_malicious = x_malicious_train[_y_mal == 1]
    if len(x_benign) == 0 or len(x_malicious) == 0:
        print("subset must contain both benign and malicious rows")
        return 1

    mask = FeatureMask.load(args.mask)
    wgan_cfg = cfg.get_path("wgan", {})
    device = resolve_device(cfg.get_path("device", "auto"))
    wgan = WGAN_GP.from_config(n_features=x_benign.shape[1], cfg=wgan_cfg, device=device)

    epochs = args.epochs or wgan_cfg.get("training", {}).get("epochs", 200)
    batch_size = wgan_cfg.get("training", {}).get("batch_size", 512)

    x_benign_t = torch.tensor(x_benign, dtype=torch.float32, device=wgan.device)
    x_malicious_t = torch.tensor(x_malicious, dtype=torch.float32, device=wgan.device)
    mask_t = torch.tensor(~mask.immutable, dtype=torch.float32, device=wgan.device)

    n_batches = min(len(x_benign), len(x_malicious)) // batch_size
    for epoch in range(epochs):
        for _ in range(n_batches):
            idx_b = torch.randint(0, len(x_benign_t), (batch_size,))
            idx_m = torch.randint(0, len(x_malicious_t), (batch_size,))
            wgan.train_step(x_benign_t[idx_b], x_malicious_t[idx_m], mask=mask_t)
        if epoch % 10 == 0:
            print(f"epoch {epoch}/{epochs}")

    out = args.output or Path(cfg.get_path("paths.artifacts_dir")) / "wgan" / f"{args.subset}_wgan.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    wgan.save(out)
    print(f"WGAN saved     : {out}")
    return 0


def cmd_run_loop(args: argparse.Namespace, cfg) -> int:
    from .loop.cycle import LoopCycle
    from .models.wgan import WGAN_GP

    x_train, y_train, feature_cols = _load_split_arrays(cfg, args.subset, "train")
    x_benign, y_benign, _ = _load_split_arrays(cfg, args.subset, "val")
    x_malicious, y_malicious, _ = _load_split_arrays(cfg, args.subset, "val")
    x_benign = x_benign[y_benign == 0]
    x_malicious = x_malicious[y_malicious == 1]

    device = resolve_device(cfg.get_path("device", "auto"))
    wgan = WGAN_GP.from_config(n_features=x_train.shape[1], cfg=cfg.get_path("wgan", {}), device=device)
    loop_cfg = cfg.get_path("loop", {})
    cycles = args.cycles or loop_cfg.get("cycles", 5)

    cycle = LoopCycle(
        wgan=wgan,
        alpha=loop_cfg.get("feedback", {}).get("deceptive_weight", 1.0),
        confidence_threshold=loop_cfg.get("feedback", {}).get("confidence_target", 0.5),
        max_inner_steps=100,
        device=torch.device(device),
    )

    out_dir = args.output_dir or Path(cfg.get_path("paths.artifacts_dir")) / "loop"
    for c in range(cycles):
        print(f"=== cycle {c + 1}/{cycles} ===")
        result = cycle.run(
            cycle_id=c,
            x_train=x_train,
            y_train=y_train,
            x_benign=torch.tensor(x_benign, dtype=torch.float32, device=wgan.device),
            x_malicious=torch.tensor(x_malicious, dtype=torch.float32, device=wgan.device),
            feature_names=feature_cols,
            artifact_dir=out_dir,
        )
        print(f"stopped_by={result.stopped_by} mean_conf={result.mean_confidence:.4f} asr={result.attack_success_rate:.4f}")
    return 0


def cmd_evaluate(args: argparse.Namespace, cfg) -> int:

    _, y_val, _ = _load_split_arrays(cfg, args.subset, "val")
    print(f"evaluation placeholder: {len(y_val)} validation rows")
    return 0


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
    if args.command == "train-surrogate":
        return cmd_train_surrogate(args, cfg)
    if args.command == "shap-rank":
        return cmd_shap_rank(args, cfg)
    if args.command == "train-ids":
        return cmd_train_ids(args, cfg)
    if args.command == "train-wgan":
        return cmd_train_wgan(args, cfg)
    if args.command == "run-loop":
        return cmd_run_loop(args, cfg)
    if args.command == "evaluate":
        return cmd_evaluate(args, cfg)
    step, description = ROADMAP_STATUS[args.command]
    print(f"'{args.command}' is planned for {step}: {description}.")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
