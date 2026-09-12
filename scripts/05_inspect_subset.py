"""Inspect a built subset manifest: sizes, thin strata, and per-split kind coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/laptop")
manifest = json.loads((folder / "manifest.json").read_text())

print(
    f"subset={manifest['subset']} builder=v{manifest['builder_version']} cap={manifest['spec']['per_stratum_cap']} seed={manifest['spec']['seed']}"
)
print(
    f"rows: sampled={manifest['rows_drawn']:,} -> distinct={manifest['rows_after_dedup']:,} x {manifest['features']} features"
)
print(f"strata at target: {manifest['strata_at_target']}/{manifest['n_files']}")
print(f"below target ({len(manifest['strata_below_target'])}): {manifest['strata_below_target']}")

table = pd.DataFrame(manifest["per_stratum"]).T
for column in ("device", "source_rows", "rows_sampled", "distinct_kept", "rounds"):
    table[column] = pd.to_numeric(table[column])
print("\nthinnest strata after dedup:")
print(
    table.nsmallest(12, "distinct_kept")[
        ["device", "attack", "source_rows", "rows_sampled", "distinct_kept", "rounds", "target_met"]
    ].to_string()
)

print("\nper split:")
for name, part in manifest["frames"].items():
    print(
        f"  {name:<5} rows={part['rows']:<7} normal={part['normal_rows']:<6} attack={part['attack_rows']:<7} devices={len(part['devices'])} kinds={part['attack_kinds']} size={part['size_mb']} MB"
    )
    print(f"        attacks: {manifest['split'][name]['attacks']}")

print("\nkind coverage across splits + rows per kind:")
frames = {
    name: pd.read_parquet(folder / f"{name}.parquet", columns=["device", "attack", "label"])
    for name in ("train", "val", "test")
}
per_kind = pd.DataFrame({name: df["attack"].value_counts() for name, df in frames.items()}).fillna(0).astype(int)
per_kind["total"] = per_kind.sum(axis=1)
print(per_kind.sort_values("total").to_string())

missing = per_kind[(per_kind[["train", "val", "test"]] == 0).any(axis=1)]
print(f"\nkinds with an empty split: {len(missing)}")
if len(missing):
    print(missing.to_string())

print("\n(device, attack) strata per split:")
for name, df in frames.items():
    counts = df.groupby(["device", "attack"], observed=True).size().to_numpy()
    print(f"  {name:<5} strata={len(counts):<3} min_rows={int(counts.min()):<5} median={int(np.median(counts))}")
