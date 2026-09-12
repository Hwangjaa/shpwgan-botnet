"""Per-stratum effective sample size: how many *distinct* flow vectors does each stratum have?

N-BaIoT repeats near-identical flow records within each capture, so a stratum's nominal row
count overstates how much independent data it holds. This script measures the real number:

    total rows, distinct feature vectors, duplicate share, top-1 vector frequency

Output: artifacts/data/stratum_uniques.json (+ a console table for the thesis appendix).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shapwgan_ids.data.loader import META_COLUMNS, discover_files
from shapwgan_ids.paths import PROJECT_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("folder", nargs="?", default=str(PROJECT_ROOT / "data/raw/N-BaIoT"))
parser.add_argument("--json", default=str(PROJECT_ROOT / "artifacts/data/stratum_uniques.json"))
args = parser.parse_args()

folder = Path(args.folder)
files = discover_files(folder)
print(f"{folder}: {len(files)} strata")

rows: list[dict[str, object]] = []
for entry in sorted(files, key=lambda f: f.path.name):
    frame = pd.read_csv(entry.path, dtype="float32", engine="c")
    feats = frame.drop(columns=[c for c in META_COLUMNS if c in frame.columns])
    total = len(feats)
    _, first_counts = np.unique(feats.to_numpy(), axis=0, return_counts=True)
    unique = len(first_counts)
    top = int(first_counts.max())
    rows.append(
        {
            "file": entry.path.name,
            "device": entry.device,
            "attack": entry.attack,
            "rows": int(total),
            "unique_rows": int(unique),
            "duplicate_share_pct": round(100 * (1 - unique / total), 2),
            "effective_rows": int(unique),
            "most_frequent_vector_count": top,
        }
    )
    print(
        f"  {entry.path.name:<22} rows={total:>7} unique={unique:>7} "
        f"dup={100 * (1 - unique / total):5.1f}%  top1={top:>5}",
        flush=True,
    )

table = pd.DataFrame(rows).sort_values("effective_rows")
out = Path(args.json)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(
        {
            "folder": str(folder),
            "strata": rows,
            "total_rows": int(table["rows"].sum()),
            "total_unique_rows": int(table["unique_rows"].sum()),
            "strata_below_1000_unique": table.loc[table["effective_rows"] < 1000, "file"].tolist(),
            "strata_below_3000_unique": table.loc[table["effective_rows"] < 3000, "file"].tolist(),
        },
        indent=2,
    )
)

print("\nweakest strata by effective sample size:")
print(
    table.head(15)[["file", "rows", "unique_rows", "duplicate_share_pct", "most_frequent_vector_count"]].to_string(
        index=False
    )
)
print(f"\ncorpus: {int(table['rows'].sum())} rows -> {int(table['unique_rows'].sum())} distinct vectors")
print(f"written: {out}")
