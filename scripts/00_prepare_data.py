#!/usr/bin/env python3
"""Build/refresh the N-BaIoT parquet cache.

Thin wrapper so the same step exists as a plain script (for VS Code "Run" / debugging)
and as ``uv run swg prepare-data``.

    uv run python scripts/00_prepare_data.py --profile sampled
"""

from __future__ import annotations

import sys

from shapwgan_ids.cli import main

if __name__ == "__main__":
    argv = ["prepare-data", *sys.argv[1:]]
    raise SystemExit(main(argv))
