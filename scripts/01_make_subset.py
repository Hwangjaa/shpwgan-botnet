#!/usr/bin/env python3
"""Materialise a laptop-sized subset of N-BaIoT (train/val/test parquet).

Same step as ``uv run swg make-subset --subset laptop``, as a plain script so VS Code
can run/debug it directly.

    uv run python scripts/01_make_subset.py --subset laptop
"""

from __future__ import annotations

import sys

from shapwgan_ids.cli import main

if __name__ == "__main__":
    argv = ["make-subset", *sys.argv[1:]]
    raise SystemExit(main(argv))
