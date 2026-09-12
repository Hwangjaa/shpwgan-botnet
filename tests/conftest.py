"""Shared fixtures: a tiny synthetic N-BaIoT-shaped corpus so tests never touch the real one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shapwgan_ids.config import Config

N_FEATURES = 4  # keep unit fixtures small; the real corpus has 115


def _write_csv(path, n_rows: int, rng: np.random.Generator, shift: float = 0.0) -> None:
    cols = [f"f{i}" for i in range(N_FEATURES)]
    frame = pd.DataFrame(rng.normal(shift, 1.0, size=(n_rows, N_FEATURES)), columns=cols)
    frame.to_csv(path, index=False)


@pytest.fixture
def synthetic_corpus(tmp_path, request):
    """Returns (cfg, corpus_dir) for a tmp corpus with metadata + rare-stratum files."""
    rare_rows = getattr(request, "param", 2)
    corpus = tmp_path / "raw" / "demo"
    corpus.mkdir(parents=True)
    rng = np.random.default_rng(7)

    _write_csv(corpus / "1.benign.csv", 200, rng, shift=0.0)
    _write_csv(corpus / "1.mirai.syn.csv", 150, rng, shift=3.0)
    _write_csv(corpus / "1.gafgyt.combo.csv", 120, rng, shift=-2.0)
    _write_csv(corpus / "2.benign.csv", 60, rng, shift=0.5)
    _write_csv(corpus / "2.mirai.udp.csv", rare_rows, rng, shift=4.0)

    # dataset metadata that must be ignored by the loader
    (corpus / "features.csv").write_text("feature\n" + "\n".join(f"f{i}" for i in range(N_FEATURES)) + "\n")
    (corpus / "data_summary.csv").write_text("a,b\n1,2\n")

    cfg = Config(
        {
            "seed": 3,
            "dataset": {
                "name": "N-BaIoT",
                "raw_dir": str(tmp_path / "raw"),
                "profiles": {"demo": {"subdir": "demo"}, "missing": {"subdir": "nope"}},
                "default_profile": "demo",
                "interim_dir": str(tmp_path / "interim"),
                "cache_format": "parquet",
                "max_rows_per_file": None,
            },
            "features": {"expected_count": N_FEATURES, "dtype": "float32"},
        }
    )
    return cfg, corpus
