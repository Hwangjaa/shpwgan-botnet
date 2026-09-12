"""Subset builder: capping, determinism, materialised parquet and manifest honesty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shapwgan_ids.data.loader import discover_files
from shapwgan_ids.data.subsets import (
    build_subset,
    draw_unique_sample,
    load_subset,
    sample_strata,
    subset_spec,
    subset_summary,
)

CAP = 50  # matches dataset.subsets.demo.per_stratum_cap in conftest
EXPECTED_DISTINCT = 50 + 50 + 50 + 50 + 2  # four strata capped at 50 distinct, one tiny stratum kept whole


def _stratum(path, rows: pd.DataFrame) -> None:
    rows.to_csv(path, index=False)


def _vectors(rng: np.random.Generator, n_rows: int, shift: float) -> pd.DataFrame:
    return pd.DataFrame(rng.normal(shift, 1.0, size=(n_rows, 4)), columns=[f"f{i}" for i in range(4)])


def test_unknown_subset_name_lists_configured_ones(synthetic_corpus):
    cfg, _ = synthetic_corpus
    with pytest.raises(KeyError, match="demo"):
        subset_spec(cfg, "does-not-exist")


def test_sample_strata_caps_every_file(synthetic_corpus):
    _, corpus = synthetic_corpus
    files = discover_files(corpus)

    frame = sample_strata(files, CAP, seed=1)

    assert len(frame) == EXPECTED_DISTINCT
    per_stratum = frame.groupby(["device", "attack"], observed=True).size()
    assert (per_stratum <= CAP).all()
    # a stratum smaller than the cap must be kept whole, not thinned further
    assert per_stratum.loc[(2, "mirai.udp")] == 2


def test_sample_strata_is_reproducible(synthetic_corpus):
    _, corpus = synthetic_corpus
    files = discover_files(corpus)

    first = sample_strata(files, CAP, seed=7)
    second = sample_strata(files, CAP, seed=7)
    other = sample_strata(files, CAP, seed=8)

    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other), "a different seed must select different rows"


def test_sample_strata_rejects_bad_cap(synthetic_corpus):
    _, corpus = synthetic_corpus
    with pytest.raises(ValueError, match="per_stratum_cap"):
        sample_strata(discover_files(corpus), 0, seed=1)


def test_build_subset_materialises_disjoint_splits(synthetic_corpus):
    cfg, _ = synthetic_corpus

    manifest = build_subset(cfg, "demo")

    assert manifest["rows_after_dedup"] == EXPECTED_DISTINCT
    assert manifest["rows_drawn"] >= EXPECTED_DISTINCT
    assert manifest["features"] == 4
    assert manifest["disjoint_and_complete"] is True
    assert int(manifest["rows_after_dedup"]) > 0

    rows_on_disk = 0
    for split in ("train", "val", "test"):
        info = manifest["frames"][split]
        frame = load_subset(cfg, "demo", split)
        assert len(frame) == info["rows"]
        rows_on_disk += len(frame)
        assert list(frame.columns)[-5:] == ["device", "family", "attack", "attack_family", "label"]
    assert rows_on_disk == int(manifest["rows_after_dedup"])

    assert "subset        : demo" in subset_summary(manifest)


def test_build_subset_reuses_a_fresh_manifest(synthetic_corpus):
    cfg, _ = synthetic_corpus

    first = build_subset(cfg, "demo")
    second = build_subset(cfg, "demo")  # no force -> manifest is reused, not rebuilt

    assert second["created"] == first["created"]
    rebuilt = build_subset(cfg, "demo", force=True)
    assert rebuilt["rows_after_dedup"] == first["rows_after_dedup"]


def test_load_subset_rejects_unknown_split(synthetic_corpus):
    cfg, _ = synthetic_corpus
    build_subset(cfg, "demo")
    with pytest.raises(ValueError, match="split must be one of"):
        load_subset(cfg, "demo", "training")


def test_draw_unique_sample_tops_up_a_duplicate_heavy_stratum(tmp_path):
    """N-BaIoT repeats records, so a capped draw must keep drawing until the cap is met."""
    rng = np.random.default_rng(11)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _stratum(corpus / "1.benign.csv", _vectors(rng, 400, 0.0))
    repeated = _vectors(rng, 200, 3.0)  # only 200 distinct vectors ...
    _stratum(corpus / "1.mirai.syn.csv", pd.concat([repeated] * 3, ignore_index=True))  # ... in 600 rows

    frame, stats = draw_unique_sample(discover_files(corpus), 100, seed=5)

    assert len(frame) == 200  # 100 distinct per stratum
    assert frame.drop_duplicates().shape[0] == 200, "no duplicate vector may survive"
    mirai = stats["per_stratum"]["1.mirai.syn.csv"]
    assert mirai["distinct_kept"] == 100
    assert mirai["rounds"] > 1, "a single 100-row draw cannot yield 100 distinct vectors here"
    assert stats["rows_sampled"] > stats["rows_unique"]
    assert stats["strata_below_target"] == []
    assert stats["strata_at_target"] == 2


def test_draw_unique_sample_reports_strata_it_cannot_fill(tmp_path):
    """A stratum whose rows are all copies of an earlier one is reported, never faked."""
    rng = np.random.default_rng(13)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    base = _vectors(rng, 80, 2.0)
    _stratum(corpus / "1.gafgyt.combo.csv", base)
    _stratum(corpus / "2.gafgyt.combo.csv", base)  # byte-identical capture -> nothing unseen left

    frame, stats = draw_unique_sample(discover_files(corpus), 100, seed=1)

    assert len(frame) == 80, "the duplicate capture must not add a single new vector"
    second = stats["per_stratum"]["2.gafgyt.combo.csv"]
    assert second["distinct_kept"] == 0
    assert second["target_met"] is False
    assert stats["strata_below_target"] == ["1.gafgyt.combo.csv", "2.gafgyt.combo.csv"]
    assert stats["strata_at_target"] == 0
