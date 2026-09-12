"""Loader behaviour on a synthetic corpus shaped like N-BaIoT."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from shapwgan_ids.data.loader import (
    META_COLUMNS,
    build_frame,
    cache_paths,
    corpus_summary,
    discover_files,
    drop_leaky_duplicates,
    iter_feature_columns,
    load_corpus,
    parse_stem,
    source_fingerprint,
)


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("1.benign", (1, "benign", "benign", False)),
        ("9.mirai.udpplain", (9, "mirai", "mirai.udpplain", True)),
        ("2.gafgyt.combo", (2, "gafgyt", "gafgyt.combo", True)),
    ],
)
def test_parse_stem(stem, expected):
    assert parse_stem(stem) == expected


def test_parse_stem_rejects_foreign_names():
    with pytest.raises(ValueError):
        parse_stem("features")


def test_discover_files_skips_dataset_metadata(synthetic_corpus):
    _, corpus = synthetic_corpus
    files = discover_files(corpus)
    assert len(files) == 5
    assert {f.path.name for f in files} == {
        "1.benign.csv",
        "1.mirai.syn.csv",
        "1.gafgyt.combo.csv",
        "2.benign.csv",
        "2.mirai.udp.csv",
    }
    assert [f.device for f in discover_files(corpus, devices=[2])] == [2, 2]


def test_build_frame_labels_and_feature_count(synthetic_corpus):
    _, corpus = synthetic_corpus
    frame = build_frame(discover_files(corpus), validate_features=4)
    assert len(frame) == 200 + 150 + 120 + 60 + 2
    assert list(frame.columns[-len(META_COLUMNS) :]) == list(META_COLUMNS)
    assert len(iter_feature_columns(frame)) == 4
    assert frame.loc[frame["attack"] == "benign", "label"].eq(0).all()
    assert frame.loc[frame["attack"] == "mirai.syn", "label"].eq(1).all()

    summary = corpus_summary(frame)
    assert summary["rows"] == len(frame)
    assert summary["features"] == 4
    assert summary["attack_rows"] == 150 + 120 + 2
    assert summary["attacks"]["mirai.syn"] == 150


def test_validate_features_mismatch_raises(synthetic_corpus):
    _, corpus = synthetic_corpus
    with pytest.raises(ValueError, match="expected 115 feature columns"):
        build_frame(discover_files(corpus), validate_features=115)


def test_drop_leaky_duplicates(synthetic_corpus):
    _, corpus = synthetic_corpus
    frame = build_frame(discover_files(corpus), validate_features=4)
    doubled = pd.concat([frame, frame], ignore_index=True)
    deduped = drop_leaky_duplicates(doubled)
    assert len(deduped) == len(frame)


def test_load_corpus_caches_then_detects_staleness(synthetic_corpus):
    cfg, corpus = synthetic_corpus
    frame = load_corpus(cfg, profile="demo")
    parquet, manifest = cache_paths(cfg, "demo")
    assert parquet.is_file() and manifest.is_file()

    again = load_corpus(cfg, profile="demo")  # cache hit path
    pd.testing.assert_frame_equal(frame, again)

    time.sleep(0.01)
    with (corpus / "1.benign.csv").open("a") as fh:  # append -> new mtime/size
        fh.write("\n")
    assert source_fingerprint(discover_files(corpus)) != manifest.read_text().split('"fingerprint": "')[1][:16]
    rebuilt = load_corpus(cfg, profile="demo")
    assert len(rebuilt) == len(frame)


def test_load_corpus_reports_missing_dataset(synthetic_corpus):
    cfg, _ = synthetic_corpus
    with pytest.raises(FileNotFoundError, match="dataset root not found"):
        load_corpus(cfg, profile="missing")


def test_unknown_profile_names_the_alternatives(synthetic_corpus):
    cfg, _ = synthetic_corpus
    with pytest.raises(KeyError, match="demo"):
        load_corpus(cfg, profile="does-not-exist")
