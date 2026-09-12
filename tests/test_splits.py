"""Split behaviour, including the rare-attack case that breaks naive stratified splits."""

from __future__ import annotations

import pytest

from shapwgan_ids.data.loader import build_frame, discover_files
from shapwgan_ids.data.splits import leave_one_attack_out, stratified_split


@pytest.fixture
def frame(synthetic_corpus):
    _, corpus = synthetic_corpus
    return build_frame(discover_files(corpus), validate_features=4)


def test_stratified_split_shapes_and_disjointness(frame):
    parts = stratified_split(frame, test_size=0.2, val_size=0.1, seed=1)
    total = sum(len(p) for p in parts.values())
    assert total == len(frame)
    assert len(parts["test"]) == pytest.approx(0.2 * len(frame), rel=0.05)
    idx = [set(p.index) for p in parts.values()]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])


def test_stratified_split_keeps_every_attack_in_train(frame):
    parts = stratified_split(frame, test_size=0.2, val_size=0.1, seed=1)
    assert set(parts["train"]["attack"]) == set(frame["attack"])


def test_rare_attack_does_not_crash_the_split(frame):
    # '2.mirai.udp' has exactly 2 rows in the fixture -> merged into a coarser bucket
    parts = stratified_split(frame, test_size=0.2, val_size=0.1, seed=1)
    assert len(parts["test"]) > 0 and len(parts["train"]) > 0


def test_invalid_sizes_are_rejected(frame):
    with pytest.raises(ValueError):
        stratified_split(frame, test_size=1.5)
    with pytest.raises(ValueError):
        stratified_split(frame, val_size=-0.1)


def test_leave_one_attack_out_moves_held_out_attack_to_test(frame):
    train, test = leave_one_attack_out(frame, ["mirai.syn"], normal_share=0.2, seed=1)
    assert "mirai.syn" not in set(train["attack"])
    assert set(test["attack"]) >= {"mirai.syn"}
    assert (train["label"] == 1).sum() == 120 + 2  # gafgyt.combo + the rare mirari.udp rows
    assert test["label"].isin([0, 1]).all()


def test_leave_one_attack_out_rejects_unknown_attack(frame):
    with pytest.raises(ValueError):
        leave_one_attack_out(frame, ["not-an-attack"])
