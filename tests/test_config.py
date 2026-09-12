"""Config layering: defaults < yaml files < user file < env / .env."""

from __future__ import annotations

import pytest
import yaml

from shapwgan_ids.config import Config, deep_merge, env_overrides, load_config, read_env_file


def test_deep_merge_is_recursive_and_override_wins():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "keep": {"only": True}}
    override = {"nested": {"y": 9}, "a": 2}
    merged = deep_merge(base, override)
    assert merged == {"a": 2, "nested": {"x": 1, "y": 9}, "keep": {"only": True}}
    assert base["nested"]["y"] == 2, "inputs must not be mutated"


def test_env_overrides_nested_paths_and_types():
    over = env_overrides({"SHAPWGAN_SEED": "11", "SHAPWGAN_DATASET__RAW_DIR": "/data/x", "OTHER": "1"})
    assert over == {"seed": 11, "dataset": {"raw_dir": "/data/x"}}


def test_dotted_access_and_require():
    cfg = Config({"a": {"b": {"c": 5}}})
    assert cfg.get_path("a.b.c") == 5
    assert cfg.get_path("a.b.z", "fallback") == "fallback"
    cfg.set_path("a.b.d", 7)
    assert cfg["a"]["b"]["d"] == 7
    assert cfg.require("a.b.c") == 5
    with pytest.raises(KeyError):
        cfg.require("a.nope")


def test_read_env_file_handles_comments_and_quotes(tmp_path):
    path = tmp_path / ".env"
    path.write_text('# comment\nKEY=value\nQUOTED="a b"\nEMPTY=\n')
    assert read_env_file(path) == {"KEY": "value", "QUOTED": "a b", "EMPTY": ""}


def test_load_config_layer_precedence(tmp_path):
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    (cfgdir / "default.yaml").write_text(
        yaml.safe_dump({"seed": 1, "device": "auto", "paths": {"artifacts_dir": "artifacts"}})
    )
    (cfgdir / "data.yaml").write_text(yaml.safe_dump({"dataset": {"raw_dir": "/base"}, "seed": 2}))
    user = tmp_path / "user.yaml"
    user.write_text(yaml.safe_dump({"device": "cpu"}))
    dotenv = tmp_path / ".env"
    dotenv.write_text("SHAPWGAN_SEED=99\n")

    cfg = load_config(
        cfgdir,
        files=("default.yaml", "data.yaml", "wgan.yaml"),
        user_config=user,
        env={"SHAPWGAN_DATASET__RAW_DIR": "/from-env"},
        dotenv=dotenv,
    )
    assert cfg.get_path("seed") == 99, "env beats yaml"
    assert cfg.get_path("device") == "cpu", "user config beats yaml, env untouched"
    assert cfg.get_path("dataset.raw_dir") == "/from-env"
    assert cfg.get_path("paths.artifacts_dir") == "artifacts"
    assert isinstance(cfg, dict)


def test_missing_layers_are_skipped(tmp_path):
    cfg = load_config(tmp_path / "does-not-exist", files=("default.yaml",), env={})
    assert cfg == {}
    assert cfg.get_path("anything", None) is None
