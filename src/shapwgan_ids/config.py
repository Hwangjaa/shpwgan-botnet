"""Layered YAML config: configs/*.yaml -> optional .env overrides -> dict access.

Precedence (lowest to highest): default.yaml, data.yaml, shap.yaml, wgan.yaml,
ids.yaml, loop.yaml, an optional user config file, then environment variables
prefixed ``SHAPWGAN_`` where ``__`` walks into nested keys, e.g.::

    SHAPWGAN_SEED=7                 -> cfg["seed"] = 7
    SHAPWGAN_DATASET__RAW_DIR=/x    -> cfg["dataset"]["raw_dir"] = "/x"

Values are parsed as YAML scalars, so ints/bools/lists survive the round trip.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

ENV_PREFIX = "SHAPWGAN_"
DEFAULT_CONFIG_FILES: tuple[str, ...] = (
    "default.yaml",
    "data.yaml",
    "shap.yaml",
    "wgan.yaml",
    "ids.yaml",
    "loop.yaml",
)


class Config(dict):
    """Nested dict with dotted lookup and a frozen snapshot helper."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get_path(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"missing required config key: {dotted}")
        return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins, dicts are joined)."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def read_env_file(path: Path) -> dict[str, str]:
    """Minimal .env reader (KEY=VALUE, ``#`` comments, optional quotes). No deps."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        values[key.strip()] = val
    return values


def _coerce(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def env_overrides(env: Mapping[str, str], prefix: str = ENV_PREFIX) -> dict:
    """Turn ``SHAPWGAN_A__B=1`` style vars into a nested dict (ignores prefix-only)."""
    out: dict[str, Any] = {}
    for key, value in env.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower()
        if not path:
            continue
        node = out
        parts = path.split("__")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return out


def load_config(
    config_dir: Path,
    files: Iterable[str] = DEFAULT_CONFIG_FILES,
    user_config: Path | None = None,
    env: Mapping[str, str] | None = None,
    dotenv: Path | None = None,
) -> Config:
    """Load and merge the config layers; missing layers are skipped, not fatal."""
    merged: dict[str, Any] = {}
    for name in files:
        path = Path(config_dir) / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"{path} must contain a YAML mapping at the top level")
        merged = deep_merge(merged, loaded)

    if user_config is not None and Path(user_config).is_file():
        with Path(user_config).open(encoding="utf-8") as fh:
            merged = deep_merge(merged, yaml.safe_load(fh) or {})

    env_source: dict[str, str] = {}
    if dotenv is not None:
        env_source.update(read_env_file(Path(dotenv)))
    env_source.update(dict(os.environ if env is None else env))
    merged = deep_merge(merged, env_overrides(env_source))

    return Config(merged)
