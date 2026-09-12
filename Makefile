.PHONY: help sync lint fmt test test-fast info prepare-data prepare-full subset-laptop subset-smoke subset-report bench-laptop clean

UV ?= uv
PY ?= $(UV) run python

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

sync:  ## create/refresh .venv from pyproject.toml + uv.lock
	$(UV) sync

lint:  ## ruff check + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:  ## autoformat and autofix
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test:  ## full unit test suite
	$(UV) run pytest -q

test-fast:  ## stop at first failure, no coverage
	$(UV) run pytest -x -q --no-header

info:  ## environment + GPU + dataset report
	$(UV) run swg info

prepare-data:  ## build parquet cache (sampled profile)
	$(UV) run swg prepare-data --profile sampled

prepare-full:  ## build parquet cache for all 9 devices
	$(UV) run swg prepare-data --profile full

subset-laptop:  ## materialise the capped "laptop" subset (train/val/test parquet)
	$(UV) run swg make-subset --subset laptop

subset-smoke:  ## materialise the tiny "smoke" subset (seconds, not minutes)
	$(UV) run swg make-subset --subset smoke

bench-laptop:  ## capacity check: XGBoost fit + 1D-CNN epoch timings on the laptop subset
	$(UV) run python scripts/04_bench_subset.py --subset laptop

subset-report:  ## re-read the laptop subset from disk and verify against its manifest
	$(UV) run swg subset-report --subset laptop
	$(UV) run python scripts/05_inspect_subset.py data/processed/laptop

clean:  ## remove caches and experiment outputs (keeps data caches)
	rm -rf .pytest_cache .ruff_cache artifacts/logs
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
