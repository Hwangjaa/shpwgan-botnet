"""N-BaIoT ingestion: file discovery, labelled loading, parquet caching, subsets, splits."""

from __future__ import annotations

from .loader import (
    DatasetFile,
    build_frame,
    corpus_summary,
    discover_files,
    drop_leaky_duplicates,
    iter_feature_columns,
    load_corpus,
    parse_stem,
    read_dataset_file,
    source_fingerprint,
)
from .splits import leave_one_attack_out, stratified_split
from .subsets import (
    SubsetSpec,
    build_subset,
    load_subset,
    sample_strata,
    save_subset,
    subset_spec,
    subset_summary,
)

__all__ = [
    "DatasetFile",
    "SubsetSpec",
    "build_frame",
    "build_subset",
    "corpus_summary",
    "discover_files",
    "drop_leaky_duplicates",
    "iter_feature_columns",
    "leave_one_attack_out",
    "load_corpus",
    "load_subset",
    "parse_stem",
    "read_dataset_file",
    "sample_strata",
    "save_subset",
    "source_fingerprint",
    "stratified_split",
    "subset_spec",
    "subset_summary",
]
