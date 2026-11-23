#!/usr/bin/env python3
"""Run the project-local Expedia preprocessing protocol for sequential-RCS.

The implementation lives in this package so changes to the separate SMC
preprocessing workflow cannot alter the frozen sequential-RCS data contract.
This canonical entry point can be executed from any working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:  # Package import used by tests and ``python -m``.
    from . import implementation as _implementation
except ImportError:  # Direct execution via ``python path/to/pipeline.py``.
    from sequential_rcs.python.preprocess import implementation as _implementation


BASE_COLUMNS = _implementation.BASE_COLUMNS
MANIFEST_NAME = _implementation.MANIFEST_NAME
MATRIX_COLUMN_COUNT = _implementation.MATRIX_COLUMN_COUNT
MODEL_FEATURE_COLUMNS = _implementation.MODEL_FEATURE_COLUMNS
PROTOCOL = _implementation.PROTOCOL
RAW_COLUMNS = _implementation.RAW_COLUMNS
SITES = _implementation.SITES
STANDARDIZED_COLUMNS = _implementation.STANDARDIZED_COLUMNS
TRIALS = _implementation.TRIALS
add_outside_rows = _implementation.add_outside_rows
build_from_h5 = _implementation.build_from_h5
build_trial = _implementation.build_trial
load_h5 = _implementation.load_h5
main = _implementation.main
prepare_site = _implementation.prepare_site
split_query_ids = _implementation.split_query_ids
standardize_split = _implementation.standardize_split

__all__ = [
    "BASE_COLUMNS",
    "MANIFEST_NAME",
    "MATRIX_COLUMN_COUNT",
    "MODEL_FEATURE_COLUMNS",
    "PROTOCOL",
    "RAW_COLUMNS",
    "SITES",
    "STANDARDIZED_COLUMNS",
    "TRIALS",
    "add_outside_rows",
    "build_from_h5",
    "build_trial",
    "load_h5",
    "prepare_site",
    "split_query_ids",
    "standardize_split",
]


if __name__ == "__main__":
    main()
