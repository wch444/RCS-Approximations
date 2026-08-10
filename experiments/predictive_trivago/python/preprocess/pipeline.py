#!/usr/bin/env python3
"""Run the four deterministic Trivago preprocessing stages in order."""

from __future__ import annotations

import argparse
from pathlib import Path

from .build_item_features import build as build_item_features
from .build_item_prices import build as build_item_prices
from .build_mat_session_p2000 import build as build_matrices
from .build_queries_session import build as build_queries


def require_input(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def run(raw_dir: Path, output_dir: Path) -> None:
    train = raw_dir / "train.csv"
    metadata = raw_dir / "item_metadata.csv"
    require_input(train)
    require_input(metadata)
    prep = output_dir / "_prep"
    prep.mkdir(parents=True, exist_ok=True)
    prices = prep / "item_mean_price.parquet"
    features = prep / "item_features.parquet"
    build_item_prices(train, prices)
    build_item_features(metadata, features)
    build_queries(train, prices, prep, 500_000)
    build_matrices(prep, output_dir, 2000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.raw_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
