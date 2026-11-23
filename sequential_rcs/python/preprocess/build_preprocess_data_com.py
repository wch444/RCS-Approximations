#!/usr/bin/env python3
"""Build ``preprocess_data_com.h5`` from the raw Expedia training CSV.

This script makes the historical upstream filtering recorded in
``preprocessing_all_data_com.ipynb`` executable and reproducible.  It preserves
the original row order and writes the pandas fixed-format HDF layout expected
by the downstream sequential-RCS preprocessing pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


HDF_KEY = "data"
MIN_PRICE_EXCLUSIVE = 10.0
MAX_PRICE_EXCLUSIVE = 5000.0
MAX_LENGTH_OF_STAY = 10.0

OUTPUT_COLUMNS = (
    "srch_id",
    "prop_id",
    "site_id",
    "position",
    "date_time",
    "booking_bool",
    "random_bool",
    "click_bool",
    "price_usd",
    "prop_starrating",
    "prop_location_score1",
    "prop_review_score",
    "srch_length_of_stay",
    "prop_log_historical_price",
    "promotion_flag",
    "srch_booking_window",
    "srch_adults_count",
    "prop_brand_bool",
    "srch_children_count",
    "srch_saturday_night_bool",
)
REQUIRED_SOURCE_COLUMNS = (*OUTPUT_COLUMNS, "srch_room_count")


def _require_columns(columns: Iterable[str]) -> None:
    available = set(columns)
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS).difference(available))
    if missing:
        raise ValueError(f"Raw Expedia CSV is missing columns: {missing}.")


def preprocess_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the historical Expedia row filters and retained-column contract."""

    _require_columns(raw.columns)
    eligible = (
        raw["price_usd"].gt(MIN_PRICE_EXCLUSIVE)
        & raw["price_usd"].lt(MAX_PRICE_EXCLUSIVE)
        & raw["srch_room_count"].eq(1)
        & raw["srch_length_of_stay"].le(MAX_LENGTH_OF_STAY)
    )
    processed = raw.loc[eligible, list(OUTPUT_COLUMNS)].dropna().copy()
    if processed.empty:
        raise ValueError("No Expedia rows remain after preprocessing.")
    return processed


def build_preprocess_h5(
    input_csv: Path,
    output_h5: Path,
    *,
    key: str = HDF_KEY,
    overwrite: bool = False,
) -> dict[str, object]:
    """Read the raw Expedia CSV and atomically write the filtered HDF file."""

    input_csv = Path(input_csv).expanduser().resolve()
    output_h5 = Path(output_h5).expanduser().resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    if input_csv == output_h5:
        raise ValueError("Input CSV and output HDF paths must be different.")
    if output_h5.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_h5}. "
            "Pass --overwrite explicitly to replace it."
        )
    if not key or "/" in key.strip("/"):
        raise ValueError("The HDF key must be one nonempty path component.")

    raw = pd.read_csv(input_csv)
    processed = preprocess_frame(raw)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_h5.with_name(f".{output_h5.name}.tmp")
    try:
        processed.to_hdf(temporary, key=key, mode="w", format="fixed")
        temporary.replace(output_h5)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "input_file": input_csv.name,
        "output_file": output_h5.name,
        "hdf_key": key,
        "input_rows": int(raw.shape[0]),
        "output_rows": int(processed.shape[0]),
        "output_columns": list(OUTPUT_COLUMNS),
        "price_filter": f"{MIN_PRICE_EXCLUSIVE:g} < price_usd < {MAX_PRICE_EXCLUSIVE:g}",
        "room_filter": "srch_room_count == 1",
        "length_of_stay_filter": (
            f"srch_length_of_stay <= {MAX_LENGTH_OF_STAY:g}"
        ),
        "missing_value_filter": "drop rows missing any retained output column",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv", required=True, type=Path, help="Raw Expedia train.csv."
    )
    parser.add_argument(
        "--output-h5",
        required=True,
        type=Path,
        help="Destination preprocess_data_com.h5.",
    )
    parser.add_argument("--hdf-key", default=HDF_KEY)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing output HDF file.",
    )
    args = parser.parse_args()
    summary = build_preprocess_h5(
        args.input_csv,
        args.output_h5,
        key=args.hdf_key,
        overwrite=args.overwrite,
    )
    print(
        f"wrote {args.output_h5} with {summary['output_rows']} of "
        f"{summary['input_rows']} input rows under key {summary['hdf_key']!r}"
    )


if __name__ == "__main__":
    main()
