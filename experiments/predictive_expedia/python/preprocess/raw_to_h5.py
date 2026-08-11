#!/usr/bin/env python3
"""Build the historical Expedia HDF5 input from Kaggle ``train.csv``.

The row filters and output-column order are the ones recovered from the
original preprocessing notebook.  The resulting logical data frame is the
input to :mod:`experiments.predictive_expedia.python.preprocess.pipeline`.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import tables


DEFAULT_CHUNK_ROWS = 250_000
FILTER_COLUMNS = (
    "price_usd",
    "srch_room_count",
    "srch_length_of_stay",
)
HDF_COLUMNS = (
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
READ_COLUMNS = tuple(dict.fromkeys((*HDF_COLUMNS, *FILTER_COLUMNS)))


def build_h5_from_csv(
    csv_path: Path,
    output_path: Path,
    *,
    hdf_key: str = "data",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    overwrite: bool = False,
) -> int:
    """Filter the Kaggle CSV and publish one fixed-format pandas HDF5 file."""

    csv_path = Path(csv_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if csv_path == output_path:
        raise ValueError("The Expedia CSV input and HDF5 output must differ.")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive.")
    if not hdf_key.strip("/"):
        raise ValueError("hdf_key must name a nonempty HDF5 key.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace existing HDF5 file without --overwrite: "
            f"{output_path}"
        )

    header = pd.read_csv(csv_path, nrows=0)
    missing = sorted(set(READ_COLUMNS).difference(header.columns))
    if missing:
        raise ValueError(f"Expedia train.csv is missing columns: {missing}.")

    selected_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=READ_COLUMNS,
        chunksize=chunk_rows,
        low_memory=False,
    ):
        keep = (
            chunk["price_usd"].gt(10)
            & chunk["price_usd"].lt(5000)
            & chunk["srch_room_count"].eq(1)
            & chunk["srch_length_of_stay"].le(10)
        )
        selected = chunk.loc[keep, list(HDF_COLUMNS)].dropna().copy()
        if not selected.empty:
            selected_chunks.append(selected)

    if not selected_chunks:
        raise ValueError("No Expedia rows remain after the historical filters.")
    # Keep the original CSV row index, as the historical notebook did when it
    # selected and filtered the full frame.
    frame = pd.concat(selected_chunks, copy=False)
    del selected_chunks
    frame = frame.loc[:, list(HDF_COLUMNS)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".expedia-h5-", dir=output_path.parent
    ) as stage_dir:
        stage_path = Path(stage_dir) / output_path.name
        frame.to_hdf(stage_path, key=hdf_key.strip("/"), mode="w", format="fixed")
        with tables.open_file(stage_path, mode="r") as store:
            group = store.get_node("/" + hdf_key.strip("/"))
            stored_rows = int(group._f_get_child("axis1").shape[0])
            stored_columns = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in group._f_get_child("axis0").read()
            )
        if stored_rows != len(frame) or stored_columns != HDF_COLUMNS:
            raise RuntimeError("Staged Expedia HDF5 file failed its write check.")
        os.replace(stage_path, output_path)

    return len(frame)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Kaggle train.csv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hdf-key", default="data")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Run the CSV-to-HDF5 build."""

    args = build_parser().parse_args()
    row_count = build_h5_from_csv(
        args.csv,
        args.output,
        hdf_key=args.hdf_key,
        chunk_rows=args.chunk_rows,
        overwrite=args.overwrite,
    )
    print(f"Wrote {row_count:,} rows to {args.output}.")


if __name__ == "__main__":
    main()
