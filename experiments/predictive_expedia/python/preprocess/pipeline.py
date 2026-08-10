#!/usr/bin/env python3
"""Build the deterministic Expedia 10,000-query MATLAB samples.

The implementation preserves the historical matrix layout and split-specific
standardization.  The one correction is deliberate: searches with more than
four clicked hotels are removed as complete queries before query sampling and
before position-bin indicators are constructed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import tables
from scipy.io import savemat


SITES = (5, 14, 15, 24, 32)
TRIALS = tuple(range(1, 11))
N_SAMPLE = 10_000
TRAIN_FRACTION = 0.60
MAX_QUERY_CLICKS = 4
POSITION_BIN_WIDTH = 5
HDF_CHUNK_ROWS = 250_000

RAW_COLUMNS = (
    "srch_id",
    "prop_id",
    "site_id",
    "position",
    "booking_bool",
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
    "srch_children_count",
    "prop_brand_bool",
    "srch_saturday_night_bool",
)

BASE_COLUMNS = (
    "srch_id",
    "prop_id",
    "site_id",
    "position",
    "booking_bool",
    "click_bool",
    "origin_price",
    "b_0",
    "price_usd",
    "prop_starrating",
    "prop_location_score1",
    "prop_review_score",
    "srch_length_of_stay",
    "prop_log_historical_price",
    "promotion_flag",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "prop_brand_bool",
    "srch_saturday_night_bool",
)

STANDARDIZED_COLUMNS = BASE_COLUMNS[8:]
MANIFEST_NAME = "expedia_preprocess_manifest.json"


def _canonical_subset(
    values: Iterable[int], allowed: Sequence[int], label: str
) -> tuple[int, ...]:
    """Validate and return a unique subset in the requested order."""

    selected = tuple(values)
    if not selected:
        raise ValueError(f"At least one {label} must be selected.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate {label} values are not allowed: {selected}.")
    unknown = sorted(set(selected).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {label} values: {unknown}.")
    return selected


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(set(RAW_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Expedia HDF5 data are missing columns: {missing}.")


def _decode_hdf_label(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_fixed_hdf_columns(path: Path, key: str, sites: Sequence[int]) -> pd.DataFrame:
    """Read only required numeric columns from the historical pandas HDF5.

    The supplied Expedia file uses pandas' fixed-frame layout.  Reading its
    numeric blocks in chunks avoids materializing the unused nine-million-row
    ``date_time`` object block.
    """

    node_path = "/" + key.strip("/")
    with tables.open_file(path, mode="r") as store:
        try:
            group = store.get_node(node_path)
        except tables.NoSuchNodeError as error:
            raise KeyError(f"HDF5 key {key!r} does not exist in {path}.") from error
        if not isinstance(group, tables.Group) or "axis1" not in group._v_children:
            raise ValueError(
                f"HDF5 key {key!r} is not a pandas fixed-format DataFrame."
            )

        block_indices = sorted(
            int(name.removeprefix("block").removesuffix("_items"))
            for name in group._v_children
            if name.startswith("block") and name.endswith("_items")
        )
        blocks: list[tuple[tables.Leaf, dict[str, int]]] = []
        available: set[str] = set()
        for block_index in block_indices:
            items_node = group._f_get_child(f"block{block_index}_items")
            values_node = group._f_get_child(f"block{block_index}_values")
            labels = [_decode_hdf_label(value) for value in items_node.read()]
            selected = {
                label: index
                for index, label in enumerate(labels)
                if label in RAW_COLUMNS
            }
            if selected:
                numeric_matrix = (
                    isinstance(values_node, tables.Leaf) and len(values_node.shape) == 2
                )
                if not numeric_matrix:
                    raise ValueError(
                        f"Required HDF5 block {block_index} is not a numeric matrix."
                    )
                blocks.append((values_node, selected))
                available.update(selected)

        missing = sorted(set(RAW_COLUMNS).difference(available))
        if missing:
            raise ValueError(f"Expedia HDF5 data are missing columns: {missing}.")
        row_count = int(group._f_get_child("axis1").shape[0])
        chunks: list[pd.DataFrame] = []
        for start in range(0, row_count, HDF_CHUNK_ROWS):
            stop = min(start + HDF_CHUNK_ROWS, row_count)
            arrays: dict[str, np.ndarray] = {}
            for values_node, selected in blocks:
                values = values_node[start:stop]
                for label, column_index in selected.items():
                    arrays[label] = values[:, column_index]
            site_mask = np.isin(arrays["site_id"], sites)
            if site_mask.any():
                chunks.append(
                    pd.DataFrame(
                        {column: arrays[column][site_mask] for column in RAW_COLUMNS}
                    )
                )
    if not chunks:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.concat(chunks, ignore_index=True)


def load_h5(path: Path, key: str, sites: Sequence[int]) -> pd.DataFrame:
    """Load the historical fixed-format HDF5 frame and keep relevant sites."""

    if not path.is_file():
        raise FileNotFoundError(path)
    frame = _read_fixed_hdf_columns(path, key, sites)
    _require_columns(frame)
    missing_sites = sorted(set(sites).difference(frame["site_id"].unique()))
    if missing_sites:
        raise ValueError(
            f"The Expedia HDF5 file contains no rows for sites {missing_sites}."
        )
    return frame


def _validate_binary(frame: pd.DataFrame, column: str) -> None:
    values = frame[column].dropna().unique()
    if not set(values).issubset({0, 1, False, True}):
        raise ValueError(
            f"{column} must be binary, but found {sorted(values.tolist())}."
        )


def prepare_site(raw: pd.DataFrame, site: int) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Filter one site and construct its fixed matrix-column contract.

    Query-level click filtering precedes both candidate-id construction and
    position-bin indicators.  This ordering prevents the historical outer
    concatenation from introducing a ``NaN`` search-id sentinel.
    """

    _require_columns(raw)
    frame = raw.loc[raw["site_id"] == site, list(RAW_COLUMNS)].copy()
    if frame.empty:
        raise ValueError(f"No Expedia rows are available for site {site}.")
    if frame["srch_id"].isna().any():
        raise ValueError(f"Site {site} contains missing srch_id values.")
    _validate_binary(frame, "booking_bool")
    _validate_binary(frame, "click_bool")

    click_count = frame.groupby("srch_id", sort=False)["click_bool"].transform("sum")
    frame = frame.loc[click_count <= MAX_QUERY_CLICKS].copy()
    if frame.empty:
        raise ValueError(f"No site-{site} query remains after click-count filtering.")

    # Historical build_map sorted each query by displayed position, then
    # mapped its distinct observed positions to 0, 1, ... .  Dense rank after
    # the same stable sort is the vectorized equivalent.
    frame.sort_values(["srch_id", "position"], kind="mergesort", inplace=True)
    frame["position"] = (
        frame.groupby("srch_id", sort=False)["position"]
        .rank(method="dense", ascending=True)
        .astype(np.int64)
        - 1
    )

    frame["origin_price"] = frame["price_usd"].to_numpy(dtype=np.float64)
    frame["b_0"] = 1.0

    position_bin = np.floor_divide(
        frame["position"].to_numpy(dtype=np.int64), POSITION_BIN_WIDTH
    )
    observed_bins = tuple(int(value) for value in np.unique(position_bin))
    if not observed_bins or observed_bins != tuple(range(observed_bins[-1] + 1)):
        raise ValueError(
            f"Site {site} has a noncontiguous position-bin contract: {observed_bins}."
        )
    bin_columns = tuple(f"position_bin_{value}" for value in observed_bins)
    for value, column in zip(observed_bins, bin_columns):
        frame[column] = (position_bin == value).astype(np.float64)

    matrix_columns = (*BASE_COLUMNS, *bin_columns)
    frame = frame.loc[:, list(matrix_columns)].reset_index(drop=True)
    if frame["srch_id"].isna().any():
        raise AssertionError(
            "Filtering and dummy construction introduced a missing srch_id."
        )
    return frame, matrix_columns


def split_query_ids(
    frame: pd.DataFrame,
    seed: int,
    sample_size: int = N_SAMPLE,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Shuffle valid query ids with the historical Python RNG convention."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between zero and one.")
    query_ids = list(frame["srch_id"].unique())
    if any(pd.isna(query_id) for query_id in query_ids):
        raise ValueError("Candidate srch_id values must not contain NaN.")
    random.Random(seed).shuffle(query_ids)
    query_ids = query_ids[: min(sample_size, len(query_ids))]
    cut = round(len(query_ids) * train_fraction)
    return tuple(query_ids[:cut]), tuple(query_ids[cut:])


def standardize_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the historical split-specific sample-SD standardization."""

    result = frame.copy()
    for column in STANDARDIZED_COLUMNS:
        # Keep the original pandas reduction order and ``var() ** 0.5``
        # expression, including pandas' default sample variance (ddof=1).
        mean = float(result[column].mean())
        std = float(result[column].var(ddof=1) ** 0.5)
        if not np.isfinite(mean) or not np.isfinite(std) or std == 0:
            raise ValueError(
                f"Cannot apply historical sample-SD standardization to {column!r}."
            )
        result[column] = (result[column] - mean) / std
    return result


def add_outside_rows(
    frame: pd.DataFrame, matrix_columns: Sequence[str]
) -> pd.DataFrame:
    """Append one outside option per query using the historical row encoding."""

    if frame.empty:
        raise ValueError("Cannot add outside rows to an empty split.")
    query_ids = frame["srch_id"].drop_duplicates().to_numpy()
    booking_count = frame.groupby("srch_id", sort=False)["booking_bool"].sum()
    if ((booking_count < 0) | (booking_count > 1)).any():
        raise ValueError("Each Expedia query must contain at most one booking.")

    outside = pd.DataFrame(0.0, index=range(len(query_ids)), columns=matrix_columns)
    outside["srch_id"] = query_ids
    outside["position"] = 100.0
    outside["booking_bool"] = (
        outside["srch_id"].map(booking_count).fillna(0).to_numpy() < 1
    ).astype(np.float64)

    result = pd.concat([frame.loc[:, matrix_columns], outside], ignore_index=True)
    return result.sort_values(
        ["srch_id", "position"], kind="mergesort", ignore_index=True
    )


def build_trial(
    site_frame: pd.DataFrame,
    matrix_columns: Sequence[str],
    trial: int,
    sample_size: int = N_SAMPLE,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build the train/test matrices for one deterministic trial."""

    train_ids, test_ids = split_query_ids(site_frame, trial, sample_size=sample_size)
    train_id_set, test_id_set = set(train_ids), set(test_ids)
    train = site_frame.loc[site_frame["srch_id"].isin(train_id_set)].copy()
    test = site_frame.loc[site_frame["srch_id"].isin(test_id_set)].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Trial {trial} must contain nonempty train and test query sets."
        )

    # This intentionally preserves the paper's historical convention: each
    # split estimates and applies its own sample mean and sample standard
    # deviation rather than sharing training-set moments.
    train = add_outside_rows(standardize_split(train), matrix_columns)
    test = add_outside_rows(standardize_split(test), matrix_columns)
    train_matrix = train.to_numpy(dtype=np.float64, copy=True)
    test_matrix = test.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(train_matrix).all() or not np.isfinite(test_matrix).all():
        raise ValueError(f"Trial {trial} produced nonfinite matrix entries.")

    metadata = {
        "train_queries": len(train_ids),
        "test_queries": len(test_ids),
        "train_rows": int(train_matrix.shape[0]),
        "test_rows": int(test_matrix.shape[0]),
        "columns": int(train_matrix.shape[1]),
    }
    return train_matrix, test_matrix, metadata


def _preflight_outputs(
    output_dir: Path, sites: Sequence[int], trials: Sequence[int], overwrite: bool
) -> tuple[list[Path], Path]:
    outputs = [
        output_dir / f"10k_sample_{site}_{trial}.mat"
        for site in sites
        for trial in trials
    ]
    manifest = output_dir / MANIFEST_NAME
    existing = [path for path in (*outputs, manifest) if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(path.name for path in existing[:5])
        suffix = " ..." if len(existing) > 5 else ""
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing output(s): "
            f"{preview}{suffix}. Pass --overwrite explicitly to replace them."
        )
    return outputs, manifest


def _write_mat(path: Path, train_matrix: np.ndarray, test_matrix: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        savemat(
            temporary,
            {"data_train": train_matrix, "data_test": test_matrix},
            appendmat=False,
            do_compression=False,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_from_h5(
    h5_path: Path,
    output_dir: Path,
    *,
    key: str = "data",
    sites: Iterable[int] = SITES,
    trials: Iterable[int] = TRIALS,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build selected Expedia site/trial MAT files from one explicit HDF5 input."""

    selected_sites = _canonical_subset(sites, SITES, "site")
    selected_trials = _canonical_subset(trials, TRIALS, "trial")
    h5_path = Path(h5_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    _, manifest_path = _preflight_outputs(
        output_dir, selected_sites, selected_trials, overwrite
    )

    frame = load_h5(h5_path, key, selected_sites)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for site in selected_sites:
        site_frame, matrix_columns = prepare_site(frame, site)
        for trial in selected_trials:
            train_matrix, test_matrix, metadata = build_trial(
                site_frame, matrix_columns, trial
            )
            output_path = output_dir / f"10k_sample_{site}_{trial}.mat"
            _write_mat(output_path, train_matrix, test_matrix)
            records.append(
                {
                    "site": site,
                    "trial": trial,
                    "file": output_path.name,
                    **metadata,
                }
            )
            print(
                f"wrote {output_path} "
                f"({metadata['train_queries']}/{metadata['test_queries']} queries)"
            )
            del train_matrix, test_matrix
        del site_frame

    manifest: dict[str, object] = {
        "input_file": h5_path.name,
        "hdf_key": key,
        "sites": list(selected_sites),
        "trials": list(selected_trials),
        "query_sample_size": N_SAMPLE,
        "train_fraction": TRAIN_FRACTION,
        "max_query_clicks": MAX_QUERY_CLICKS,
        "split_seed": "trial",
        "standardization": "separate train/test sample SD (ddof=1)",
        "files": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path, help="Input HDF5 file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Explicit staging/output directory for generated MAT files.",
    )
    parser.add_argument("--hdf-key", default="data")
    parser.add_argument("--sites", nargs="+", type=int, default=list(SITES))
    parser.add_argument("--trials", nargs="+", type=int, default=list(TRIALS))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace selected MAT files and the manifest if present.",
    )
    args = parser.parse_args()
    build_from_h5(
        args.h5,
        args.output_dir,
        key=args.hdf_key,
        sites=args.sites,
        trials=args.trials,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
