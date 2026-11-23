#!/usr/bin/env python3
"""Build deterministic Expedia samples for the four-model comparison.

The maintained protocol retains every otherwise valid query, irrespective of
its click count, and writes the fixed legacy 20-column matrix schema without
position-bin utility features.  Displayed position and click remain available
as raw columns for stage construction, ranking, tie-breaking, and estimation
of the NS-MNL stopping distribution.  Training and test covariates are
standardized separately, preserving the convention requested for this study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import tables
from scipy.io import savemat, whosmat


SITES = (5, 14, 15, 24, 32)
TRIALS = tuple(range(1, 11))
N_SAMPLE = 10_000
TRAIN_FRACTION = 0.60
HDF_CHUNK_ROWS = 250_000
MATRIX_COLUMN_COUNT = 20
PROTOCOL = "expedia_no_click_filter_no_position_v1"

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
MODEL_FEATURE_COLUMNS = BASE_COLUMNS[7:]
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


def _require_finite_numeric(frame: pd.DataFrame, site: int) -> None:
    try:
        values = frame.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Site {site} contains nonnumeric required values.") from error
    if not np.isfinite(values).all():
        raise ValueError(f"Site {site} contains missing or nonfinite required values.")


def prepare_site(raw: pd.DataFrame, site: int) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Select one site and construct the fixed 20-column matrix contract."""

    _require_columns(raw)
    frame = raw.loc[raw["site_id"] == site, list(RAW_COLUMNS)].copy()
    if frame.empty:
        raise ValueError(f"No Expedia rows are available for site {site}.")
    if frame["srch_id"].isna().any():
        raise ValueError(f"Site {site} contains missing srch_id values.")
    _require_finite_numeric(frame, site)
    _validate_binary(frame, "booking_bool")
    _validate_binary(frame, "click_bool")
    if not np.equal(frame["srch_id"], np.floor(frame["srch_id"])).all():
        raise ValueError(f"Site {site} contains noninteger srch_id values.")
    booking_count = frame.groupby("srch_id", sort=False)["booking_bool"].sum()
    if (booking_count > 1).any():
        raise ValueError(f"Site {site} contains a query with multiple bookings.")

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
    if frame["position"].max() >= 100:
        raise ValueError(
            f"Site {site} has at least 100 displayed alternatives in one query; "
            "position 100 is reserved for the outside option."
        )

    matrix_columns = BASE_COLUMNS
    frame = frame.loc[:, list(matrix_columns)].reset_index(drop=True)
    if tuple(frame.columns) != BASE_COLUMNS or frame.shape[1] != MATRIX_COLUMN_COUNT:
        raise AssertionError("The Expedia matrix-column contract is not 20 columns.")
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
    if len(query_ids) < sample_size:
        raise ValueError(
            f"Need at least {sample_size} queries, but found only {len(query_ids)}."
        )
    random.Random(seed).shuffle(query_ids)
    query_ids = query_ids[:sample_size]
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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_integer_sequence(values: Sequence[object]) -> str:
    integers = np.asarray(values, dtype=np.int64)
    if not np.equal(integers, np.asarray(values)).all():
        raise ValueError("Query identifiers must be exactly representable as integers.")
    canonical = integers.astype("<i8", copy=False)
    return hashlib.sha256(memoryview(np.ascontiguousarray(canonical))).hexdigest()


def _sha256_matrix(matrix: np.ndarray) -> str:
    canonical = np.ascontiguousarray(matrix, dtype="<f8")
    return hashlib.sha256(memoryview(canonical)).hexdigest()


def _validate_output_matrix(
    matrix: np.ndarray,
    expected_query_ids: Sequence[object],
    matrix_columns: Sequence[str],
    label: str,
) -> dict[str, int]:
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.ndim != 2
        or matrix.shape[1] != MATRIX_COLUMN_COUNT
        or tuple(matrix_columns) != BASE_COLUMNS
        or not np.isrealobj(matrix)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(f"{label} is not a finite real 20-column Expedia matrix.")

    ids = matrix[:, 0]
    if not np.equal(ids, np.floor(ids)).all() or np.any(np.diff(ids) < 0):
        raise ValueError(f"{label} query identifiers must be sorted integers.")
    observed_ids = ids[np.r_[True, ids[1:] != ids[:-1]]]
    expected = np.sort(np.asarray(expected_query_ids, dtype=np.float64))
    if not np.array_equal(observed_ids, expected):
        raise ValueError(f"{label} query identifiers do not match the sampled split.")

    outside = matrix[:, 3] == 100
    outside_count = pd.Series(outside.astype(np.int64)).groupby(ids, sort=False).sum()
    if not (outside_count.to_numpy() == 1).all():
        raise ValueError(f"{label} must contain exactly one outside row per query.")
    if not (matrix[outside, 1] == 0).all() or not (matrix[outside, 5:] == 0).all():
        raise ValueError(f"{label} outside rows do not follow the zero-row contract.")
    if not set(np.unique(matrix[:, 4])).issubset({0.0, 1.0}):
        raise ValueError(f"{label} booking indicators must be binary.")
    chosen_count = pd.Series(matrix[:, 4]).groupby(ids, sort=False).sum()
    if not (chosen_count.to_numpy() == 1).all():
        raise ValueError(f"{label} must contain exactly one chosen outcome per query.")
    if not set(np.unique(matrix[:, 5])).issubset({0.0, 1.0}):
        raise ValueError(f"{label} click indicators must be binary.")

    product = ~outside
    if not (matrix[product, 7] == 1).all():
        raise ValueError(f"{label} product rows must have intercept b_0=1.")
    if (matrix[product, 3] < 0).any() or not np.equal(
        matrix[product, 3], np.floor(matrix[product, 3])
    ).all():
        raise ValueError(f"{label} product positions must be nonnegative integers.")
    for column_index, column in enumerate(STANDARDIZED_COLUMNS, start=8):
        values = matrix[product, column_index]
        if not np.isclose(values.mean(), 0.0, atol=1e-10, rtol=0.0) or not np.isclose(
            values.std(ddof=1), 1.0, atol=1e-10, rtol=0.0
        ):
            raise ValueError(
                f"{label} standardized product column {column!r} is not mean 0, "
                "sample SD 1."
            )

    return {
        "queries": int(observed_ids.size),
        "rows": int(matrix.shape[0]),
        "product_rows": int(product.sum()),
        "booked_queries": int(matrix[product, 4].sum()),
        "outside_chosen_queries": int(matrix[outside, 4].sum()),
        "clicked_product_rows": int(matrix[product, 5].sum()),
    }


def _click_count_audit(
    site_frame: pd.DataFrame, query_ids: Sequence[object]
) -> dict[str, int]:
    selected = site_frame.loc[site_frame["srch_id"].isin(set(query_ids))]
    click_counts = selected.groupby("srch_id", sort=False)["click_bool"].sum()
    if click_counts.size != len(query_ids):
        raise AssertionError("Click audit does not cover every sampled query.")
    return {
        "queries_click_count_eq_0": int((click_counts == 0).sum()),
        "queries_click_count_gt_4": int((click_counts > 4).sum()),
        "max_click_count": int(click_counts.max()),
    }


def build_trial(
    site_frame: pd.DataFrame,
    matrix_columns: Sequence[str],
    trial: int,
    sample_size: int = N_SAMPLE,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build the train/test matrices for one deterministic trial."""

    train_ids, test_ids = split_query_ids(site_frame, trial, sample_size=sample_size)
    train_id_set, test_id_set = set(train_ids), set(test_ids)
    if train_id_set.intersection(test_id_set):
        raise AssertionError(f"Trial {trial} has overlapping train and test queries.")
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

    train_audit = _validate_output_matrix(
        train_matrix, train_ids, matrix_columns, f"trial {trial} training matrix"
    )
    test_audit = _validate_output_matrix(
        test_matrix, test_ids, matrix_columns, f"trial {trial} test matrix"
    )
    train_click_audit = _click_count_audit(site_frame, train_ids)
    test_click_audit = _click_count_audit(site_frame, test_ids)
    metadata = {
        "train_queries": train_audit["queries"],
        "test_queries": test_audit["queries"],
        "train_rows": train_audit["rows"],
        "test_rows": test_audit["rows"],
        "train_product_rows": train_audit["product_rows"],
        "test_product_rows": test_audit["product_rows"],
        "train_booked_queries": train_audit["booked_queries"],
        "test_booked_queries": test_audit["booked_queries"],
        "train_outside_chosen_queries": train_audit["outside_chosen_queries"],
        "test_outside_chosen_queries": test_audit["outside_chosen_queries"],
        "train_clicked_product_rows": train_audit["clicked_product_rows"],
        "test_clicked_product_rows": test_audit["clicked_product_rows"],
        "train_queries_click_count_eq_0": train_click_audit["queries_click_count_eq_0"],
        "test_queries_click_count_eq_0": test_click_audit["queries_click_count_eq_0"],
        "train_queries_click_count_gt_4": train_click_audit["queries_click_count_gt_4"],
        "test_queries_click_count_gt_4": test_click_audit["queries_click_count_gt_4"],
        "train_max_click_count": train_click_audit["max_click_count"],
        "test_max_click_count": test_click_audit["max_click_count"],
        "columns": int(train_matrix.shape[1]),
        "train_query_ids_sha256": _sha256_integer_sequence(train_ids),
        "test_query_ids_sha256": _sha256_integer_sequence(test_ids),
        "data_train_sha256": _sha256_matrix(train_matrix),
        "data_test_sha256": _sha256_matrix(test_matrix),
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


def _validate_written_mat(
    path: Path, train_shape: tuple[int, int], test_shape: tuple[int, int]
) -> None:
    variables = {name: tuple(shape) for name, shape, _ in whosmat(path)}
    expected = {"data_train": train_shape, "data_test": test_shape}
    if variables != expected:
        raise ValueError(
            f"Written MAT header does not match the expected payload: {path}."
        )


def _site_source_audit(site_frame: pd.DataFrame, site: int) -> dict[str, int]:
    click_counts = site_frame.groupby("srch_id", sort=False)["click_bool"].sum()
    booking_counts = site_frame.groupby("srch_id", sort=False)["booking_bool"].sum()
    return {
        "site": int(site),
        "source_rows": int(site_frame.shape[0]),
        "source_queries": int(click_counts.size),
        "source_queries_click_count_eq_0": int((click_counts == 0).sum()),
        "source_queries_click_count_gt_4": int((click_counts > 4).sum()),
        "source_max_click_count": int(click_counts.max()),
        "source_booked_queries": int((booking_counts == 1).sum()),
        "source_outside_choice_queries": int((booking_counts == 0).sum()),
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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

    input_size_bytes = h5_path.stat().st_size
    input_sha256 = _sha256_path(h5_path)
    frame = load_h5(h5_path, key, selected_sites)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    source_records: list[dict[str, int]] = []
    for site in selected_sites:
        site_frame, matrix_columns = prepare_site(frame, site)
        source_records.append(_site_source_audit(site_frame, site))
        for trial in selected_trials:
            train_matrix, test_matrix, metadata = build_trial(
                site_frame, matrix_columns, trial
            )
            output_path = output_dir / f"10k_sample_{site}_{trial}.mat"
            _write_mat(output_path, train_matrix, test_matrix)
            _validate_written_mat(
                output_path, tuple(train_matrix.shape), tuple(test_matrix.shape)
            )
            records.append(
                {
                    "site": site,
                    "trial": trial,
                    "file": output_path.name,
                    "file_size_bytes": output_path.stat().st_size,
                    "file_sha256": _sha256_path(output_path),
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
        "protocol": PROTOCOL,
        "input_file": h5_path.name,
        "input_size_bytes": input_size_bytes,
        "input_sha256": input_sha256,
        "hdf_key": key,
        "sites": list(selected_sites),
        "trials": list(selected_trials),
        "query_sample_size": N_SAMPLE,
        "train_fraction": TRAIN_FRACTION,
        "train_queries": round(N_SAMPLE * TRAIN_FRACTION),
        "test_queries": N_SAMPLE - round(N_SAMPLE * TRAIN_FRACTION),
        "click_count_filter": None,
        "click_count_filter_enabled": False,
        "position_encoding": None,
        "position_encoding_enabled": False,
        "position_semantics": (
            "dense zero-based displayed order; retained for stages, rankings, "
            "and deterministic tie-breaking"
        ),
        "matrix_column_count": MATRIX_COLUMN_COUNT,
        "matrix_columns": list(BASE_COLUMNS),
        "model_feature_column_range_1_based": [8, 20],
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "standardized_columns": list(STANDARDIZED_COLUMNS),
        "split_seed": "trial",
        "standardization": "separate train/test sample SD (ddof=1)",
        "hash_contract": (
            "SHA-256; query IDs are little-endian int64 sequences in seeded "
            "shuffle order; matrix payloads are row-major little-endian float64"
        ),
        "source_sites": source_records,
        "files": records,
    }
    _write_manifest(manifest_path, manifest)
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
