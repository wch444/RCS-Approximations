#!/usr/bin/env python3
"""Validate and summarize the canonical mixed-order counterfactual experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import stat
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from common.python.runtime import ROOT


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RECORD = EXPERIMENT_ROOT / "results"

SITES = (5, 14, 15, 24, 32)
FULL_TRIALS = tuple(range(1, 11))
SMOKE_TRIALS = (1,)
FULL_QUERIES_PER_TASK = 500
SMOKE_QUERIES_PER_TASK = 30
SAMPLING_SEED_BASE = 20260809
EVALUATION_SEED_BASE = 120260809
DIRICHLET_SEED = 2026
DIRICHLET_DRAWS = 20_000
PROVENANCE_SCHEMA_VERSION = 3
PUBLICATION_LOCK_FILE = ".mixed-order-analysis.lock"

ORDERINGS = (
    "Default",
    "PriceLowHigh",
    "PriceHighLow",
    "Location",
    "StarRating",
)
ORDERING_TEX = (
    "Default (position)",
    "Price low$\\to$high",
    "Price high$\\to$low",
    "Location score",
    "Star rating",
)
H_COLUMNS = tuple(f"H_{name}" for name in ORDERINGS)
G_COLUMNS = tuple(f"G_{name}" for name in ORDERINGS)
ALL_COLUMNS = tuple(f"ALL_{name}" for name in ORDERINGS)
IDENTIFIER_COLUMNS = (
    "site",
    "trial",
    "srch_id",
    "n_prod",
    "card_SqL",
    "UB",
    "sampling_seed",
    "evaluation_seed",
)
EXPECTED_COLUMNS = (*IDENTIFIER_COLUMNS, *H_COLUMNS, *G_COLUMNS, *ALL_COLUMNS)
NUMERIC_COLUMNS = EXPECTED_COLUMNS
INTEGER_COLUMNS = (
    "site",
    "trial",
    "srch_id",
    "n_prod",
    "card_SqL",
    "sampling_seed",
    "evaluation_seed",
)
EXPEDIA_COLUMNS = {5: 28, 14: 28, 15: 28, 24: 27, 32: 28}
MMNL_MARKET_SEED_INDEX = {
    market: index for index, market in enumerate((15, 14, 5, 24, 32), 1)
}
UNIFIED_MODEL_FIELDS = {
    "scm",
    "mnl",
    "mmnl_alpha",
    "mmnl_beta",
    "cmnl",
    "seqinv",
    "nsmnl",
    "nsmnl_lambda",
}
UNIFIED_COMMON_FIELDS = {
    "schema_version",
    "artifact_type",
    "dataset",
    "market",
    "trial",
    "seed",
    "rng",
    "mmnl_seed",
    "mmnl_seed_rule",
    "feature_start",
    "feature_end",
    "data_schema_version",
    "matrix_columns",
    "page_size",
    "train_rows",
    "test_rows",
    "train_sessions",
    "test_sessions",
    "mmnl_components",
    "mmnl_estimator_version",
    "scm_parameter_order",
    "data_file",
    "models",
}
EXPEDIA_EXTRA_FIELDS = {
    "parameterization",
    "reference_bin",
    "reference_feature_index",
    "position_feature_indices",
    "constrained_zero_bins",
    "constrained_zero_feature_indices",
    "reduced_feature_indices",
    "full_feature_dimension",
    "reduced_feature_dimension",
    "full_train_rank",
    "train_rank",
    "mmnl_beta0_full",
    "mmnl_beta0_reduced",
    "mmnl_beta_reduced",
}
EXPEDIA_FIT_FIELDS = UNIFIED_COMMON_FIELDS | EXPEDIA_EXTRA_FIELDS


def sampling_seed(site: int, trial: int) -> int:
    return SAMPLING_SEED_BASE + 100_000 * site + 1_000 * trial


def evaluation_seed(site: int, trial: int, query_index: int) -> int:
    return EVALUATION_SEED_BASE + 100_000 * site + 1_000 * trial + query_index


def _scalar_string(value: object, label: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} must be UTF-8 text.") from error
    if isinstance(value, (str, np.str_)):
        return str(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        return _scalar_string(value.reshape(-1)[0], label)
    raise ValueError(f"{label} must be scalar text, found {type(value).__name__}.")


def _integer(value: object, label: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must be a scalar integer.")
    try:
        number = float(array.reshape(-1)[0])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a scalar integer.") from error
    if not np.isfinite(number) or number != round(number):
        raise ValueError(f"{label} must be a finite exact integer.")
    return int(number)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, np.str_)):
        return (_scalar_string(value, label),)
    array = np.asarray(value, dtype=object)
    return tuple(
        _scalar_string(item, f"{label}[{index}]")
        for index, item in enumerate(array.reshape(-1))
    )


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    array = np.asarray(value, dtype=object)
    return tuple(
        _integer(item, f"{label}[{index}]")
        for index, item in enumerate(array.reshape(-1))
    )


def _record_list(value: object, label: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    def collect(item: object) -> None:
        if isinstance(item, dict):
            records.append(item)
        elif isinstance(item, np.ndarray):
            for child in item.reshape(-1):
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
        else:
            raise ValueError(
                f"{label} must be a MATLAB struct array; "
                f"found {type(item).__name__}."
            )

    collect(value)
    return records


def _required_fields(
    record: dict[str, object], required: Sequence[str], label: str
) -> None:
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}.")


def _repo_file(
    value: object,
    label: str,
    *,
    reject_symlink: bool = False,
) -> tuple[str, Path]:
    text = _scalar_string(value, label)
    if not text or text != text.strip() or "\x00" in text:
        raise ValueError(f"{label} must be a canonical repo-relative path.")
    normalized = text.replace("\\", "/")
    relative = Path(normalized)
    if (
        relative.is_absolute()
        or re.match(r"^[A-Za-z]:/", normalized)
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != normalized
    ):
        raise ValueError(
            f"{label} must be repo-relative and cannot contain a path escape: {text!r}."
        )
    logical_path = BASE.joinpath(*relative.parts)
    base_resolved = BASE.resolve()
    resolved = logical_path.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise ValueError(f"{label} resolves outside the repository: {relative}.")
    if not logical_path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {logical_path}")
    if reject_symlink and logical_path.is_symlink():
        raise ValueError(f"{label} must be a regular file, not a symbolic link.")
    return relative.as_posix(), logical_path


def _validate_mat_rows(out: object, frame: pd.DataFrame) -> None:
    matrix = np.asarray(out)
    expected_shape = (len(frame), len(NUMERIC_COLUMNS))
    if matrix.ndim != 2 or matrix.shape != expected_shape:
        raise ValueError(
            f"Sibling MAT out shape mismatch: expected {expected_shape}, "
            f"found {matrix.shape}."
        )
    if not np.issubdtype(matrix.dtype, np.number) or not np.isrealobj(matrix):
        raise ValueError("Sibling MAT out must be a real numeric matrix.")
    matrix = matrix.astype(float, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("Sibling MAT out contains NaN or infinite values.")

    csv_matrix = frame.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=float)
    integer_indices = [NUMERIC_COLUMNS.index(name) for name in INTEGER_COLUMNS]
    if not np.array_equal(matrix[:, integer_indices], csv_matrix[:, integer_indices]):
        raise ValueError(
            "Mixed-order CSV rows do not match sibling MAT out integer fields "
            "in the same order."
        )
    # MATLAB's writetable decimal round trip can move a finite double by a few
    # 1e-13 (observed for UB), so compare the redundant numeric MAT copy with a
    # tight finite-number tolerance.
    if not np.allclose(matrix, csv_matrix, rtol=1e-12, atol=1e-10):
        difference = float(np.max(np.abs(matrix - csv_matrix)))
        raise ValueError(
            "Mixed-order CSV rows do not match sibling MAT out numeric rows; "
            f"maximum absolute difference is {difference:.3g}."
        )


def _recompute_query_diagnostics(path: Path, label: str) -> dict[str, int]:
    """Recompute the producer's eligible-query counts from data_test."""

    try:
        payload = loadmat(path, variable_names=["data_test"])
    except Exception as error:
        raise ValueError(f"Cannot load {label} data_test: {path}") from error
    if "data_test" not in payload:
        raise ValueError(f"{label} is missing data_test.")
    matrix = payload["data_test"]
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] < 7
        or not np.issubdtype(matrix.dtype, np.number)
        or not np.isrealobj(matrix)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(
            f"{label} data_test must be a nonempty finite real matrix "
            "with at least seven columns."
        )

    session_ids = matrix[:, 0]
    stable_order = np.argsort(session_ids, kind="stable")
    ordered = matrix[stable_order]
    ordered_ids = ordered[:, 0]
    starts = np.r_[0, np.flatnonzero(ordered_ids[1:] != ordered_ids[:-1]) + 1]
    stops = np.r_[starts[1:], len(ordered)]
    outside = (ordered[:, 1] == 0) & (ordered[:, 3] == 100) & (ordered[:, 6] == 0)
    outside_counts = np.add.reduceat(outside.astype(np.int64), starts)
    trailing_outside = outside[stops - 1]
    invalid = (outside_counts != 1) | ~trailing_outside
    if invalid.any():
        raise ValueError(
            f"{label} has {int(invalid.sum())} sessions without exactly one "
            "trailing outside row (col2=0, col4=100, col7=0)."
        )

    product_counts = stops - starts - 1
    eligible = int(np.count_nonzero(product_counts >= 2))
    total = int(len(starts))
    return {
        "total_queries": total,
        "eligible_queries": eligible,
        "excluded_ineligible_queries": total - eligible,
        "test_rows": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
    }


def _numeric_array(value: object, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{label} must be finite real numeric data.")
    return array.astype(float, copy=False)


def _check_fit(
    path: Path,
    *,
    site: int,
    trial: int,
    data_diagnostics: dict[str, int],
) -> None:
    try:
        payload = loadmat(path, variable_names=["fit"], simplify_cells=True)
    except Exception as error:
        raise ValueError(f"Cannot load unified fit: {path}") from error
    fit = payload.get("fit")
    if not isinstance(fit, dict):
        raise ValueError(f"Unified fit must contain one scalar fit struct: {path}")
    if set(fit) != EXPEDIA_FIT_FIELDS:
        raise ValueError(
            f"Unified fit must contain exactly the Expedia schema fields: {path}"
        )
    expected_seed = 42 + 1000 * trial + MMNL_MARKET_SEED_INDEX[site]
    expected_data = f"data/processed/expedia/10k_sample_{site}_{trial}.mat"
    if (
        _integer(fit["schema_version"], "fit.schema_version") != 2
        or _scalar_string(fit["artifact_type"], "fit.artifact_type") != "unified_fit"
        or _scalar_string(fit["dataset"], "fit.dataset") != "expedia"
        or _scalar_string(fit["market"], "fit.market") != str(site)
        or _integer(fit["trial"], "fit.trial") != trial
        or _integer(fit["seed"], "fit.seed") != expected_seed
        or _integer(fit["mmnl_seed"], "fit.mmnl_seed") != expected_seed
        or _scalar_string(fit["mmnl_seed_rule"], "fit.mmnl_seed_rule")
        != "legacy_paper_mmnl_v1"
        or _scalar_string(fit["rng"], "fit.rng") != "twister"
        or _integer(fit["feature_start"], "fit.feature_start") != 8
        or _integer(fit["feature_end"], "fit.feature_end") != 20
        or _integer(fit["data_schema_version"], "fit.data_schema_version") != 1
        or _integer(fit["matrix_columns"], "fit.matrix_columns")
        != EXPEDIA_COLUMNS[site]
        or _integer(fit["matrix_columns"], "fit.matrix_columns")
        != data_diagnostics["matrix_columns"]
        or _integer(fit["page_size"], "fit.page_size") != 10
        or _scalar_string(fit["data_file"], "fit.data_file").replace("\\", "/")
        != expected_data
        or _integer(fit["test_rows"], "fit.test_rows") != data_diagnostics["test_rows"]
        or _integer(fit["test_sessions"], "fit.test_sessions")
        != data_diagnostics["total_queries"]
        or _integer(fit["train_rows"], "fit.train_rows") <= 0
        or _integer(fit["train_sessions"], "fit.train_sessions") <= 0
        or _integer(fit["train_sessions"], "fit.train_sessions")
        > _integer(fit["train_rows"], "fit.train_rows")
        or _integer(fit["mmnl_components"], "fit.mmnl_components") != 3
        or _integer(fit["mmnl_estimator_version"], "fit.mmnl_estimator_version") != 2
        or _scalar_string(fit["scm_parameter_order"], "fit.scm_parameter_order")
        != "beta_alpha"
    ):
        raise ValueError(f"Unified fit metadata are noncanonical: {path}")

    models = fit["models"]
    if not isinstance(models, dict) or set(models) != UNIFIED_MODEL_FIELDS:
        raise ValueError(f"fit.models must contain exactly eight fields: {path}")
    full_dim = EXPEDIA_COLUMNS[site] - 7
    expected_shapes = {
        "scm": 26,
        "mnl": full_dim,
        "cmnl": 2 * full_dim,
        "seqinv": 13,
        "nsmnl": 13,
        "nsmnl_lambda": 4,
        "mmnl_alpha": 3,
    }
    arrays: dict[str, np.ndarray] = {}
    for name, length in expected_shapes.items():
        arrays[name] = _numeric_array(models[name], f"fit.models.{name}").reshape(-1)
        if arrays[name].size != length:
            raise ValueError(f"fit.models.{name} has the wrong dimension: {path}")
    beta = _numeric_array(models["mmnl_beta"], "fit.models.mmnl_beta")
    if beta.shape != (full_dim, 3):
        raise ValueError(f"fit.models.mmnl_beta has the wrong dimension: {path}")
    if (
        (arrays["mmnl_alpha"] < 0).any()
        or abs(float(arrays["mmnl_alpha"].sum()) - 1) > 1e-10
        or (arrays["nsmnl_lambda"] < 0).any()
        or abs(float(arrays["nsmnl_lambda"].sum()) - 1) > 1e-8
    ):
        raise ValueError(f"Unified fit mixture weights are invalid: {path}")

    def indices(name: str) -> np.ndarray:
        values = _numeric_array(fit[name], f"fit.{name}").reshape(-1)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"fit.{name} must contain exact indices: {path}")
        return values.astype(int)

    reference = _integer(fit["reference_feature_index"], "fit reference")
    constrained = indices("constrained_zero_feature_indices")
    reduced = indices("reduced_feature_indices")
    expected_reduced = np.array(
        [
            index
            for index in range(1, full_dim + 1)
            if index not in {reference, *constrained}
        ]
    )
    if (
        _scalar_string(fit["parameterization"], "fit.parameterization")
        != "position_bin_0_reference_and_train_zero_v1"
        or _integer(fit["reference_bin"], "fit.reference_bin") != 0
        or reference != 14
        or not np.array_equal(
            indices("position_feature_indices"), np.arange(14, full_dim + 1)
        )
        or not np.array_equal(reduced, expected_reduced)
        or _integer(fit["full_feature_dimension"], "fit full dimension") != full_dim
        or _integer(fit["reduced_feature_dimension"], "fit reduced dimension")
        != reduced.size
        or not 0 < _integer(fit["full_train_rank"], "fit full rank") <= full_dim
        or not 0 < _integer(fit["train_rank"], "fit reduced rank") <= reduced.size
    ):
        raise ValueError(f"Unified fit identified parameterization is invalid: {path}")
    beta0_full = _numeric_array(fit["mmnl_beta0_full"], "fit.mmnl_beta0_full")
    beta0_reduced = _numeric_array(fit["mmnl_beta0_reduced"], "fit.mmnl_beta0_reduced")
    beta_reduced = _numeric_array(fit["mmnl_beta_reduced"], "fit.mmnl_beta_reduced")
    if (
        beta0_full.shape != (full_dim, 3)
        or beta0_reduced.shape != (reduced.size, 3)
        or beta_reduced.shape != (reduced.size, 3)
        or not np.array_equal(beta_reduced, beta[reduced - 1])
        or not np.all(beta[np.setdiff1d(np.arange(full_dim), reduced - 1)] == 0)
    ):
        raise ValueError(f"Unified fit full/reduced MMNL parameters disagree: {path}")


def load_and_validate_provenance(
    input_path: Path,
    frame: pd.DataFrame,
    *,
    mode: str,
    sites: Sequence[int],
    trials: Sequence[int],
    queries_per_task: int,
) -> dict[str, object]:
    """Validate the sibling MAT's design contract and numeric copy."""

    sibling_mat = input_path.with_suffix(".mat")
    if not sibling_mat.is_file():
        raise FileNotFoundError(
            f"A sibling MAT provenance artifact is required: {sibling_mat}"
        )
    try:
        payload = loadmat(
            sibling_mat,
            variable_names=["out", "sites", "trials", "provenance"],
            simplify_cells=True,
        )
    except Exception as error:
        raise ValueError(
            f"Cannot load sibling MAT provenance: {sibling_mat}"
        ) from error
    required_variables = {"out", "sites", "trials", "provenance"}
    missing_variables = sorted(required_variables - payload.keys())
    if missing_variables:
        raise ValueError(
            f"Sibling MAT is missing required variables: {missing_variables}."
        )
    provenance = payload["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("Sibling MAT provenance must be a scalar MATLAB struct.")

    required_provenance_fields = (
        "schema_version",
        "artifact_type",
        "mode",
        "sites",
        "trials",
        "queries_per_task",
        "expected_rows",
        "actual_rows",
        "feature_start",
        "feature_end",
        "orderings",
        "csv_columns",
        "sampling_seed_base",
        "sampling_seed_rule",
        "evaluation_seed_base",
        "evaluation_seed_rule",
        "input_files",
    )
    _required_fields(provenance, required_provenance_fields, "MAT provenance")

    schema_version = _integer(provenance["schema_version"], "schema_version")
    if schema_version != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported mixed-order provenance schema: expected "
            f"{PROVENANCE_SCHEMA_VERSION}, found {schema_version}."
        )
    if _scalar_string(provenance["artifact_type"], "artifact_type") != (
        "mixed_order_counterfactual"
    ):
        raise ValueError(
            "MAT provenance artifact_type is not mixed_order_counterfactual."
        )
    if _scalar_string(provenance["mode"], "mode") != mode:
        raise ValueError("CSV analysis mode does not match MAT provenance mode.")

    expected_sites = tuple(map(int, sites))
    expected_trials = tuple(map(int, trials))
    if _integer_tuple(payload["sites"], "MAT sites") != expected_sites:
        raise ValueError(
            "Sibling MAT site axis does not match the requested site axis."
        )
    if _integer_tuple(payload["trials"], "MAT trials") != expected_trials:
        raise ValueError(
            "Sibling MAT trial axis does not match the requested trial axis."
        )
    if _integer_tuple(provenance["sites"], "provenance.sites") != expected_sites:
        raise ValueError(
            "MAT provenance site axis does not match the requested site axis."
        )
    if _integer_tuple(provenance["trials"], "provenance.trials") != expected_trials:
        raise ValueError(
            "MAT provenance trial axis does not match the requested trial axis."
        )
    if _string_tuple(provenance["orderings"], "provenance.orderings") != ORDERINGS:
        raise ValueError("MAT provenance ordering axis does not match the CSV schema.")
    if _string_tuple(provenance["csv_columns"], "provenance.csv_columns") != (
        EXPECTED_COLUMNS
    ):
        raise ValueError(
            "MAT provenance CSV schema does not match the analyzer schema."
        )

    expected_rows = len(expected_sites) * len(expected_trials) * queries_per_task
    scalar_contract = {
        "queries_per_task": queries_per_task,
        "expected_rows": expected_rows,
        "actual_rows": len(frame),
        "feature_start": 8,
        "feature_end": 20,
        "sampling_seed_base": SAMPLING_SEED_BASE,
        "evaluation_seed_base": EVALUATION_SEED_BASE,
    }
    for field, expected_value in scalar_contract.items():
        actual_value = _integer(provenance[field], f"provenance.{field}")
        if actual_value != expected_value:
            raise ValueError(
                f"MAT provenance {field} mismatch: expected {expected_value}, "
                f"found {actual_value}."
            )

    seed_rules = {
        "sampling_seed_rule": "sampling_seed_base + 100000*site + 1000*trial",
        "evaluation_seed_rule": (
            "evaluation_seed_base + 100000*site + 1000*trial + query_index"
        ),
    }
    for field, expected_value in seed_rules.items():
        if _scalar_string(provenance[field], f"provenance.{field}") != expected_value:
            raise ValueError(f"MAT provenance {field} is noncanonical.")

    _validate_mat_rows(payload["out"], frame)

    expected_task_keys = [
        (site, trial) for site in expected_sites for trial in expected_trials
    ]
    input_records = _record_list(provenance["input_files"], "input_files")
    if len(input_records) != len(expected_task_keys):
        raise ValueError(
            "MAT provenance input_files must contain exactly one record per task: "
            f"expected {len(expected_task_keys)}, found {len(input_records)}."
        )
    verified_inputs: list[dict[str, object]] = []
    verified_paths: list[Path] = []
    actual_task_keys: list[tuple[int, int]] = []
    for index, record in enumerate(input_records):
        label = f"input_files[{index}]"
        _required_fields(
            record,
            (
                "site",
                "trial",
                "data_path",
                "fit_path",
                "total_queries",
                "eligible_queries",
                "excluded_ineligible_queries",
            ),
            label,
        )
        site = _integer(record["site"], f"{label}.site")
        trial = _integer(record["trial"], f"{label}.trial")
        actual_task_keys.append((site, trial))
        total = _integer(record["total_queries"], f"{label}.total_queries")
        eligible = _integer(record["eligible_queries"], f"{label}.eligible_queries")
        excluded = _integer(
            record["excluded_ineligible_queries"],
            f"{label}.excluded_ineligible_queries",
        )
        if total < 0 or eligible < 0 or excluded < 0 or total != eligible + excluded:
            raise ValueError(
                f"{label} query diagnostics must satisfy "
                "total_queries = eligible_queries + excluded_ineligible_queries."
            )
        if eligible < queries_per_task:
            raise ValueError(
                f"{label} has {eligible} eligible queries but "
                f"{queries_per_task} are required."
            )

        expected_data = f"data/processed/expedia/10k_sample_{site}_{trial}.mat"
        expected_fit = f"opt_data/expedia/fit_{site}_{trial}.mat"
        data_relative, data_path = _repo_file(record["data_path"], f"{label}.data")
        fit_relative, fit_path = _repo_file(
            record["fit_path"], f"{label}.fit", reject_symlink=True
        )
        if fit_path.parent.is_symlink() or not fit_path.parent.is_dir():
            raise ValueError("Unified Expedia fit directory cannot be a symlink.")
        if data_relative != expected_data or fit_relative != expected_fit:
            raise ValueError(f"{label} input paths do not match the canonical task.")
        recomputed_diagnostics = _recompute_query_diagnostics(
            data_path, f"{label}.data"
        )
        recorded_diagnostics = {
            "total_queries": total,
            "eligible_queries": eligible,
            "excluded_ineligible_queries": excluded,
        }
        recomputed_recorded_fields = {
            key: recomputed_diagnostics[key] for key in recorded_diagnostics
        }
        if recomputed_recorded_fields != recorded_diagnostics:
            raise ValueError(
                f"{label} query diagnostics mismatch: recorded "
                f"{recorded_diagnostics}, recomputed {recomputed_recorded_fields}."
            )
        _check_fit(
            fit_path,
            site=site,
            trial=trial,
            data_diagnostics=recomputed_diagnostics,
        )
        verified_inputs.append(
            {
                "site": site,
                "trial": trial,
                "total_queries": total,
                "eligible_queries": eligible,
                "excluded_ineligible_queries": excluded,
                "data_path": data_relative,
                "fit_path": fit_relative,
            }
        )
        verified_paths.extend((data_path, fit_path))
    if actual_task_keys != expected_task_keys:
        raise ValueError(
            "MAT provenance input task grid/order mismatch: "
            f"expected {expected_task_keys}, found {actual_task_keys}."
        )

    return {
        "sibling_mat": sibling_mat,
        "producer_schema_version": schema_version,
        "csv_rows_match_mat_out": True,
        "verified_inputs": verified_inputs,
        "input_paths": tuple(verified_paths),
    }


def _require_exact_columns(frame: pd.DataFrame) -> None:
    actual = tuple(frame.columns)
    if actual != EXPECTED_COLUMNS:
        missing = [name for name in EXPECTED_COLUMNS if name not in actual]
        extra = [name for name in actual if name not in EXPECTED_COLUMNS]
        raise ValueError(
            "Mixed-order CSV schema mismatch: "
            f"missing={missing}, extra={extra}, order={actual}"
        )


def validate_results(
    frame: pd.DataFrame,
    *,
    sites: Sequence[int] = SITES,
    trials: Sequence[int] = FULL_TRIALS,
    queries_per_task: int = FULL_QUERIES_PER_TASK,
) -> None:
    """Validate the complete experiment axis, seeds, and numeric bounds."""

    _require_exact_columns(frame)
    expected_rows = len(sites) * len(trials) * queries_per_task
    if len(frame) != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} rows, found {len(frame)}.")

    numeric = frame.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Mixed-order CSV contains NaN or infinite numeric values.")

    for column in INTEGER_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.array_equal(values, np.round(values)):
            raise ValueError(f"Column {column} must contain exact integers.")

    actual_sites = tuple(sorted(frame["site"].astype(int).unique()))
    actual_trials = tuple(sorted(frame["trial"].astype(int).unique()))
    if actual_sites != tuple(sorted(sites)):
        raise ValueError(f"Site axis mismatch: expected {sites}, found {actual_sites}.")
    if actual_trials != tuple(sorted(trials)):
        raise ValueError(
            f"Trial axis mismatch: expected {trials}, found {actual_trials}."
        )

    if frame.duplicated(["site", "trial", "srch_id"]).any():
        raise ValueError("Duplicate (site, trial, srch_id) key in mixed-order CSV.")

    expected_tasks = {(int(site), int(trial)) for site in sites for trial in trials}
    actual_tasks = {
        (int(site), int(trial))
        for site, trial in frame[["site", "trial"]].itertuples(index=False, name=None)
    }
    if actual_tasks != expected_tasks:
        missing = sorted(expected_tasks - actual_tasks)
        extra = sorted(actual_tasks - expected_tasks)
        raise ValueError(f"Task axis mismatch: missing={missing}, extra={extra}.")

    expected_row_axis = [
        (int(site), int(trial), evaluation_seed(int(site), int(trial), query_index))
        for site in sites
        for trial in trials
        for query_index in range(1, queries_per_task + 1)
    ]
    actual_row_axis = list(
        frame[["site", "trial", "evaluation_seed"]]
        .astype(int)
        .itertuples(index=False, name=None)
    )
    if actual_row_axis != expected_row_axis:
        raise ValueError(
            "Mixed-order row axis/order must be site-major, trial-minor, and "
            "query-index ordered by evaluation_seed."
        )

    for (site, trial), task in frame.groupby(["site", "trial"], sort=True):
        site = int(site)
        trial = int(trial)
        if len(task) != queries_per_task:
            raise ValueError(
                f"Site {site} trial {trial} has {len(task)} rows; "
                f"expected {queries_per_task}."
            )
        if task["srch_id"].nunique() != queries_per_task:
            raise ValueError(f"Duplicate query within site {site} trial {trial}.")

        expected_sampling_seed = sampling_seed(site, trial)
        task_sampling_seeds = set(task["sampling_seed"].astype(int))
        if task_sampling_seeds != {expected_sampling_seed}:
            raise ValueError(
                f"Sampling seed mismatch for site {site} trial {trial}: "
                f"expected {expected_sampling_seed}, found {task_sampling_seeds}."
            )

        first_evaluation_seed = evaluation_seed(site, trial, 1)
        expected_evaluation_seeds = set(
            range(first_evaluation_seed, first_evaluation_seed + queries_per_task)
        )
        actual_evaluation_seeds = set(task["evaluation_seed"].astype(int))
        if actual_evaluation_seeds != expected_evaluation_seeds:
            raise ValueError(
                f"Evaluation seed axis mismatch for site {site} trial {trial}."
            )

    if (frame["n_prod"] < 2).any():
        raise ValueError("Every evaluated query must contain at least two products.")
    if (frame["card_SqL"] < 1).any() or (frame["card_SqL"] > frame["n_prod"]).any():
        raise ValueError("Invalid robust-assortment cardinality.")
    if (frame["UB"] <= 0).any():
        raise ValueError("Every query must have a positive upper bound.")

    ratios = frame.loc[:, (*H_COLUMNS, *G_COLUMNS, *ALL_COLUMNS)].to_numpy(dtype=float)
    if (ratios < 0).any() or (ratios > 1 + 1e-6).any():
        raise ValueError("A normalized revenue ratio lies outside [0, 1].")


def aggregate(
    frame: pd.DataFrame, *, pooled: bool | None = None
) -> dict[str, np.ndarray | float | int | bool | str]:
    """Aggregate query ratios and estimate uncertainty across trial replicates.

    A site-level standard error uses the trial-specific query means.  A pooled
    standard error first averages each site/trial cell across queries and then
    averages those cell means equally across sites within each trial.  Queries
    within a trial are therefore not treated as independent replications.
    """

    H = frame.loc[:, H_COLUMNS].to_numpy(dtype=float)
    G = frame.loc[:, G_COLUMNS].to_numpy(dtype=float)
    A = frame.loc[:, ALL_COLUMNS].to_numpy(dtype=float)
    n = len(frame)
    if n == 0:
        raise ValueError("Cannot aggregate an empty mixed-order result frame.")
    if pooled is None:
        pooled = frame["site"].nunique() > 1
    if not pooled and frame["site"].nunique() != 1:
        raise ValueError("A site-level aggregate must contain exactly one site.")

    metric_columns = (*H_COLUMNS, *G_COLUMNS, *ALL_COLUMNS)
    cell_means = frame.groupby(["site", "trial"], sort=True)[
        list(metric_columns)
    ].mean()
    if pooled:
        trial_means = cell_means.groupby(level="trial", sort=True).mean()
        se_unit = "trial mean of equally weighted site cell means"
    else:
        trial_means = cell_means.droplevel("site")
        se_unit = "within-site trial cell mean"

    n_trials = len(trial_means)
    means = trial_means.loc[:, list(metric_columns)].to_numpy(dtype=float)
    H_trials = means[:, : len(H_COLUMNS)]
    G_trials = means[:, len(H_COLUMNS) : len(H_COLUMNS) + len(G_COLUMNS)]
    A_trials = means[:, len(H_COLUMNS) + len(G_COLUMNS) :]
    rH = H_trials.mean(axis=0)
    rG = G_trials.mean(axis=0)
    rA = A_trials.mean(axis=0)
    if n_trials > 1:
        seH = H_trials.std(axis=0, ddof=1) / np.sqrt(n_trials)
        seG = G_trials.std(axis=0, ddof=1) / np.sqrt(n_trials)
        seA = A_trials.std(axis=0, ddof=1) / np.sqrt(n_trials)
        se_estimable = True
    else:
        seH = np.full(len(H_COLUMNS), np.nan)
        seG = np.full(len(G_COLUMNS), np.nan)
        seA = np.full(len(ALL_COLUMNS), np.nan)
        se_estimable = False
    return {
        "n": n,
        "n_trials": n_trials,
        "se_estimable": se_estimable,
        "se_unit": se_unit,
        "rH": rH,
        "rG": rG,
        "rA": rA,
        "seH": seH,
        "seG": seG,
        "seA": seA,
        "worstH": float(rH.min()),
        "worstG": float(rG.min()),
        "worstA": float(rA.min()),
        "unifH": float(rH.mean()),
        "unifG": float(rG.mean()),
        "unifA": float(rA.mean()),
        "per_query_worst_H_mean": float(H.min(axis=1).mean()),
        "per_query_worst_H_min": float(H.min(axis=1).min()),
    }


def build_summary(
    overall: dict[str, np.ndarray | float | int | bool | str],
    by_site: dict[int, dict[str, np.ndarray | float | int | bool | str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: list[
        tuple[str, int | str, dict[str, np.ndarray | float | int | bool | str]]
    ] = [("site", site, by_site[site]) for site in sorted(by_site)]
    scopes.append(("all", "All", overall))
    for scope, site, stats in scopes:
        for index, ordering in enumerate(ORDERINGS):
            rows.append(
                {
                    "Scope": scope,
                    "Site": site,
                    "N": int(stats["n"]),
                    "Trials": int(stats["n_trials"]),
                    "SE_Estimable": bool(stats["se_estimable"]),
                    "SE_Unit": str(stats["se_unit"]),
                    "Ordering": ordering,
                    "H_Mean": float(stats["rH"][index]),
                    "H_SE": float(stats["seH"][index]),
                    "G_Mean": float(stats["rG"][index]),
                    "G_SE": float(stats["seG"][index]),
                    "ALL_Mean": float(stats["rA"][index]),
                    "ALL_SE": float(stats["seA"][index]),
                    "H_WorstOrdering": float(stats["worstH"]),
                    "H_UniformMixture": float(stats["unifH"]),
                    "G_UniformMixture": float(stats["unifG"]),
                    "ALL_WorstOrdering": float(stats["worstA"]),
                }
            )
    return pd.DataFrame(rows)


def build_table(
    overall: dict[str, np.ndarray | float | int | bool | str],
    by_site: dict[int, dict[str, np.ndarray | float | int | bool | str]],
) -> str:
    def f3(value: float) -> str:
        return f"{value:.3f}"

    lines = [
        r"\begin{tabular}{lccccc|cc|c}",
        r"\toprule",
        "Site & "
        + " & ".join(ORDERING_TEX)
        + r" & \shortstack{Worst\\$\theta$}"
        + r" & \shortstack{Uniform\\$\theta$}"
        + r" & \shortstack{Oracle\\(unif. $\theta$)} \\",
        r"\midrule",
    ]
    for site in sorted(by_site):
        stats = by_site[site]
        lines.append(
            f"{site} & "
            + " & ".join(f3(value) for value in stats["rH"])
            + f" & {f3(stats['worstH'])}"
            + f" & {f3(stats['unifH'])}"
            + f" & {f3(stats['unifG'])} "
            + r"\\"
        )
    lines.append(r"\midrule")
    lines.append(
        r"\textbf{All} & "
        + " & ".join(rf"\textbf{{{f3(value)}}}" for value in overall["rH"])
        + rf" & \textbf{{{f3(overall['worstH'])}}}"
        + rf" & \textbf{{{f3(overall['unifH'])}}}"
        + f" & {f3(overall['unifG'])} "
        + r"\\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _temporary_path(target: Path, *, prefix: str | None = None) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=prefix or f".{target.stem}.",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _regular_directory(path: Path, label: str) -> Path:
    lexical = Path(path)
    try:
        mode = lexical.lstat().st_mode
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISDIR(mode):
            raise ValueError(f"{label} must be a regular directory: {lexical}")
    lexical.mkdir(parents=True, exist_ok=True)
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError(f"{label} must be a regular directory: {lexical}")
    return lexical.resolve()


def _validate_publication_targets(targets: Sequence[Path]) -> None:
    for target in targets:
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(
                f"Mixed-order publication target is not a regular file: {target}"
            )


@contextmanager
def _exclusive_publication_lock() -> Iterable[None]:
    # One fixed experiment-local lock serializes the entire generation even
    # when callers use distinct output directories and shared paper targets.
    lock_root = _regular_directory(EXPERIMENT_ROOT, "Mixed-order publication lock root")
    lock_path = lock_root / PUBLICATION_LOCK_FILE
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            "Another mixed-order analysis publication holds the exclusive lock."
        ) from error
    owned_identity = os.fstat(descriptor)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            current_identity = lock_path.lstat()
        except FileNotFoundError:  # pragma: no cover - external lock tampering
            pass
        else:
            if (
                current_identity.st_dev == owned_identity.st_dev
                and current_identity.st_ino == owned_identity.st_ino
            ):
                lock_path.unlink()


def _artifact_for_target(staged: Path, target: Path) -> dict[str, object]:
    return {
        "path": str(target.resolve()),
        "bytes": staged.stat().st_size,
    }


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def _analysis_input_snapshots(
    paths: Sequence[Path],
) -> dict[Path, tuple[int, int, int, int]]:
    return {path: _file_identity(path) for path in paths}


def _revalidate_analysis_inputs(
    snapshots: dict[Path, tuple[int, int, int, int]],
) -> None:
    for path, expected in snapshots.items():
        if not path.is_file() or _file_identity(path) != expected:
            raise ValueError(f"Mixed-order input changed during publication: {path}")


def _restore_publication_targets(
    targets: Sequence[Path],
    had_original: dict[Path, bool],
    backups: dict[Path, Path],
) -> list[str]:
    rollback_errors: list[str] = []
    for target in targets:
        restore_copy: Path | None = None
        try:
            if had_original[target]:
                restore_copy = _temporary_path(target, prefix=".mixed-order-restore-")
                shutil.copy2(backups[target], restore_copy)
                os.replace(restore_copy, target)
                if not target.is_file() or target.is_symlink():
                    raise OSError(
                        f"Restored mixed-order target failed verification: {target}"
                    )
            elif target.is_symlink() or target.exists():
                target.unlink()
            if had_original[target] != target.is_file():
                raise OSError(
                    f"Restored mixed-order target has the wrong state: {target}"
                )
        except BaseException as rollback_error:  # pragma: no cover - fatal I/O
            rollback_errors.append(f"{target}: {rollback_error}")
        finally:
            if restore_copy is not None:
                restore_copy.unlink(missing_ok=True)
    return rollback_errors


def _figure_a(
    overall: dict[str, np.ndarray | float | int | bool | str],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    x = np.arange(len(ORDERINGS))
    width = 0.38
    seH = np.asarray(overall["seH"], dtype=float)
    seG = np.asarray(overall["seG"], dtype=float)
    axis.bar(
        x - width / 2,
        overall["rH"],
        width,
        color="#2c7fb8",
        capsize=3,
        yerr=seH if np.isfinite(seH).all() else None,
        label="W-heuristic",
    )
    axis.bar(
        x + width / 2,
        overall["rG"],
        width,
        color="#fec44f",
        capsize=3,
        yerr=seG if np.isfinite(seG).all() else None,
        label="G-heuristic",
    )
    axis.axhline(0.375, linestyle="--", color="black", linewidth=1)
    axis.set_xticks(x)
    axis.set_xticklabels(
        ["Default", "Price\nlow-high", "Price\nhigh-low", "Location", "Star\nrating"],
        fontsize=9,
    )
    axis.set_ylabel("Revenue Ratio")
    axis.set_ylim(0, 1.0)
    axis.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure


def _figure_b(market_H: np.ndarray, market_G: np.ndarray) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    axis.hist(
        market_H,
        bins=60,
        density=False,
        color="#2c7fb8",
        alpha=0.85,
        label="W-heuristic",
    )
    axis.hist(
        market_G,
        bins=60,
        density=False,
        color="#fec44f",
        alpha=0.6,
        label="G-heuristic",
    )
    axis.set_xlabel("Revenue Ratio")
    axis.set_ylabel("Frequency")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
    }


def _publish_analysis_generation(
    *,
    paths: dict[str, Path],
    paper_outputs: Sequence[Path],
    summary_csv: str,
    table: str,
    overall: dict[str, np.ndarray | float | int | bool | str],
    market_H: np.ndarray,
    market_G: np.ndarray,
    manifest: dict[str, object],
    input_snapshots: dict[Path, tuple[int, int, int, int]],
    output_dir: Path,
) -> None:
    publication_targets = [
        *(path for key, path in paths.items() if key != "manifest"),
        *paper_outputs,
        paths["manifest"],
    ]
    if len(set(publication_targets)) != len(publication_targets):
        raise ValueError("Mixed-order publication targets must be unique.")
    _validate_publication_targets(publication_targets)

    with ExitStack() as stack:
        output_stage = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(
                    dir=output_dir, prefix=".mixed-order-generation-"
                )
            )
        )
        staged: dict[Path, Path] = {
            target: output_stage / target.name for target in paths.values()
        }
        paper_stage_dirs: dict[Path, Path] = {}
        for target in paper_outputs:
            parent = target.parent
            if parent not in paper_stage_dirs:
                paper_stage_dirs[parent] = Path(
                    stack.enter_context(
                        tempfile.TemporaryDirectory(
                            dir=parent, prefix=".mixed-order-paper-generation-"
                        )
                    )
                )
            staged[target] = paper_stage_dirs[parent] / target.name

        staged[paths["table"]].write_text(table, encoding="utf-8")
        staged[paths["summary"]].write_text(summary_csv, encoding="utf-8")
        figure_a = _figure_a(overall)
        figure_b = _figure_b(market_H, market_G)
        try:
            figure_a.savefig(
                staged[paths["figure_a_pdf"]], format="pdf", bbox_inches="tight"
            )
            figure_a.savefig(
                staged[paths["figure_a_png"]],
                format="png",
                dpi=150,
                bbox_inches="tight",
            )
            figure_b.savefig(
                staged[paths["figure_b_pdf"]], format="pdf", bbox_inches="tight"
            )
            figure_b.savefig(
                staged[paths["figure_b_png"]],
                format="png",
                dpi=150,
                bbox_inches="tight",
            )
        finally:
            plt.close(figure_a)
            plt.close(figure_b)

        for target in paper_outputs:
            source = (
                staged[paths["figure_a_pdf"]]
                if target.name == "mixed_order_a.pdf"
                else staged[paths["figure_b_pdf"]]
            )
            shutil.copyfile(source, staged[target])

        artifact_targets = [
            *(path for key, path in paths.items() if key != "manifest"),
            *paper_outputs,
        ]
        manifest["outputs"] = [
            _artifact_for_target(staged[target], target) for target in artifact_targets
        ]
        staged[paths["manifest"]].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected_sizes = {
            target: staged[target].stat().st_size for target in publication_targets
        }

        with _exclusive_publication_lock():
            _validate_publication_targets(publication_targets)
            had_original = {target: target.is_file() for target in publication_targets}
            backups: dict[Path, Path] = {}
            try:
                for target in publication_targets:
                    if had_original[target]:
                        backup = _temporary_path(target, prefix=".mixed-order-backup-")
                        backups[target] = backup
                        shutil.copy2(target, backup)
                try:
                    _revalidate_analysis_inputs(input_snapshots)
                    for target in publication_targets:
                        os.replace(staged[target], target)
                    for target, expected in expected_sizes.items():
                        if (
                            not target.is_file()
                            or target.is_symlink()
                            or target.stat().st_size != expected
                        ):
                            raise OSError(
                                "Committed mixed-order output failed verification: "
                                f"{target}"
                            )
                    if paths["table"].read_text(encoding="utf-8") != table:
                        raise ValueError("Committed mixed-order table changed.")
                    if paths["summary"].read_text(encoding="utf-8") != summary_csv:
                        raise ValueError("Committed mixed-order summary changed.")
                    committed_manifest = json.loads(
                        paths["manifest"].read_text(encoding="utf-8")
                    )
                    if committed_manifest != manifest:
                        raise ValueError(
                            "Committed mixed-order manifest changed during publication."
                        )
                    _revalidate_analysis_inputs(input_snapshots)
                except BaseException as publication_error:
                    rollback_errors = _restore_publication_targets(
                        publication_targets,
                        had_original,
                        backups,
                    )
                    if rollback_errors:
                        publication_error.add_note(
                            "Mixed-order publication failed and rollback was incomplete: "
                            + " | ".join(rollback_errors)
                        )
                    raise
            finally:
                for backup in backups.values():
                    backup.unlink(missing_ok=True)


def run_analysis(
    input_path: Path,
    output_dir: Path,
    *,
    mode: str = "full",
    sites: Sequence[int] = SITES,
    trials: Sequence[int] | None = None,
    queries_per_task: int | None = None,
    paper_figure_dir: Path | None = None,
) -> dict[str, object]:
    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be 'full' or 'smoke'.")
    if trials is None:
        trials = FULL_TRIALS if mode == "full" else SMOKE_TRIALS
    if queries_per_task is None:
        queries_per_task = (
            FULL_QUERIES_PER_TASK if mode == "full" else SMOKE_QUERIES_PER_TASK
        )

    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    frame = pd.read_csv(input_path)
    validate_results(
        frame,
        sites=sites,
        trials=trials,
        queries_per_task=queries_per_task,
    )
    provenance_validation = load_and_validate_provenance(
        input_path,
        frame,
        mode=mode,
        sites=sites,
        trials=trials,
        queries_per_task=queries_per_task,
    )

    overall = aggregate(frame, pooled=True)
    by_site = {
        int(site): aggregate(frame.loc[frame["site"].eq(site)], pooled=False)
        for site in sites
    }
    summary = build_summary(overall, by_site)
    table = build_table(overall, by_site)

    rng = np.random.default_rng(DIRICHLET_SEED)
    theta = rng.dirichlet(np.ones(len(ORDERINGS)), size=DIRICHLET_DRAWS)
    market_H = theta @ overall["rH"]
    market_G = theta @ overall["rG"]

    output_dir = _regular_directory(Path(output_dir), "Mixed-order output directory")
    paper_outputs: list[Path] = []
    if mode == "full" and paper_figure_dir is not None:
        paper_figure_dir = _regular_directory(
            Path(paper_figure_dir), "Mixed-order paper-figure directory"
        )
        paper_outputs = [
            paper_figure_dir / "mixed_order_a.pdf",
            paper_figure_dir / "mixed_order_b.pdf",
        ]

    suffix = "" if mode == "full" else "_smoke"
    paths = {
        "table": output_dir / f"mixed_order_table{suffix}.tex",
        "summary": output_dir / f"mixed_order_summary{suffix}.csv",
        "figure_a_pdf": output_dir / f"fig_mixed_order_a{suffix}.pdf",
        "figure_a_png": output_dir / f"fig_mixed_order_a{suffix}.png",
        "figure_b_pdf": output_dir / f"fig_mixed_order_b{suffix}.pdf",
        "figure_b_png": output_dir / f"fig_mixed_order_b{suffix}.png",
        "manifest": output_dir / f"mixed_order_manifest{suffix}.json",
    }

    sibling_mat = Path(provenance_validation["sibling_mat"])
    input_artifacts = [_artifact(input_path), _artifact(sibling_mat)]
    input_snapshots = _analysis_input_snapshots(
        (input_path, sibling_mat, *provenance_validation["input_paths"])
    )
    manifest = {
        "schema_version": 3,
        "artifact_type": "mixed_order_analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "sites": list(map(int, sites)),
        "trials": list(map(int, trials)),
        "queries_per_task": int(queries_per_task),
        "expected_rows": len(sites) * len(trials) * queries_per_task,
        "actual_rows": len(frame),
        "orderings": list(ORDERINGS),
        "sampling_seed_rule": (f"{SAMPLING_SEED_BASE} + 100000*site + 1000*trial"),
        "evaluation_seed_rule": (
            f"{EVALUATION_SEED_BASE} + 100000*site + 1000*trial + query_index"
        ),
        "dirichlet_seed": DIRICHLET_SEED,
        "dirichlet_draws": DIRICHLET_DRAWS,
        "metric": (
            "Mean across queries of query-level "
            "R(S|pi_k)/R_C(N|q,pi_U), not a ratio of pooled revenues."
        ),
        "standard_error": {
            "independence_unit": "trial",
            "ddof": 1,
            "trial_count": len(trials),
            "estimable": len(trials) > 1,
            "site": (
                "sample standard deviation across within-site trial cell means "
                "divided by sqrt(number of trials)"
            ),
            "pooled": (
                "sample standard deviation across trial means, where each trial "
                "mean equally averages its site cell means, divided by "
                "sqrt(number of trials)"
            ),
            "not_estimable_representation": (
                None
                if len(trials) > 1
                else "NA in summary CSV and no plotted error bar"
            ),
        },
        "provenance_validation": {
            "producer_schema_version": provenance_validation["producer_schema_version"],
            "csv_rows_match_mat_out": provenance_validation["csv_rows_match_mat_out"],
            "task_grid_verified": True,
            "verified_task_count": len(provenance_validation["verified_inputs"]),
            "verified_data_file_count": len(provenance_validation["verified_inputs"]),
            "verified_fit_file_count": len(provenance_validation["verified_inputs"]),
            "verified_input_tasks": provenance_validation["verified_inputs"],
        },
        "headline": {
            "H_worst": overall["worstH"],
            "H_uniform": overall["unifH"],
            "G_uniform": overall["unifG"],
            "ALL_worst": overall["worstA"],
            "H_fraction_of_G_uniform": overall["unifH"] / overall["unifG"],
            "dirichlet_H_mean": float(market_H.mean()),
            "dirichlet_H_p05": float(np.percentile(market_H, 5)),
            "dirichlet_H_p95": float(np.percentile(market_H, 95)),
        },
        "inputs": input_artifacts,
        "outputs": [],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    _publish_analysis_generation(
        paths=paths,
        paper_outputs=paper_outputs,
        summary_csv=summary.to_csv(index=False, na_rep="NA"),
        table=table,
        overall=overall,
        market_H=market_H,
        market_G=market_G,
        manifest=manifest,
        input_snapshots=input_snapshots,
        output_dir=output_dir,
    )

    return {
        "frame": frame,
        "overall": overall,
        "by_site": by_site,
        "summary": summary,
        "table": table,
        "manifest": manifest,
        "paths": paths,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=RECORD)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.input is None:
        stem = (
            "mixed_order_results"
            if args.mode == "full"
            else "mixed_order_results_smoke"
        )
        args.input = RECORD / f"{stem}.csv"
    result = run_analysis(
        args.input,
        args.output_dir,
        mode=args.mode,
        paper_figure_dir=None,
    )
    overall = result["overall"]
    print(
        f"validated {len(result['frame'])} rows; "
        f"H worst={overall['worstH']:.4f}, "
        f"H uniform={overall['unifH']:.4f}, "
        f"G uniform={overall['unifG']:.4f}"
    )
    for name, path in result["paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
