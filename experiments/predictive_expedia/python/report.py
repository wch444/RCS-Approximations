#!/usr/bin/env python3
"""Validate and merge the two canonical Expedia predictive evaluations."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

from common.python.predictive_reporting import (
    MARKET_ORDER,
    MODELS,
    RESULT_METRICS,
    TRIALS,
    aggregate,
    latex_table,
    paper_values,
)


MARKETS = MARKET_ORDER["expedia"]
NON_MMNL_MODELS = [model for model in MODELS if model != "MMNL"]
CORE_COLUMNS = ["Market", "Trial", "Model", *RESULT_METRICS]
LEGACY_COLUMNS = CORE_COLUMNS
MMNL_STATUS_COLUMNS = [
    "Converged",
    "HitIterationCap",
    "TerminationReason",
    "IterationsAccepted",
]
MMNL_COLUMNS = [*CORE_COLUMNS, *MMNL_STATUS_COLUMNS]
MMNL_MEANS_COLUMNS = [
    "Market",
    "Model",
    *RESULT_METRICS,
    "NTasks",
    "ConvergedTasks",
    "CappedTasks",
]
OUTPUT_TRIALS = "predictive_expedia_trials.csv"
OUTPUT_MEANS = "predictive_expedia_means.csv"
OUTPUT_PAPER = "predictive_expedia_paper.csv"
OUTPUT_TEX = "predictive_expedia_table.tex"
OUTPUT_LOCK = ".predictive_expedia_publish.lock"
EXPECTED_DATA_NAMES = {
    f"10k_sample_{market}_{trial}.mat" for market in MARKETS for trial in TRIALS
}
EXPECTED_FIT_NAMES = {
    f"fit_{market}_{trial}.mat" for market in MARKETS for trial in TRIALS
}
EXPECTED_COLUMNS = {"5": 28, "14": 28, "15": 28, "24": 27, "32": 28}
MMNL_MARKET_SEED_INDEX = {
    market: index for index, market in enumerate(["15", "14", "5", "24", "32"], 1)
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
    "mmnl_diagnostics",
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _identity(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def _read_csv(path: Path) -> tuple[pd.DataFrame, tuple[int, int, int, int]]:
    _require(
        path.is_file() and not path.is_symlink(),
        f"CSV is missing or not regular: {path}",
    )
    state = _identity(path)
    frame = pd.read_csv(path, dtype={"Market": str})
    _require(_identity(path) == state, f"CSV changed while being read: {path}")
    return frame, state


def _integer_series(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    _require(
        numeric.notna().all() and np.equal(numeric, np.floor(numeric)).all(),
        f"{label} must contain integers.",
    )
    return numeric.astype(int)


def _boolean_series(values: pd.Series, label: str) -> pd.Series:
    parsed: list[bool] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, (int, np.integer, float, np.floating)) and value in (
            0,
            1,
        ):
            parsed.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {
            "true",
            "false",
            "1",
            "0",
        }:
            parsed.append(value.strip().lower() in {"true", "1"})
        else:
            raise ValueError(f"{label} must contain logical values.")
    return pd.Series(parsed, index=values.index, dtype=bool)


def _validate_metrics(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    checked = frame.copy()
    numeric = checked[RESULT_METRICS].apply(pd.to_numeric, errors="coerce")
    other = [metric for metric in RESULT_METRICS if metric != "NLL"]
    _require(
        np.isfinite(numeric[other].to_numpy(float)).all(),
        f"{label} metrics must be finite.",
    )
    undefined = checked["Model"].isin({"CMNL", "SCM"}).to_numpy()
    nll = numeric["NLL"].to_numpy(float)
    missing = checked["NLL"].isna().to_numpy()
    _require(
        missing[undefined].all() and np.isnan(nll[undefined]).all(),
        f"{label} CMNL/SCM NLL must be NaN.",
    )
    _require(
        (~missing[~undefined]).all()
        and np.isfinite(nll[~undefined]).all()
        and (nll[~undefined] >= 0).all(),
        f"{label} defined NLL must be finite and nonnegative.",
    )
    _require(
        (numeric[["RMSE", "WeightedSquaredError", "MAPE"]] >= 0).all().all(),
        f"{label} errors must be nonnegative.",
    )
    hits = numeric[["Hit1", "Hit2", "Hit3"]].to_numpy(float)
    _require(
        (hits >= 0).all()
        and (hits <= 1).all()
        and (np.diff(hits, axis=1) >= -1e-12).all(),
        f"{label} hit rates must be ordered probabilities.",
    )
    checked[RESULT_METRICS] = numeric
    return checked


def validate_legacy_trials(frame: pd.DataFrame) -> pd.DataFrame:
    _require(
        frame.columns.tolist() == LEGACY_COLUMNS,
        f"Legacy columns must be exactly {LEGACY_COLUMNS}.",
    )
    checked = frame.copy()
    checked["Market"] = checked["Market"].astype(str)
    checked["Trial"] = _integer_series(checked["Trial"], "Legacy trials")
    checked["Model"] = checked["Model"].astype(str)
    expected = [
        (market, trial, model)
        for market in MARKETS
        for trial in TRIALS
        for model in NON_MMNL_MODELS
    ]
    observed = list(
        checked[["Market", "Trial", "Model"]].itertuples(index=False, name=None)
    )
    _require(
        observed == expected,
        "Legacy trials must be the ordered canonical 250-task grid.",
    )
    return _validate_metrics(checked, "Legacy")


def validate_mmnl_trials(frame: pd.DataFrame) -> pd.DataFrame:
    _require(
        frame.columns.tolist() == MMNL_COLUMNS,
        f"MMNL columns must be exactly {MMNL_COLUMNS}.",
    )
    checked = frame.copy()
    checked["Market"] = checked["Market"].astype(str)
    checked["Trial"] = _integer_series(checked["Trial"], "MMNL trials")
    checked["Model"] = checked["Model"].astype(str)
    expected = [(market, trial, "MMNL") for trial in TRIALS for market in MARKETS]
    observed = list(
        checked[["Market", "Trial", "Model"]].itertuples(index=False, name=None)
    )
    _require(
        observed == expected, "MMNL trials must be the ordered canonical 50-task grid."
    )
    checked["Converged"] = _boolean_series(checked["Converged"], "Converged")
    checked["HitIterationCap"] = _boolean_series(
        checked["HitIterationCap"], "HitIterationCap"
    )
    iterations = _integer_series(checked["IterationsAccepted"], "IterationsAccepted")
    converged = checked["Converged"].to_numpy()
    capped = checked["HitIterationCap"].to_numpy()
    reasons = checked["TerminationReason"].astype(str).to_numpy()
    _require(
        np.logical_xor(converged, capped).all(),
        "Every MMNL task must converge or hit the iteration cap.",
    )
    _require(
        ((iterations >= 3) & (iterations <= 100)).all(),
        "MMNL iterations must be between 3 and 100.",
    )
    _require(
        (reasons[converged] == "consecutive_relative_nll_tolerance").all(),
        "Converged MMNL reasons are invalid.",
    )
    _require(
        (reasons[capped] == "maximum_iterations").all()
        and (iterations.to_numpy()[capped] == 100).all(),
        "Capped MMNL status is invalid.",
    )
    checked["TerminationReason"] = reasons
    checked["IterationsAccepted"] = iterations
    return _validate_metrics(checked, "MMNL")


def _require_exact_grid(
    directory: Path,
    expected: set[str],
    pattern: str,
    label: str,
    *,
    reject_symlinks: bool = False,
) -> list[Path]:
    _require(
        directory.is_dir() and not directory.is_symlink(),
        f"{label} directory is invalid: {directory}",
    )
    entries = [path for path in directory.iterdir() if re.fullmatch(pattern, path.name)]
    found = {path.name for path in entries}
    _require(
        found == expected, f"{label} must contain the exact canonical 50-file grid."
    )
    _require(
        all(
            path.is_file() and (not reject_symlinks or not path.is_symlink())
            for path in entries
        ),
        f"{label} files must be regular and cannot be symbolic links.",
    )
    return [directory / name for name in sorted(expected)]


def _mat_shape(path: Path, variable: str) -> tuple[int, int]:
    state = _identity(path)
    entries = whosmat(path)
    _require(
        _identity(path) == state, f"MAT file changed while being inspected: {path}"
    )
    matches = [entry for entry in entries if entry[0] == variable]
    _require(
        len(matches) == 1 and len(matches[0][1]) == 2 and matches[0][2] == "double",
        f"{path.name} must contain one double {variable} matrix.",
    )
    rows, columns = matches[0][1]
    _require(rows > 0 and columns > 0, f"{path.name} {variable} must be nonempty.")
    return int(rows), int(columns)


def validate_data_grid(
    data_dir: Path,
) -> tuple[
    dict[tuple[str, int], dict[str, int]], dict[Path, tuple[int, int, int, int]]
]:
    paths = _require_exact_grid(
        data_dir, EXPECTED_DATA_NAMES, r"10k_sample_\d+_\d+\.mat", "Data"
    )
    metadata: dict[tuple[str, int], dict[str, int]] = {}
    states: dict[Path, tuple[int, int, int, int]] = {}
    for path in paths:
        match = re.fullmatch(r"10k_sample_(\d+)_(\d+)\.mat", path.name)
        assert match is not None
        market, trial = match.group(1), int(match.group(2))
        train_rows, train_columns = _mat_shape(path, "data_train")
        test_rows, test_columns = _mat_shape(path, "data_test")
        expected_columns = EXPECTED_COLUMNS[market]
        _require(
            train_columns == test_columns == expected_columns,
            f"{path.name} matrix dimensions are noncanonical.",
        )
        metadata[(market, trial)] = {
            "train_rows": train_rows,
            "test_rows": test_rows,
            "columns": train_columns,
        }
        states[path] = _identity(path)
    _require(
        set(metadata) == {(market, trial) for market in MARKETS for trial in TRIALS},
        "Data task grid is incomplete.",
    )
    return metadata, states


def _load_struct(
    path: Path, variable: str
) -> tuple[dict[str, Any], tuple[int, int, int, int]]:
    state = _identity(path)
    payload = loadmat(path, variable_names=[variable], simplify_cells=True)
    _require(_identity(path) == state, f"MAT file changed while being read: {path}")
    value = payload.get(variable)
    _require(
        isinstance(value, dict),
        f"{path.name} must contain one scalar {variable} structure.",
    )
    return value, state


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    _require(
        np.issubdtype(array.dtype, np.number) and not np.iscomplexobj(array),
        f"{label} must be real numeric.",
    )
    array = np.ravel(array).astype(float)
    _require(
        array.size == length and np.isfinite(array).all(),
        f"{label} has invalid dimension or nonfinite values.",
    )
    return array


def _scalar(value: Any, label: str) -> Any:
    array = np.asarray(value)
    _require(array.size == 1, f"{label} must be scalar.")
    return (
        array.reshape(-1)[0].item()
        if hasattr(array.reshape(-1)[0], "item")
        else array.reshape(-1)[0]
    )


def _integer(value: Any, label: str) -> int:
    scalar = _scalar(value, label)
    _require(
        isinstance(scalar, (int, float, np.integer, np.floating))
        and np.isfinite(scalar)
        and float(scalar).is_integer(),
        f"{label} must be an integer.",
    )
    return int(scalar)


def _logical(value: Any, label: str) -> bool:
    scalar = _scalar(value, label)
    _require(scalar in (0, 1, False, True), f"{label} must be logical.")
    return bool(scalar)


def _text(value: Any, label: str) -> str:
    scalar = _scalar(value, label)
    _require(
        isinstance(scalar, str) and scalar != "", f"{label} must be nonempty text."
    )
    return scalar


def _numeric_matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    array = np.asarray(value)
    _require(
        np.issubdtype(array.dtype, np.number)
        and not np.iscomplexobj(array)
        and array.shape == shape
        and np.isfinite(array).all(),
        f"{label} has invalid dimensions or values.",
    )
    return array.astype(float)


def _index_vector(value: Any, label: str) -> np.ndarray:
    array = np.ravel(np.asarray(value))
    _require(
        np.issubdtype(array.dtype, np.number)
        and np.isfinite(array).all()
        and np.equal(array, np.floor(array)).all(),
        f"{label} must contain integer indices.",
    )
    return array.astype(int)


def _validate_diagnostics(fit: dict[str, Any], row: pd.Series, label: str) -> None:
    diagnostics = fit["mmnl_diagnostics"]
    _require(
        isinstance(diagnostics, dict),
        f"{label} diagnostics must be a scalar structure.",
    )
    required = {
        "algorithm",
        "estimator_version",
        "estimation_complete",
        "fitted",
        "converged",
        "hit_iteration_cap",
        "termination_reason",
        "iterations_accepted",
        "iterations_attempted",
        "max_iterations",
        "relative_tolerance",
        "required_consecutive",
        "consecutive",
        "initial_train_nll",
        "final_train_nll",
        "train_nll_history",
        "relative_nll_improvement_history",
        "mstep_exitflag",
        "mstep_beta_accepted",
        "component_order",
        "em_options",
    }
    _require(
        required.issubset(diagnostics), f"{label} diagnostics schema is incomplete."
    )
    iterations = _integer(
        diagnostics["iterations_accepted"], f"{label} accepted iterations"
    )
    converged = _logical(diagnostics["converged"], f"{label} converged")
    capped = _logical(diagnostics["hit_iteration_cap"], f"{label} cap")
    reason = _text(diagnostics["termination_reason"], f"{label} reason")
    _require(
        _logical(diagnostics["estimation_complete"], f"{label} complete")
        and _logical(diagnostics["fitted"], f"{label} fitted")
        and _text(diagnostics["algorithm"], f"{label} algorithm")
        == "chosen_choice_generalized_em"
        and _integer(diagnostics["estimator_version"], f"{label} estimator") == 2
        and _integer(diagnostics["iterations_attempted"], f"{label} attempted")
        == iterations
        and _integer(diagnostics["max_iterations"], f"{label} maximum") == 100
        and _integer(
            diagnostics["required_consecutive"], f"{label} required consecutive"
        )
        == 3
        and np.isclose(float(_scalar(diagnostics["relative_tolerance"], label)), 1e-7),
        f"{label} diagnostic identity is invalid.",
    )
    _require(
        converged == bool(row["Converged"])
        and capped == bool(row["HitIterationCap"])
        and reason == str(row["TerminationReason"])
        and iterations == int(row["IterationsAccepted"]),
        f"{label} fit status disagrees with its evaluation row.",
    )
    options = diagnostics["em_options"]
    _require(
        isinstance(options, dict)
        and _integer(options.get("MaxIterations"), label) == 100
        and np.isclose(float(_scalar(options.get("RelativeTolerance"), label)), 1e-7),
        f"{label} diagnostic EM options are invalid.",
    )
    history = np.ravel(np.asarray(diagnostics["train_nll_history"], dtype=float))
    relative = np.ravel(
        np.asarray(diagnostics["relative_nll_improvement_history"], dtype=float)
    )
    exits = np.ravel(np.asarray(diagnostics["mstep_exitflag"], dtype=float))
    accepted = np.ravel(np.asarray(diagnostics["mstep_beta_accepted"], dtype=float))
    _require(
        history.size == iterations + 1
        and relative.size == exits.size == accepted.size == iterations,
        f"{label} diagnostic histories have invalid lengths.",
    )
    _require(
        np.isfinite(history).all()
        and np.isfinite(relative).all()
        and np.isfinite(exits).all()
        and np.isin(accepted, [0, 1]).all(),
        f"{label} diagnostic histories contain invalid values.",
    )
    _require(
        np.all(np.diff(history) <= 1e-8 * (1 + np.abs(history[:-1]))),
        f"{label} training NLL is not monotone.",
    )
    _require(
        np.isclose(float(_scalar(diagnostics["initial_train_nll"], label)), history[0])
        and np.isclose(
            float(_scalar(diagnostics["final_train_nll"], label)), history[-1]
        ),
        f"{label} terminal diagnostic NLL is inconsistent.",
    )
    order = _index_vector(diagnostics["component_order"], f"{label} component order")
    _require(
        np.array_equal(np.sort(order), np.array([1, 2, 3])),
        f"{label} component order is not a permutation.",
    )


def _check_fit(
    fit: dict[str, Any],
    market: str,
    trial: int,
    info: dict[str, int],
    row: pd.Series,
) -> None:
    label = f"Unified fit {market}/{trial}"
    _require(
        set(fit) == EXPEDIA_FIT_FIELDS,
        f"{label} must contain exactly the canonical Expedia fields.",
    )
    expected_seed = 42 + 1000 * trial + MMNL_MARKET_SEED_INDEX[market]
    expected_file = f"data/processed/expedia/10k_sample_{market}_{trial}.mat"
    _require(
        _integer(fit["schema_version"], label) == 1
        and _text(fit["artifact_type"], label) == "unified_fit"
        and _text(fit["dataset"], label) == "expedia"
        and _text(fit["market"], label) == market
        and _integer(fit["trial"], label) == trial
        and _integer(fit["seed"], label) == expected_seed
        and _integer(fit["mmnl_seed"], label) == expected_seed
        and _text(fit["mmnl_seed_rule"], label) == "legacy_paper_mmnl_v1"
        and _text(fit["rng"], label) == "twister"
        and _integer(fit["feature_start"], label) == 8
        and _integer(fit["feature_end"], label) == 20
        and _integer(fit["data_schema_version"], label) == 1
        and _integer(fit["matrix_columns"], label) == info["columns"]
        and _integer(fit["page_size"], label) == 10
        and _text(fit["data_file"], label).replace("\\", "/") == expected_file
        and _integer(fit["train_rows"], label) == info["train_rows"]
        and _integer(fit["test_rows"], label) == info["test_rows"]
        and 0 < _integer(fit["train_sessions"], label) <= info["train_rows"]
        and 0 < _integer(fit["test_sessions"], label) <= info["test_rows"]
        and _integer(fit["mmnl_components"], label) == 3
        and _integer(fit["mmnl_estimator_version"], label) == 2
        and _text(fit["scm_parameter_order"], label) == "beta_alpha",
        f"{label} metadata are noncanonical.",
    )
    full_dim = info["columns"] - 7
    models = fit["models"]
    _require(
        isinstance(models, dict) and set(models) == UNIFIED_MODEL_FIELDS,
        f"{label} models must contain exactly the eight canonical fields.",
    )
    lengths = {
        "scm": 26,
        "mnl": full_dim,
        "cmnl": 2 * full_dim,
        "seqinv": 13,
        "nsmnl": 13,
        "nsmnl_lambda": 4,
    }
    for name, length in lengths.items():
        _finite_vector(models[name], length, f"{label}:{name}")
    alpha = _finite_vector(models["mmnl_alpha"], 3, f"{label}:mmnl_alpha")
    beta = _numeric_matrix(models["mmnl_beta"], (full_dim, 3), f"{label}:mmnl_beta")
    lam = _finite_vector(models["nsmnl_lambda"], 4, f"{label}:nsmnl_lambda")
    _require(
        (alpha >= 0).all()
        and abs(float(alpha.sum()) - 1) <= 1e-10
        and (lam >= 0).all()
        and abs(float(lam.sum()) - 1) <= 1e-8,
        f"{label} mixture weights are outside their simplices.",
    )

    reduced = _index_vector(fit["reduced_feature_indices"], f"{label} reduced indices")
    constrained = _index_vector(
        fit["constrained_zero_feature_indices"], f"{label} constrained indices"
    )
    reference = _integer(fit["reference_feature_index"], label)
    excluded = np.concatenate(([reference], constrained))
    expected_reduced = np.array(
        [index for index in range(1, full_dim + 1) if index not in set(excluded)]
    )
    _require(
        _text(fit["parameterization"], label)
        == "position_bin_0_reference_and_train_zero_v1"
        and _integer(fit["reference_bin"], label) == 0
        and reference == 14
        and np.array_equal(
            _index_vector(fit["position_feature_indices"], label),
            np.arange(14, full_dim + 1),
        )
        and np.array_equal(reduced, expected_reduced)
        and _integer(fit["full_feature_dimension"], label) == full_dim
        and _integer(fit["reduced_feature_dimension"], label) == reduced.size
        and 0 < _integer(fit["train_rank"], label) <= reduced.size
        and 0 < _integer(fit["full_train_rank"], label) <= full_dim,
        f"{label} parameterization or dimensions are invalid.",
    )
    beta0_full = _numeric_matrix(
        fit["mmnl_beta0_full"], (full_dim, 3), f"{label} beta0 full"
    )
    beta0_reduced = _numeric_matrix(
        fit["mmnl_beta0_reduced"], (reduced.size, 3), f"{label} beta0 reduced"
    )
    beta_reduced = _numeric_matrix(
        fit["mmnl_beta_reduced"], (reduced.size, 3), f"{label} beta reduced"
    )
    zero_rows = (
        np.array(
            [index for index in range(1, full_dim + 1) if index not in set(reduced)]
        )
        - 1
    )
    beta0_equivalent = beta0_full.copy()
    reference_coefficient = beta0_full[reference - 1].copy()
    beta0_equivalent[0] += reference_coefficient
    beta0_equivalent[np.arange(14, full_dim + 1) - 1] -= reference_coefficient
    beta0_equivalent[reference - 1] = 0
    _require(
        np.array_equal(beta0_reduced, beta0_equivalent[reduced - 1])
        and np.array_equal(beta_reduced, beta[reduced - 1])
        and np.all(beta[zero_rows] == 0),
        f"{label} reduced/full parameters disagree.",
    )
    _validate_diagnostics(fit, row, label)


def validate_fits(
    fit_dir: Path,
    data: dict[tuple[str, int], dict[str, int]],
    trials: pd.DataFrame,
) -> dict[Path, tuple[int, int, int, int]]:
    paths = _require_exact_grid(
        fit_dir,
        EXPECTED_FIT_NAMES,
        r"fit_\d+_\d+\.mat",
        "Unified fit",
        reject_symlinks=True,
    )
    indexed = trials.set_index(["Market", "Trial"])
    states: dict[Path, tuple[int, int, int, int]] = {}
    for path in paths:
        match = re.fullmatch(r"fit_(\d+)_(\d+)\.mat", path.name)
        assert match is not None
        market, trial = match.group(1), int(match.group(2))
        fit, state = _load_struct(path, "fit")
        _check_fit(
            fit, market, trial, data[(market, trial)], indexed.loc[(market, trial)]
        )
        states[path] = state
    return states


def validate_mmnl_means(frame: pd.DataFrame, trials: pd.DataFrame) -> None:
    _require(
        frame.columns.tolist() == MMNL_MEANS_COLUMNS,
        f"MMNL means columns must be exactly {MMNL_MEANS_COLUMNS}.",
    )
    checked = frame.copy()
    checked["Market"] = checked["Market"].astype(str)
    checked["Model"] = checked["Model"].astype(str)
    _require(
        list(checked[["Market", "Model"]].itertuples(index=False, name=None))
        == [(market, "MMNL") for market in MARKETS],
        "MMNL means must be the ordered five-market grid.",
    )
    checked[RESULT_METRICS] = checked[RESULT_METRICS].apply(
        pd.to_numeric, errors="coerce"
    )
    for column in ["NTasks", "ConvergedTasks", "CappedTasks"]:
        checked[column] = _integer_series(checked[column], column)
    expected_metrics = (
        trials.groupby("Market", sort=False)[RESULT_METRICS].mean().reindex(MARKETS)
    )
    observed_metrics = checked.set_index("Market")[RESULT_METRICS].reindex(MARKETS)
    np.testing.assert_allclose(
        observed_metrics.to_numpy(float),
        expected_metrics.to_numpy(float),
        rtol=5e-15,
        atol=5e-12,
    )
    expected_converged = (
        trials.groupby("Market", sort=False)["Converged"]
        .sum()
        .reindex(MARKETS)
        .to_numpy(int)
    )
    expected_capped = (
        trials.groupby("Market", sort=False)["HitIterationCap"]
        .sum()
        .reindex(MARKETS)
        .to_numpy(int)
    )
    _require(
        (checked["NTasks"] == 10).all()
        and np.array_equal(checked["ConvergedTasks"].to_numpy(), expected_converged)
        and np.array_equal(checked["CappedTasks"].to_numpy(), expected_capped),
        "MMNL means status counts are inconsistent.",
    )


def build(
    legacy: pd.DataFrame, mmnl: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    legacy_checked = validate_legacy_trials(legacy)
    mmnl_checked = validate_mmnl_trials(mmnl)
    combined = pd.concat(
        [legacy_checked[CORE_COLUMNS], mmnl_checked[CORE_COLUMNS]], ignore_index=True
    )
    market_order = {market: index for index, market in enumerate(MARKETS)}
    model_order = {model: index for index, model in enumerate(MODELS)}
    combined = (
        combined.assign(
            _market=combined["Market"].map(market_order),
            _model=combined["Model"].map(model_order),
        )
        .sort_values(["_market", "Trial", "_model"], kind="stable")
        .drop(columns=["_market", "_model"])
        .reset_index(drop=True)
    )
    means = aggregate(combined, "expedia")
    paper = paper_values(means, "expedia")
    return combined, means, paper, latex_table(means, "expedia")


@contextmanager
def _lock(output_dir: Path) -> Iterator[None]:
    path = output_dir / OUTPUT_LOCK
    _require(
        not path.is_symlink() and (not path.exists() or path.is_file()),
        f"Publication lock is not regular: {path}",
    )
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _restore(
    names: list[str], output_dir: Path, backup: Path, originals: dict[str, bool]
) -> list[str]:
    errors: list[str] = []
    for name in names:
        target = output_dir / name
        try:
            if originals[name]:
                os.replace(backup / name, target)
            elif target.exists() or target.is_symlink():
                target.unlink()
        except BaseException as error:  # pragma: no cover
            errors.append(f"{name}: {error}")
    return errors


def _publish(
    prepared: dict[str, str],
    output_dir: Path,
    states: dict[Path, tuple[int, int, int, int]],
) -> dict[str, Path]:
    names = list(prepared)
    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix=".expedia-generation-"
    ) as staging_name:
        staging = Path(staging_name)
        for name, content in prepared.items():
            (staging / name).write_text(content, encoding="utf-8")
        with _lock(output_dir):
            for path, state in states.items():
                _require(
                    path.is_file() and _identity(path) == state,
                    f"Input changed before publication: {path}",
                )
            with tempfile.TemporaryDirectory(
                dir=output_dir, prefix=".expedia-backup-"
            ) as backup_name:
                backup = Path(backup_name)
                originals: dict[str, bool] = {}
                for name in names:
                    target = output_dir / name
                    _require(
                        not target.is_symlink()
                        and (not target.exists() or target.is_file()),
                        f"Output target is not regular: {target}",
                    )
                    originals[name] = target.exists()
                    if originals[name]:
                        shutil.copy2(target, backup / name)
                try:
                    for name in names:
                        os.replace(staging / name, output_dir / name)
                    for name, content in prepared.items():
                        _require(
                            (output_dir / name).read_text(encoding="utf-8") == content,
                            f"Published output changed: {name}",
                        )
                except BaseException as error:
                    rollback = _restore(names, output_dir, backup, originals)
                    if rollback:
                        raise RuntimeError(
                            f"Publication failed ({error}); rollback failed: {' | '.join(rollback)}"
                        ) from error
                    raise
    return {name: output_dir / name for name in names}


def run(
    legacy_trials: Path,
    mmnl_trials: Path,
    data_dir: Path,
    fit_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    paths = [
        Path(value).resolve()
        for value in [legacy_trials, mmnl_trials, data_dir, fit_dir, output_dir]
    ]
    legacy_path, mmnl_path, data_path, fit_path, output_path = paths
    _require(
        output_path.is_dir() and not output_path.is_symlink(),
        f"Output directory is invalid: {output_path}",
    )
    legacy, legacy_state = _read_csv(legacy_path)
    mmnl, mmnl_state = _read_csv(mmnl_path)
    legacy_checked = validate_legacy_trials(legacy)
    mmnl_checked = validate_mmnl_trials(mmnl)
    means_path = mmnl_path.with_name("predictive_expedia_mmnl_means.csv")
    evaluator_means, means_state = _read_csv(means_path)
    validate_mmnl_means(evaluator_means, mmnl_checked)
    data_metadata, states = validate_data_grid(data_path)
    states.update(validate_fits(fit_path, data_metadata, mmnl_checked))
    states.update(
        {legacy_path: legacy_state, mmnl_path: mmnl_state, means_path: means_state}
    )
    combined, means, paper, tex = build(legacy_checked, mmnl_checked)
    prepared = {
        OUTPUT_TRIALS: combined.to_csv(index=False, lineterminator="\n"),
        OUTPUT_MEANS: means.to_csv(index=False, lineterminator="\n"),
        OUTPUT_PAPER: paper.to_csv(
            index=False, float_format="%.4f", lineterminator="\n"
        ),
        OUTPUT_TEX: tex,
    }
    return _publish(prepared, output_path, states)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-trials", required=True, type=Path)
    parser.add_argument("--mmnl-trials", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--fit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(
        args.legacy_trials,
        args.mmnl_trials,
        args.data_dir,
        args.fit_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
