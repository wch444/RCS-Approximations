#!/usr/bin/env python3
"""Shared predictive-result validation, aggregation, and table formatting."""

from __future__ import annotations

import numpy as np
import pandas as pd


MODELS = ["CMNL", "MNL", "MMNL", "SeqINV", "NSMNL", "SCM"]
METRICS = ["RMSE", "Hit1", "Hit2", "Hit3"]
RESULT_METRICS = ["NLL", "RMSE", "WeightedSquaredError", "MAPE", "Hit1", "Hit2", "Hit3"]
MARKET_ORDER = {
    "expedia": ["5", "14", "15", "24", "32"],
    "trivago": ["BR", "US", "JP", "UK", "DE"],
}
TRIALS = list(range(1, 11))
UNDEFINED_NLL_MODELS = {"CMNL", "SCM"}


def validate_tasks(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Validate and normalize an exact 300-row predictive trial grid."""

    if dataset not in MARKET_ORDER:
        raise ValueError(f"Unknown dataset: {dataset}")
    expected_columns = ["Market", "Trial", "Model", *RESULT_METRICS]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(f"Columns must be exactly {expected_columns}")
    checked = frame.copy()
    if checked[["Market", "Trial", "Model"]].isna().any().any():
        raise ValueError("Task identifiers cannot be missing.")
    checked["Market"] = checked["Market"].astype(str)
    checked["Model"] = checked["Model"].astype(str)
    trial = pd.to_numeric(checked["Trial"], errors="coerce")
    if trial.isna().any() or not np.equal(trial, np.floor(trial)).all():
        raise ValueError("Trials must be integer values 1 through 10.")
    checked["Trial"] = trial.astype(int)

    keys = ["Market", "Trial", "Model"]
    expected = {
        (market, trial_number, model)
        for market in MARKET_ORDER[dataset]
        for trial_number in TRIALS
        for model in MODELS
    }
    observed = set(checked[keys].itertuples(index=False, name=None))
    if len(checked) != 300 or observed != expected or checked.duplicated(keys).any():
        raise ValueError("Predictive trials must contain the exact 300-task grid.")

    numeric = checked[RESULT_METRICS].apply(pd.to_numeric, errors="coerce")
    other = [metric for metric in RESULT_METRICS if metric != "NLL"]
    if not np.isfinite(numeric[other].to_numpy(dtype=float)).all():
        raise ValueError("Predictive-error and hit metrics must be finite.")
    nll = numeric["NLL"].to_numpy(dtype=float)
    undefined = checked["Model"].isin(UNDEFINED_NLL_MODELS).to_numpy()
    if (
        not checked["NLL"].isna().to_numpy()[undefined].all()
        or not np.isnan(nll[undefined]).all()
    ):
        raise ValueError("CMNL and SCM NLL must be uniformly undefined (NaN).")
    if (
        checked["NLL"].isna().to_numpy()[~undefined].any()
        or not np.isfinite(nll[~undefined]).all()
    ):
        raise ValueError("All other model NLL values must be finite.")
    if (nll[~undefined] < 0).any():
        raise ValueError("Defined NLL values must be nonnegative.")
    if (numeric[["RMSE", "WeightedSquaredError", "MAPE"]] < 0).any().any():
        raise ValueError("Predictive-error metrics must be nonnegative.")
    hits = numeric[["Hit1", "Hit2", "Hit3"]].to_numpy(dtype=float)
    if (hits < 0).any() or (hits > 1).any() or (np.diff(hits, axis=1) < -1e-12).any():
        raise ValueError("Predictive hit rates must be ordered probabilities.")
    checked[RESULT_METRICS] = numeric
    return checked


def aggregate(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    checked = validate_tasks(frame, dataset)
    means = checked.groupby(["Market", "Model"], as_index=False)[RESULT_METRICS].mean()
    axis = pd.MultiIndex.from_product(
        [MARKET_ORDER[dataset], MODELS], names=["Market", "Model"]
    )
    means = means.set_index(["Market", "Model"]).reindex(axis).reset_index()
    undefined = means["Model"].isin(UNDEFINED_NLL_MODELS)
    if (
        len(means) != 30
        or means[[metric for metric in RESULT_METRICS if metric != "NLL"]]
        .isna()
        .any()
        .any()
        or not means.loc[undefined, "NLL"].isna().all()
        or not np.isfinite(means.loc[~undefined, "NLL"].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Aggregation did not produce 30 complete market/model means.")
    return means


def best_mask(values: np.ndarray, metric: str) -> np.ndarray:
    target = np.nanmin(values) if metric == "RMSE" else np.nanmax(values)
    return np.isclose(values, target, rtol=0, atol=5e-12)


def paper_values(means: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset not in MARKET_ORDER:
        raise ValueError(f"Unknown dataset: {dataset}")
    checked = means.copy()
    if not {"Market", "Model", *METRICS}.issubset(checked.columns):
        raise ValueError("Predictive means are missing displayed metrics.")
    checked["Market"] = checked["Market"].astype(str)
    if (
        len(checked) != 30
        or checked.duplicated(["Market", "Model"]).any()
        or set(checked["Market"]) != set(MARKET_ORDER[dataset])
    ):
        raise ValueError("Predictive means must contain the exact 30-row grid.")
    rows: list[dict[str, object]] = []
    for market in MARKET_ORDER[dataset]:
        block = checked[checked["Market"] == market].set_index("Model")
        if set(block.index) != set(MODELS):
            raise ValueError(f"Market {market} does not have all six models.")
        for metric in METRICS:
            values = pd.to_numeric(block.loc[MODELS, metric], errors="coerce").to_numpy(
                float
            )
            if not np.isfinite(values).all():
                raise ValueError("Displayed predictive means must be finite.")
            row: dict[str, object] = {"Market": market, "Measure": metric}
            row.update(dict(zip(MODELS, np.round(values, 4), strict=True)))
            rows.append(row)
    return pd.DataFrame(rows, columns=["Market", "Measure", *MODELS])


def render_latex(paper: pd.DataFrame, dataset: str) -> str:
    """Render already validated paper values as the manuscript table body."""

    lines = [
        r"\begin{tabular}{c|c|cccccc}",
        r"\hline",
        "Market & Measure & " + " & ".join(MODELS) + " \\\\",
        r"\hline",
    ]
    labels = {"RMSE": "RMSE", "Hit1": "1-hit", "Hit2": "2-hit", "Hit3": "3-hit"}
    for market in MARKET_ORDER[dataset]:
        block = paper[paper["Market"].astype(str) == market].set_index("Measure")
        market_label = f"Site {market}" if dataset == "expedia" else market
        for index, metric in enumerate(METRICS):
            values = block.loc[metric, MODELS].to_numpy(float)
            cells = [f"{value:.4f}" for value in values]
            cells = [
                rf"\textbf{{{cell}}}" if mark else cell
                for cell, mark in zip(cells, best_mask(values, metric), strict=True)
            ]
            prefix = rf"\multirow{{4}}*{{{market_label}}}" if index == 0 else ""
            lines.append(
                prefix + " & " + labels[metric] + " & " + " & ".join(cells) + " \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def latex_table(means: pd.DataFrame, dataset: str) -> str:
    return render_latex(paper_values(means, dataset), dataset)
