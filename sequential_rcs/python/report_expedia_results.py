#!/usr/bin/env python3
"""Validate and report the canonical Expedia four-model results.

This module is deliberately dependency-free and noninteractive.  It treats the
canonical 200-row CSV as an auditable artifact: every required key must be
present exactly once, all four models for a site/trial must refer to the same
data hash, and predictive/probability diagnostics must pass before any report
is written.  Nonconverged fits are retained in every derived artifact and
block the publication gate in ``expedia_claims.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RESULT_SCHEMA_VERSION = "expedia_four_model_result_v1"
REPORT_SCHEMA_VERSION = "expedia_four_model_report_v1"
SITES = (5, 14, 15, 24, 32)
TRIALS = tuple(range(1, 11))
MODELS = ("sqmnl", "nsmnl", "iml", "imrcs")
MODEL_LABELS = {
    "sqmnl": "SQ-MNL",
    "nsmnl": "NS-MNL",
    "iml": "IML",
    "imrcs": "IM-RCS",
}
SEED_BASE = 2026081600
PROBABILITY_TOLERANCE = Decimal("1e-8")
EXPECTED_FEATURE_COLUMNS = "8:20"
EXPECTED_STAGE_SIZE = 10
EXPECTED_TRAIN_SESSIONS = 6000
EXPECTED_TEST_SESSIONS = 4000

CANONICAL_COLUMNS = (
    "schema_version",
    "site",
    "trial",
    "model_id",
    "seed",
    "data_file",
    "data_file_sha256",
    "feature_columns",
    "stage_size",
    "train_sessions",
    "test_sessions",
    "train_nll",
    "train_nll_per_query",
    "test_nll",
    "test_nll_per_query",
    "test_hit1",
    "test_hit2",
    "test_hit3",
    "test_booked_queries",
    "test_outside_queries",
    "max_probability_sum_error",
    "min_observed_probability",
    "runtime_seconds",
    "convergence_status",
    "converged",
    "iterations",
    "max_iterations",
    "hit_iteration_cap",
    "final_loglik_gain",
    "final_relative_loglik_gain",
    "monotonicity_violations",
    "worst_loglik_decrease",
    "inner_optimizer_failures",
    "optimizer_exit_flag",
    "created_utc",
)

INTEGER_COLUMNS = (
    "site",
    "trial",
    "seed",
    "stage_size",
    "train_sessions",
    "test_sessions",
    "test_booked_queries",
    "test_outside_queries",
    "iterations",
    "max_iterations",
    "monotonicity_violations",
    "inner_optimizer_failures",
)
BOOLEAN_COLUMNS = ("converged", "hit_iteration_cap")
FINITE_DECIMAL_COLUMNS = (
    "train_nll",
    "train_nll_per_query",
    "test_nll",
    "test_nll_per_query",
    "test_hit1",
    "test_hit2",
    "test_hit3",
    "max_probability_sum_error",
    "min_observed_probability",
    "runtime_seconds",
    "worst_loglik_decrease",
)
OPTIONAL_DECIMAL_COLUMNS = (
    "final_loglik_gain",
    "final_relative_loglik_gain",
    "optimizer_exit_flag",
)
SUMMARY_METRICS = (
    "train_nll",
    "train_nll_per_query",
    "test_nll",
    "test_nll_per_query",
    "test_hit1",
    "test_hit2",
    "test_hit3",
    "runtime_seconds",
)
MANUSCRIPT_METRICS = (
    ("test_nll", "NLL", "lower"),
    ("test_hit1", "1-hit", "higher"),
    ("test_hit2", "2-hit", "higher"),
    ("test_hit3", "3-hit", "higher"),
)
ALLOWED_STATUSES = {
    "converged",
    "optimizer_not_converged",
    "hit_cap",
    "monotonicity_violation",
    "inner_optimizer_failure",
    "stopped_not_converged",
}
OUTPUT_FILENAMES = (
    "expedia_site_summary.csv",
    "expedia_predictive_table.tex",
    "expedia_fit_diagnostics.csv",
    "expedia_fit_diagnostics.json",
    "expedia_claims.json",
)


class ValidationError(ValueError):
    """Raised when the canonical CSV violates the frozen protocol."""


@dataclass(frozen=True)
class ValidatedResults:
    """A sorted, fully validated canonical result grid."""

    rows: tuple[Mapping[str, Any], ...]
    source_path: Path
    source_sha256: str


def _key_text(row: Mapping[str, Any]) -> str:
    return f"site={row.get('site')},trial={row.get('trial')},model={row.get('model_id')}"


def _parse_int(value: str, field: str, row_number: int) -> int:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValidationError(
            f"Row {row_number}: {field} is not numeric: {value!r}."
        ) from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValidationError(
            f"Row {row_number}: {field} must be a finite integer, got {value!r}."
        )
    return int(number)


def _parse_bool(value: str, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValidationError(
        f"Row {row_number}: {field} must be 0/1 or true/false, got {value!r}."
    )


def _parse_decimal(
    value: str, field: str, row_number: int, *, optional: bool = False
) -> Decimal | None:
    normalized = value.strip()
    if optional and normalized.lower() in {"", "nan"}:
        return None
    try:
        number = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValidationError(
            f"Row {row_number}: {field} is not numeric: {value!r}."
        ) from exc
    if not number.is_finite():
        raise ValidationError(
            f"Row {row_number}: {field} must be finite, got {value!r}."
        )
    return number


def _nearly_equal(left: Decimal, right: Decimal, tolerance: Decimal) -> bool:
    scale = max(Decimal(1), abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


def _validate_created_utc(value: str, row_number: int) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(
            f"Row {row_number}: created_utc is not an ISO-8601 timestamp: {value!r}."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValidationError(
            f"Row {row_number}: created_utc must explicitly use UTC: {value!r}."
        )


def _expected_seed(site: int, trial: int, model_id: str) -> int:
    return SEED_BASE + 1000 * site + 10 * trial + MODELS.index(model_id) + 1


def _expected_status(row: Mapping[str, Any]) -> str:
    if row["model_id"] != "imrcs":
        return "converged" if row["converged"] else "optimizer_not_converged"
    if row["monotonicity_violations"] > 0:
        return "monotonicity_violation"
    if row["inner_optimizer_failures"] > 0:
        return "inner_optimizer_failure"
    if row["hit_iteration_cap"]:
        return "hit_cap"
    if row["converged"]:
        return "converged"
    return "stopped_not_converged"


def _validate_row(row: Mapping[str, Any], row_number: int) -> None:
    key = _key_text(row)
    if row["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValidationError(
            f"Row {row_number} ({key}): unexpected schema_version "
            f"{row['schema_version']!r}."
        )
    if row["site"] not in SITES or row["trial"] not in TRIALS:
        raise ValidationError(f"Row {row_number} ({key}): key is outside the frozen grid.")
    if row["model_id"] not in MODELS:
        raise ValidationError(f"Row {row_number} ({key}): unknown model_id.")
    if row["seed"] != _expected_seed(row["site"], row["trial"], row["model_id"]):
        raise ValidationError(f"Row {row_number} ({key}): seed violates the frozen key rule.")
    if not row["data_file"]:
        raise ValidationError(f"Row {row_number} ({key}): data_file is empty.")
    if re.fullmatch(r"[0-9a-f]{64}", row["data_file_sha256"]) is None:
        raise ValidationError(f"Row {row_number} ({key}): invalid data SHA-256.")
    if row["feature_columns"] != EXPECTED_FEATURE_COLUMNS:
        raise ValidationError(f"Row {row_number} ({key}): features must be 8:20.")
    if row["stage_size"] != EXPECTED_STAGE_SIZE:
        raise ValidationError(f"Row {row_number} ({key}): stage size must be 10.")
    if row["train_sessions"] != EXPECTED_TRAIN_SESSIONS:
        raise ValidationError(f"Row {row_number} ({key}): train_sessions must be 6000.")
    if row["test_sessions"] != EXPECTED_TEST_SESSIONS:
        raise ValidationError(f"Row {row_number} ({key}): test_sessions must be 4000.")

    for metric in ("train_nll", "train_nll_per_query", "test_nll", "test_nll_per_query"):
        if row[metric] < 0:
            raise ValidationError(f"Row {row_number} ({key}): {metric} must be nonnegative.")
    if not _nearly_equal(
        row["train_nll_per_query"],
        row["train_nll"] / Decimal(row["train_sessions"]),
        Decimal("1e-11"),
    ):
        raise ValidationError(f"Row {row_number} ({key}): inconsistent train NLL per query.")
    if not _nearly_equal(
        row["test_nll_per_query"],
        row["test_nll"] / Decimal(row["test_sessions"]),
        Decimal("1e-11"),
    ):
        raise ValidationError(f"Row {row_number} ({key}): inconsistent test NLL per query.")

    hits = (row["test_hit1"], row["test_hit2"], row["test_hit3"])
    if any(hit < 0 or hit > 1 for hit in hits):
        raise ValidationError(f"Row {row_number} ({key}): hit rates must lie in [0,1].")
    if not (hits[0] <= hits[1] <= hits[2]):
        raise ValidationError(f"Row {row_number} ({key}): hit@k must be nondecreasing in k.")
    booked = row["test_booked_queries"]
    outside = row["test_outside_queries"]
    if booked <= 0 or outside < 0 or booked + outside != row["test_sessions"]:
        raise ValidationError(f"Row {row_number} ({key}): invalid booked/outside counts.")
    for index, hit in enumerate(hits, start=1):
        implied_count = hit * Decimal(booked)
        if abs(implied_count - implied_count.to_integral_value()) > Decimal("1e-8"):
            raise ValidationError(
                f"Row {row_number} ({key}): hit@{index} is inconsistent with booked count."
            )

    if not (0 <= row["max_probability_sum_error"] <= PROBABILITY_TOLERANCE):
        raise ValidationError(
            f"Row {row_number} ({key}): probability-sum error exceeds "
            f"{PROBABILITY_TOLERANCE}."
        )
    if not (0 < row["min_observed_probability"] <= 1):
        raise ValidationError(
            f"Row {row_number} ({key}): observed probability must lie in (0,1]."
        )
    if row["runtime_seconds"] < 0:
        raise ValidationError(f"Row {row_number} ({key}): runtime must be nonnegative.")
    if row["iterations"] < 0 or row["max_iterations"] <= 0:
        raise ValidationError(f"Row {row_number} ({key}): invalid iteration counts.")
    if row["monotonicity_violations"] < 0 or row["inner_optimizer_failures"] < 0:
        raise ValidationError(f"Row {row_number} ({key}): negative diagnostic count.")
    if row["worst_loglik_decrease"] > 0:
        raise ValidationError(
            f"Row {row_number} ({key}): worst_loglik_decrease must be nonpositive."
        )
    if row["convergence_status"] not in ALLOWED_STATUSES:
        raise ValidationError(
            f"Row {row_number} ({key}): unknown convergence status "
            f"{row['convergence_status']!r}."
        )
    expected_status = _expected_status(row)
    if row["convergence_status"] != expected_status:
        raise ValidationError(
            f"Row {row_number} ({key}): status/diagnostics mismatch; expected "
            f"{expected_status!r}."
        )
    if row["converged"] and row["hit_iteration_cap"]:
        raise ValidationError(f"Row {row_number} ({key}): converged fit cannot hit cap.")

    if row["model_id"] == "imrcs":
        if row["max_iterations"] != 100 or row["iterations"] > 100:
            raise ValidationError(
                f"Row {row_number} ({key}): IM-RCS must use the 100-iteration cap."
            )
        if row["hit_iteration_cap"] != (row["iterations"] >= row["max_iterations"]):
            raise ValidationError(
                f"Row {row_number} ({key}): IM-RCS cap flag disagrees with iterations."
            )
        if row["iterations"] > 0 and (
            row["final_loglik_gain"] is None
            or row["final_relative_loglik_gain"] is None
        ):
            raise ValidationError(
                f"Row {row_number} ({key}): IM-RCS final likelihood gains are missing."
            )
    else:
        if row["max_iterations"] != 500 or row["hit_iteration_cap"]:
            raise ValidationError(
                f"Row {row_number} ({key}): direct model diagnostic protocol mismatch."
            )
        if row["monotonicity_violations"] != 0 or row["inner_optimizer_failures"] != 0:
            raise ValidationError(
                f"Row {row_number} ({key}): direct model has EM-only failures."
            )
        exit_flag = row["optimizer_exit_flag"]
        if exit_flag is None or exit_flag != exit_flag.to_integral_value():
            raise ValidationError(
                f"Row {row_number} ({key}): direct optimizer exit flag is missing/invalid."
            )
        if row["converged"] != (exit_flag > 0):
            raise ValidationError(
                f"Row {row_number} ({key}): convergence flag disagrees with exit flag."
            )


def validate_results_csv(path: str | Path) -> ValidatedResults:
    """Load and strictly validate a canonical 200-row result CSV."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ValidationError(f"Canonical CSV does not exist: {source_path}")
    source_sha256 = _sha256_file(source_path)
    parsed_rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError("Canonical CSV has no header.")
        if tuple(reader.fieldnames) != CANONICAL_COLUMNS:
            raise ValidationError(
                "Canonical CSV columns/order do not match "
                f"{RESULT_SCHEMA_VERSION}.\nExpected: {CANONICAL_COLUMNS}\n"
                f"Found: {tuple(reader.fieldnames)}"
            )
        for row_number, raw in enumerate(reader, start=2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValidationError(f"Row {row_number}: malformed CSV row.")
            parsed: dict[str, Any] = dict(raw)
            for field in INTEGER_COLUMNS:
                parsed[field] = _parse_int(raw[field], field, row_number)
            for field in BOOLEAN_COLUMNS:
                parsed[field] = _parse_bool(raw[field], field, row_number)
            for field in FINITE_DECIMAL_COLUMNS:
                parsed[field] = _parse_decimal(raw[field], field, row_number)
            for field in OPTIONAL_DECIMAL_COLUMNS:
                parsed[field] = _parse_decimal(
                    raw[field], field, row_number, optional=True
                )
            _validate_created_utc(parsed["created_utc"], row_number)
            _validate_row(parsed, row_number)
            parsed_rows.append(parsed)

    if len(parsed_rows) != 200:
        raise ValidationError(
            f"Canonical CSV must contain exactly 200 rows; found {len(parsed_rows)}."
        )
    expected_keys = {
        (site, trial, model_id)
        for site in SITES
        for trial in TRIALS
        for model_id in MODELS
    }
    actual_keys = [
        (row["site"], row["trial"], row["model_id"]) for row in parsed_rows
    ]
    actual_key_set = set(actual_keys)
    if len(actual_key_set) != len(actual_keys):
        duplicates = sorted(key for key, count in Counter(actual_keys).items() if count > 1)
        raise ValidationError(f"Duplicate canonical keys: {duplicates}")
    missing = sorted(expected_keys - actual_key_set)
    extra = sorted(actual_key_set - expected_keys)
    if missing or extra:
        raise ValidationError(f"Canonical key mismatch; missing={missing}, extra={extra}.")

    pair_groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in parsed_rows:
        pair_groups.setdefault((row["site"], row["trial"]), []).append(row)
    for pair, group in sorted(pair_groups.items()):
        hashes = {row["data_file_sha256"] for row in group}
        paths = {row["data_file"] for row in group}
        booked_counts = {row["test_booked_queries"] for row in group}
        outside_counts = {row["test_outside_queries"] for row in group}
        if len(group) != 4 or len(hashes) != 1:
            raise ValidationError(
                f"Site/trial {pair} must have four models sharing exactly one data hash."
            )
        if len(paths) != 1:
            raise ValidationError(f"Site/trial {pair} has inconsistent data paths.")
        if len(booked_counts) != 1 or len(outside_counts) != 1:
            raise ValidationError(
                f"Site/trial {pair} has model-dependent test outcome counts."
            )

    model_order = {model_id: index for index, model_id in enumerate(MODELS)}
    parsed_rows.sort(
        key=lambda row: (row["site"], row["trial"], model_order[row["model_id"]])
    )
    return ValidatedResults(tuple(parsed_rows), source_path, source_sha256)


def _mean(values: Sequence[Decimal]) -> Decimal:
    if not values:
        raise ValueError("Cannot average an empty sequence.")
    return sum(values, Decimal(0)) / Decimal(len(values))


def _sample_sd(values: Sequence[Decimal], mean: Decimal) -> Decimal:
    if len(values) < 2:
        raise ValueError("Sample SD requires at least two values.")
    with localcontext() as context:
        context.prec = 50
        variance = sum((value - mean) ** 2 for value in values) / Decimal(
            len(values) - 1
        )
        return variance.sqrt()


def _plain_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _sd_decimal(value: Decimal) -> str:
    return format(value, ".17g")


def _optional_decimal(value: Decimal | None) -> str:
    return "NaN" if value is None else _plain_decimal(value)


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for site in SITES:
        for model_id in MODELS:
            group = [
                row
                for row in rows
                if row["site"] == site and row["model_id"] == model_id
            ]
            if len(group) != 10:
                raise AssertionError("Validated grid unexpectedly lost a trial.")
            summary: dict[str, Any] = {
                "site": site,
                "model_id": model_id,
                "model_label": MODEL_LABELS[model_id],
                "trials": len(group),
                "converged_fits": sum(bool(row["converged"]) for row in group),
                "nonconverged_fits": sum(not bool(row["converged"]) for row in group),
            }
            for metric in SUMMARY_METRICS:
                values = [row[metric] for row in group]
                metric_mean = _mean(values)
                summary[f"{metric}_mean"] = metric_mean
                summary[f"{metric}_sd"] = _sample_sd(values, metric_mean)
            summaries.append(summary)
    return summaries


def _summary_header() -> tuple[str, ...]:
    fields = [
        "site",
        "model_id",
        "model_label",
        "trials",
        "converged_fits",
        "nonconverged_fits",
    ]
    for metric in SUMMARY_METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_sd"))
    return tuple(fields)


def _format_summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    formatted: dict[str, Any] = {}
    for field in _summary_header():
        value = row[field]
        if field.endswith("_mean"):
            formatted[field] = _plain_decimal(value)
        elif field.endswith("_sd"):
            formatted[field] = _sd_decimal(value)
        else:
            formatted[field] = value
    return formatted


def _best_models(
    summaries: Sequence[Mapping[str, Any]], site: int, metric: str, direction: str
) -> tuple[str, ...]:
    values = {
        row["model_id"]: row[f"{metric}_mean"]
        for row in summaries
        if row["site"] == site
    }
    best_value = min(values.values()) if direction == "lower" else max(values.values())
    return tuple(model_id for model_id in MODELS if values[model_id] == best_value)


def _latex_number(metric: str, value: Decimal) -> str:
    if metric == "test_nll":
        return str(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return format(value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP), ".4f")


def _latex_table(
    summaries: Sequence[Mapping[str, Any]], *, publishable: bool
) -> str:
    lookup = {
        (row["site"], row["model_id"]): row for row in summaries
    }
    gate = "PASSED" if publishable else "BLOCKED"
    lines = [
        "% Generated deterministically from expedia_four_model_results.csv.",
        f"% Publication gate: {gate}. All 200 fits, including nonconverged fits, are retained.",
        "\\begin{table}[htb!]",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\centering",
        "\\begin{tabular}{c|c|cccc}",
        "\\hline",
        "Site & Measure & SQ-MNL & NS-MNL & IML & IM-RCS \\\\",
        "\\hline",
    ]
    for site in SITES:
        for metric_index, (metric, label, direction) in enumerate(MANUSCRIPT_METRICS):
            best = set(_best_models(summaries, site, metric, direction))
            cells: list[str] = []
            for model_id in MODELS:
                value = lookup[(site, model_id)][f"{metric}_mean"]
                cell = _latex_number(metric, value)
                if model_id in best:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            site_cell = f"\\multirow{{4}}{{*}}{{{site}}}" if metric_index == 0 else ""
            lines.append(f"{site_cell} & {label} & " + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    lines.extend(
        [
            "\\end{tabular}",
            "\\vspace{.4cm}",
            "\\caption{Average predictive performance of the choice models on the Expedia data, best model per row in bold.}",
            "\\label{Tab:compare_expedia_nll}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


DIAGNOSTIC_COLUMNS = (
    "site",
    "trial",
    "model_id",
    "seed",
    "data_file_sha256",
    "convergence_status",
    "converged",
    "iterations",
    "max_iterations",
    "hit_iteration_cap",
    "final_loglik_gain",
    "final_relative_loglik_gain",
    "monotonicity_violations",
    "worst_loglik_decrease",
    "inner_optimizer_failures",
    "optimizer_exit_flag",
    "runtime_seconds",
    "max_probability_sum_error",
    "min_observed_probability",
    "test_nll",
    "test_nll_per_query",
    "test_hit1",
    "test_hit2",
    "test_hit3",
    "probability_audit_passed",
)


def _diagnostic_record(row: Mapping[str, Any], *, json_mode: bool) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for field in DIAGNOSTIC_COLUMNS:
        if field == "probability_audit_passed":
            value: Any = True
        else:
            value = row[field]
        if isinstance(value, Decimal):
            record[field] = _plain_decimal(value)
        elif value is None:
            record[field] = None if json_mode else "NaN"
        elif isinstance(value, bool):
            record[field] = value if json_mode else int(value)
        else:
            record[field] = value
    return record


def _diagnostics_json(results: ValidatedResults) -> dict[str, Any]:
    rows = results.rows
    status_counts = Counter(row["convergence_status"] for row in rows)
    nonconverged = [row for row in rows if not row["converged"]]
    max_sum_error = max(row["max_probability_sum_error"] for row in rows)
    min_observed = min(row["min_observed_probability"] for row in rows)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "fit_diagnostics",
        "source_csv": str(results.source_path),
        "source_csv_sha256": results.source_sha256,
        "required_fit_count": 200,
        "included_fit_count": len(rows),
        "dropped_fit_count": 0,
        "converged_fit_count": len(rows) - len(nonconverged),
        "nonconverged_fit_count": len(nonconverged),
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "probability_audit": {
            "passed": True,
            "tolerance": _plain_decimal(PROBABILITY_TOLERANCE),
            "maximum_probability_sum_error": _plain_decimal(max_sum_error),
            "minimum_observed_probability": _plain_decimal(min_observed),
        },
        "fits": [_diagnostic_record(row, json_mode=True) for row in rows],
    }


def _claim_records(summaries: Sequence[Mapping[str, Any]], publishable: bool) -> list[dict[str, Any]]:
    lookup = {
        (row["site"], row["model_id"]): row for row in summaries
    }
    claims: list[dict[str, Any]] = []
    for site in SITES:
        for metric, label, direction in MANUSCRIPT_METRICS:
            means = {
                model_id: lookup[(site, model_id)][f"{metric}_mean"]
                for model_id in MODELS
            }
            best_ids = list(_best_models(summaries, site, metric, direction))
            best_value = means[best_ids[0]]
            benchmark_values = [means[model_id] for model_id in MODELS if model_id != "imrcs"]
            best_benchmark = (
                min(benchmark_values) if direction == "lower" else max(benchmark_values)
            )
            if direction == "lower":
                favorable_gap = best_benchmark - means["imrcs"]
            else:
                favorable_gap = means["imrcs"] - best_benchmark
            relative_gap = (
                favorable_gap / best_benchmark if best_benchmark != 0 else None
            )
            claims.append(
                {
                    "site": site,
                    "metric": metric,
                    "manuscript_label": label,
                    "direction": direction,
                    "model_means": {
                        model_id: _plain_decimal(means[model_id]) for model_id in MODELS
                    },
                    "best_model_ids": best_ids,
                    "best_mean": _plain_decimal(best_value),
                    "imrcs_is_best": "imrcs" in best_ids,
                    "imrcs_favorable_gap_vs_best_benchmark": _plain_decimal(
                        favorable_gap
                    ),
                    "imrcs_relative_gap_vs_best_benchmark": (
                        None if relative_gap is None else _plain_decimal(relative_gap)
                    ),
                    "publication_eligible": publishable,
                }
            )
    return claims


def _claims_json(
    results: ValidatedResults, summaries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    nonconverged = [row for row in results.rows if not row["converged"]]
    publishable = len(nonconverged) == 0
    reasons = [] if publishable else ["required_fit_not_converged"]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "descriptive_metric_claims",
        "source_csv": str(results.source_path),
        "source_csv_sha256": results.source_sha256,
        "publishable": publishable,
        "publication_gate": {
            "passed": publishable,
            "reason_codes": reasons,
            "required_fit_count": 200,
            "converged_fit_count": 200 - len(nonconverged),
            "nonconverged_fit_count": len(nonconverged),
            "nonconverged_keys": [
                {
                    "site": row["site"],
                    "trial": row["trial"],
                    "model_id": row["model_id"],
                    "status": row["convergence_status"],
                }
                for row in nonconverged
            ],
        },
        "aggregation": {
            "trials_per_site_model": 10,
            "standard_deviation_ddof": 1,
            "nonconverged_rows_retained": True,
            "table_rounding": {"test_nll": "nearest integer", "hit_rates": "4 decimals"},
            "best_model_selection_uses_unrounded_means": True,
        },
        "metric_claims": _claim_records(summaries, publishable),
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _csv_text(fieldnames: Sequence[str], records: Iterable[Mapping[str, Any]]) -> str:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for record in records:
        writer.writerow(record)
    return buffer.getvalue()


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_reports(
    input_csv: str | Path, output_dir: str | Path
) -> dict[str, str]:
    """Validate ``input_csv`` and atomically write all deterministic reports.

    Validation completes before the first output is replaced.  The returned
    mapping contains SHA-256 hashes for the five generated artifacts.
    """

    results = validate_results_csv(input_csv)
    summaries = _summary_rows(results.rows)
    publishable = all(bool(row["converged"]) for row in results.rows)

    summary_content = _csv_text(
        _summary_header(), (_format_summary_row(row) for row in summaries)
    )
    table_content = _latex_table(summaries, publishable=publishable)
    diagnostic_csv_content = _csv_text(
        DIAGNOSTIC_COLUMNS,
        (_diagnostic_record(row, json_mode=False) for row in results.rows),
    )
    diagnostic_json_content = _json_text(_diagnostics_json(results))
    claims_content = _json_text(_claims_json(results, summaries))
    contents = {
        "expedia_site_summary.csv": summary_content,
        "expedia_predictive_table.tex": table_content,
        "expedia_fit_diagnostics.csv": diagnostic_csv_content,
        "expedia_fit_diagnostics.json": diagnostic_json_content,
        "expedia_claims.json": claims_content,
    }

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILENAMES:
        _atomic_write_text(destination / filename, contents[filename])
    return {
        filename: _sha256_file(destination / filename) for filename in OUTPUT_FILENAMES
    }


def _default_paths() -> tuple[Path, Path]:
    sequential_root = Path(__file__).resolve().parents[1]
    results_dir = sequential_root / "results"
    return results_dir / "expedia_four_model_results.csv", results_dir


def main(argv: Sequence[str] | None = None) -> int:
    default_input, default_output = _default_paths()
    parser = argparse.ArgumentParser(
        description="Validate the canonical Expedia CSV and generate audited reports."
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args(argv)
    try:
        hashes = generate_reports(args.input, args.output_dir)
    except ValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 2
    print("Validated exactly 200 canonical Expedia fits; no rows were dropped.")
    for filename in OUTPUT_FILENAMES:
        print(f"{filename}  sha256={hashes[filename]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
