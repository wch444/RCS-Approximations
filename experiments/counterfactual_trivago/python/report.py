#!/usr/bin/env python3
"""Build the Trivago table of mean session-level gains from task aggregates."""

from __future__ import annotations

import argparse
import fcntl
import io
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


MODELS = ["SCM", "MNL", "MMNL", "CMNL", "SeqINV", "NSMNL"]
MARKETS = ["BR", "US", "JP", "UK", "DE"]
TRIALS = list(range(1, 11))
QUERIES_PER_TASK = 4_000
RATIO_COLUMNS = ["MeanGRatio", "MeanWRatio"]
TASK_COLUMNS = [
    "Model",
    "Market",
    "Trial",
    "NQueries",
    "NUniqueQueries",
    "PlatformRevenue",
    "GRevenue",
    "WRevenue",
    *RATIO_COLUMNS,
]
SUMMARY_COLUMNS = ["Market", "Model", "GImprovementPct", "WImprovementPct", "NQueries"]
SUMMARY_FILE = "counterfactual_trivago_summary.csv"
TABLE_FILE = "counterfactual_trivago_table.tex"
REPORT_LOCK_FILE = ".counterfactual-trivago-report.lock"


def mean_session_improvement(frame: pd.DataFrame, ratio: str) -> float:
    """Return the equally weighted mean percentage improvement per session."""

    weighted_ratio = (frame[ratio] * frame["NQueries"]).sum() / frame[
        "NQueries"
    ].sum()
    return 100.0 * (weighted_ratio - 1.0)


def _validated_tasks(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != TASK_COLUMNS:
        raise ValueError(f"Columns must be exactly {TASK_COLUMNS}")

    task_columns = ["Model", "Market", "Trial"]
    if frame[task_columns].isna().any().any():
        raise ValueError("Task identifiers cannot be missing.")

    checked = frame.copy()
    checked["Model"] = checked["Model"].astype(str)
    checked["Market"] = checked["Market"].astype(str)
    numeric_trials = pd.to_numeric(checked["Trial"], errors="coerce")
    trial_values = numeric_trials.to_numpy(dtype=float)
    if (
        not np.isfinite(trial_values).all()
        or not np.equal(trial_values, np.floor(trial_values)).all()
    ):
        raise ValueError("Trials must be integer values 1 through 10.")
    checked["Trial"] = numeric_trials.astype(int)

    unknown_models = sorted(set(checked["Model"]) - set(MODELS))
    unknown_markets = sorted(set(checked["Market"]) - set(MARKETS))
    unknown_trials = sorted(set(checked["Trial"]) - set(TRIALS))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    if unknown_markets:
        raise ValueError(f"Unknown Trivago markets: {unknown_markets}")
    if unknown_trials:
        raise ValueError(f"Unknown trials: {unknown_trials}")

    duplicates = checked.duplicated(task_columns, keep=False)
    if duplicates.any():
        tasks = (
            checked.loc[duplicates, task_columns].drop_duplicates().to_dict("records")
        )
        raise ValueError(f"Duplicate task rows: {tasks}")

    expected = {
        (model, market, trial)
        for model in MODELS
        for market in MARKETS
        for trial in TRIALS
    }
    observed = set(checked[task_columns].itertuples(index=False, name=None))
    missing = sorted(expected - observed)
    if len(checked) != len(expected) or missing or observed - expected:
        raise ValueError(
            "Expected the exact 300-row model/market/trial grid; "
            f"first missing tasks are {missing[:5]}."
        )

    numeric_columns = [
        "NQueries",
        "NUniqueQueries",
        "PlatformRevenue",
        "GRevenue",
        "WRevenue",
        *RATIO_COLUMNS,
    ]
    numeric = checked[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Query counts, revenues, and ratios must all be finite.")
    checked[numeric_columns] = numeric

    counts = checked[["NQueries", "NUniqueQueries"]].to_numpy(dtype=float)
    if not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Query counts must be integer values.")
    if not (
        checked["NQueries"].eq(QUERIES_PER_TASK).all()
        and checked["NUniqueQueries"].eq(QUERIES_PER_TASK).all()
    ):
        raise ValueError("Every task must contain exactly 4000 unique queries.")
    if (checked["PlatformRevenue"] <= 0).any():
        raise ValueError("PlatformRevenue must be strictly positive for every task.")
    if (checked[["GRevenue", "WRevenue", *RATIO_COLUMNS]] < 0).any().any():
        raise ValueError("Counterfactual revenues and mean ratios cannot be negative.")
    return checked


def build(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    frame = _validated_tasks(frame)
    rows = []
    for market in [*MARKETS, "Pooled"]:
        for model in MODELS:
            block = frame[frame["Model"] == model]
            if market != "Pooled":
                block = block[block["Market"] == market]
            rows.append(
                {
                    "Market": market,
                    "Model": model,
                    "GImprovementPct": mean_session_improvement(
                        block, "MeanGRatio"
                    ),
                    "WImprovementPct": mean_session_improvement(
                        block, "MeanWRatio"
                    ),
                    "NQueries": int(block["NQueries"].sum()),
                }
            )
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    lines = [
        r"\begin{tabular}{c|cc|cc|cc|cc|cc|cc}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{SCM (ref.)} & \multicolumn{2}{c|}{MNL} "
        r"& \multicolumn{2}{c|}{MMNL} & \multicolumn{2}{c|}{CMNL} "
        r"& \multicolumn{2}{c|}{SeqINV} & \multicolumn{2}{c}{NSMNL} \\",
        r"Market & G & W & G & W & G & W & G & W & G & W & G & W \\",
        r"\midrule",
    ]
    for market in MARKETS:
        block = summary[summary["Market"] == market].set_index("Model")
        cells = []
        for model in MODELS:
            cells.extend(
                [
                    f"{block.loc[model, 'GImprovementPct']:+.2f}",
                    f"{block.loc[model, 'WImprovementPct']:+.2f}",
                ]
            )
        lines.append(market + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    block = summary[summary["Market"] == "Pooled"].set_index("Model")
    cells = []
    for model in MODELS:
        cells.extend(
            [
                rf"\textbf{{{block.loc[model, 'GImprovementPct']:+.2f}}}",
                rf"\textbf{{{block.loc[model, 'WImprovementPct']:+.2f}}}",
            ]
        )
    lines.extend(
        ["Pooled & " + " & ".join(cells) + r" \\", r"\bottomrule", r"\end{tabular}"]
    )
    return summary, "\n".join(lines) + "\n"


def _regular_output_directory(output_dir: Path) -> Path:
    output_dir = Path(os.path.abspath(Path(output_dir).expanduser()))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = output_dir.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"Output directory must be regular: {output_dir}")
    return output_dir


def _require_regular_target(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"Publication target must be a regular file: {path}")


@contextmanager
def _report_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / REPORT_LOCK_FILE
    _require_regular_target(lock_path)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Report lock is not a regular file: {lock_path}")
        with os.fdopen(descriptor, "a+b") as handle:
            descriptor = -1
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                current = lock_path.lstat()
                if (current.st_dev, current.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise ValueError(f"Report lock changed while opened: {lock_path}")
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _restore_outputs(
    names: list[str], output_dir: Path, backup_dir: Path, had_original: dict[str, bool]
) -> list[str]:
    errors: list[str] = []
    for name in names:
        target = output_dir / name
        try:
            if had_original[name]:
                os.replace(backup_dir / name, target)
            elif target.exists() or target.is_symlink():
                target.unlink()
        except BaseException as error:  # pragma: no cover - fatal I/O
            errors.append(f"{name}: {error}")
    return errors


def write_outputs(summary: pd.DataFrame, latex: str, output_dir: Path) -> None:
    """Publish the summary and table together, with rollback on failure."""

    if summary.columns.tolist() != SUMMARY_COLUMNS or len(summary) != 36:
        raise ValueError("Counterfactual summary does not have the canonical schema.")
    summary_text = summary.to_csv(index=False, lineterminator="\n")
    prepared = {SUMMARY_FILE: summary_text, TABLE_FILE: latex}
    names = list(prepared)
    output_dir = _regular_output_directory(output_dir)

    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix=".counterfactual-trivago-generation-"
    ) as staging, _report_lock(output_dir):
        staging_dir = Path(staging)
        for name, value in prepared.items():
            path = staging_dir / name
            path.write_text(value, encoding="utf-8")
            if path.read_text(encoding="utf-8") != value:
                raise ValueError(f"Prepared output changed: {name}")
        for name in names:
            _require_regular_target(output_dir / name)
        with tempfile.TemporaryDirectory(
            dir=output_dir, prefix=".counterfactual-trivago-backup-"
        ) as backup:
            backup_dir = Path(backup)
            had_original = {name: (output_dir / name).is_file() for name in names}
            for name in names:
                if had_original[name]:
                    shutil.copy2(output_dir / name, backup_dir / name)
            try:
                for name in names:
                    os.replace(staging_dir / name, output_dir / name)
                committed = pd.read_csv(
                    io.StringIO((output_dir / SUMMARY_FILE).read_text())
                )
                expected = pd.read_csv(io.StringIO(summary_text))
                pd.testing.assert_frame_equal(
                    committed, expected, check_dtype=False, check_exact=True
                )
                if (output_dir / TABLE_FILE).read_text(encoding="utf-8") != latex:
                    raise ValueError("Committed LaTeX table changed.")
            except BaseException as publication_error:
                rollback_errors = _restore_outputs(
                    names, output_dir, backup_dir, had_original
                )
                if rollback_errors:
                    publication_error.add_note(
                        "Rollback incomplete: " + " | ".join(rollback_errors)
                    )
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary, latex = build(pd.read_csv(args.input))
    write_outputs(summary, latex, args.output_dir)


if __name__ == "__main__":
    main()
