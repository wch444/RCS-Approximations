#!/usr/bin/env python3
"""Validate and atomically publish the canonical synthetic results."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


MODELS = ("CMNL", "SCM")
N_VALUES = (10, 15, 20)
B_VALUES = (3, 5, 8)
N_REPLICATIONS = 1000
MASTER_SEED = 20260808
RATIO_TOLERANCE = 1e-10

TRIAL_COLUMNS = [
    "Model",
    "N",
    "B",
    "Replication",
    "Seed",
    "HeuristicRevenue",
    "OptimalRevenue",
    "Ratio",
]
SUMMARY_COLUMNS = ["Model", "N", "B", "MeanRatio", "MinRatio", "StdRatio"]
PAPER_COLUMNS = [
    "N",
    "B",
    "CMNLMean",
    "CMNLMin",
    "CMNLStd",
    "SCMMean",
    "SCMMin",
    "SCMStd",
]
TRIAL_FILE = "synthetic_trial_level.csv"
SUMMARY_FILE = "synthetic_summary.csv"
PAPER_FILE = "paper_synthetic.csv"
TEX_FILE = "synthetic_table.tex"
LOCK_FILE = ".synthetic-publication.lock"


def task_seed(
    master_seed: int, model_index: int, n: int, b: int, replication: int
) -> int:
    return (
        master_seed
        + model_index * 100_000_000
        + n * 1_000_000
        + b * 10_000
        + replication
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer_column(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    values = numeric.to_numpy(dtype=float)
    _require(
        bool(np.isfinite(values).all() and np.equal(values, np.floor(values)).all()),
        f"{column} must contain finite integers.",
    )
    return numeric.astype(np.int64)


def _exact_columns(frame: pd.DataFrame, expected: list[str], label: str) -> None:
    _require(
        frame.columns.tolist() == expected,
        f"{label} columns must be exactly {expected}.",
    )


def _expected_trial_keys() -> list[tuple[str, int, int, int]]:
    return [
        (model, n, b, replication)
        for model in MODELS
        for n in N_VALUES
        for b in B_VALUES
        for replication in range(1, N_REPLICATIONS + 1)
    ]


def _expected_summary_keys() -> list[tuple[str, int, int]]:
    return [(model, n, b) for model in MODELS for n in N_VALUES for b in B_VALUES]


def _validate_trials(frame: pd.DataFrame) -> pd.DataFrame:
    _exact_columns(frame, TRIAL_COLUMNS, TRIAL_FILE)
    checked = frame.copy()
    checked["Model"] = checked["Model"].astype("string")
    for column in ("N", "B", "Replication", "Seed"):
        checked[column] = _integer_column(checked, column)
    keys = list(
        checked[["Model", "N", "B", "Replication"]].itertuples(index=False, name=None)
    )
    _require(
        keys == _expected_trial_keys(), "Synthetic trial grid/order is not canonical."
    )

    expected_seeds = np.fromiter(
        (
            task_seed(MASTER_SEED, model_index, n, b, replication)
            for model_index, _model in enumerate(MODELS, start=1)
            for n in N_VALUES
            for b in B_VALUES
            for replication in range(1, N_REPLICATIONS + 1)
        ),
        dtype=np.int64,
        count=18_000,
    )
    observed_seeds = checked["Seed"].to_numpy(dtype=np.int64)
    _require(
        np.array_equal(observed_seeds, expected_seeds)
        and len(np.unique(observed_seeds)) == 18_000,
        "Synthetic task seeds are missing, duplicated, or incorrect.",
    )
    for column in ("HeuristicRevenue", "OptimalRevenue", "Ratio"):
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
    numeric = checked[["HeuristicRevenue", "OptimalRevenue", "Ratio"]].to_numpy(float)
    _require(np.isfinite(numeric).all(), "Synthetic trial metrics must be finite.")
    _require(
        bool((checked["HeuristicRevenue"] >= 0).all())
        and bool((checked["OptimalRevenue"] > 0).all())
        and bool((checked["Ratio"] >= -1e-12).all())
        and bool((checked["Ratio"] <= 1 + RATIO_TOLERANCE).all()),
        "Synthetic trial revenues or ratios are outside the feasible range.",
    )
    recomputed = checked["HeuristicRevenue"].to_numpy(float) / checked[
        "OptimalRevenue"
    ].to_numpy(float)
    _require(
        np.allclose(
            checked["Ratio"].to_numpy(float), recomputed, rtol=5e-15, atol=5e-16
        ),
        "Synthetic stored ratios do not match the revenues.",
    )
    return checked


def _recompute_summary(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["Model", "N", "B"], sort=False, observed=True)["Ratio"]
        .agg(MeanRatio="mean", MinRatio="min", StdRatio="std")
        .reset_index()
        .loc[:, SUMMARY_COLUMNS]
    )


def _validate_summary(frame: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    _exact_columns(frame, SUMMARY_COLUMNS, SUMMARY_FILE)
    checked = frame.copy()
    checked["Model"] = checked["Model"].astype("string")
    checked["N"] = _integer_column(checked, "N")
    checked["B"] = _integer_column(checked, "B")
    keys = list(checked[["Model", "N", "B"]].itertuples(index=False, name=None))
    _require(
        keys == _expected_summary_keys(),
        "Synthetic summary grid/order is not canonical.",
    )
    for column in ("MeanRatio", "MinRatio", "StdRatio"):
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
    _require(
        np.isfinite(checked[["MeanRatio", "MinRatio", "StdRatio"]]).all().all(),
        "Summary metrics must be finite.",
    )
    recomputed = _recompute_summary(trials)
    for column in ("MeanRatio", "MinRatio", "StdRatio"):
        _require(
            np.allclose(
                checked[column].to_numpy(float),
                recomputed[column].to_numpy(float),
                rtol=2e-13,
                atol=2e-15,
            ),
            f"Synthetic {column} does not recompute from trial rows.",
        )
    return recomputed


def make_paper_frame(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for n in N_VALUES:
        for b in B_VALUES:
            row: dict[str, float | int] = {"N": n, "B": b}
            for model in MODELS:
                match = summary[
                    summary["Model"].eq(model) & summary["N"].eq(n) & summary["B"].eq(b)
                ]
                _require(
                    len(match) == 1, f"Missing synthetic row for {model}, N={n}, B={b}."
                )
                values = match.iloc[0]
                row[f"{model}Mean"] = 100 * float(values["MeanRatio"])
                row[f"{model}Min"] = 100 * float(values["MinRatio"])
                row[f"{model}Std"] = 100 * float(values["StdRatio"])
            rows.append(row)
    paper = pd.DataFrame(rows, columns=PAPER_COLUMNS)
    numeric = paper[PAPER_COLUMNS[2:]].to_numpy(float)
    _require(
        np.isfinite(numeric).all()
        and bool((numeric >= -1e-10).all())
        and bool((numeric <= 100 + 100 * RATIO_TOLERANCE).all()),
        "Synthetic paper percentages are invalid.",
    )
    return paper


def render_paper_table(paper: pd.DataFrame) -> str:
    _exact_columns(paper, PAPER_COLUMNS, PAPER_FILE)
    _require(
        list(paper[["N", "B"]].itertuples(index=False, name=None))
        == [(n, b) for n in N_VALUES for b in B_VALUES],
        "Synthetic paper grid is not canonical.",
    )
    lines = [
        r"\begin{tabular}{cc|ccc|ccc}",
        r"\toprule",
        r"\multicolumn{2}{c|}{Parameters} & \multicolumn{3}{c|}{CMNL} & \multicolumn{3}{c}{SCM} \\",
        r"$n$ & $b$ & mean & min & std & mean & min & std \\",
        r"\midrule",
    ]
    for values in paper.itertuples(index=False):
        cells = [
            values.CMNLMean,
            values.CMNLMin,
            values.CMNLStd,
            values.SCMMean,
            values.SCMMin,
            values.SCMStd,
        ]
        lines.append(
            f"{values.N} & {values.B} & "
            + " & ".join(f"{cell:.2f}" for cell in cells)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build(frame: pd.DataFrame) -> str:
    _exact_columns(frame, SUMMARY_COLUMNS, SUMMARY_FILE)
    _require(
        list(frame[["Model", "N", "B"]].itertuples(index=False, name=None))
        == _expected_summary_keys(),
        "Synthetic summary grid/order is not canonical.",
    )
    metrics = frame[["MeanRatio", "MinRatio", "StdRatio"]].apply(
        pd.to_numeric, errors="coerce"
    )
    _require(
        np.isfinite(metrics).all().all(), "Synthetic summary metrics must be finite."
    )
    checked = frame.copy()
    checked[["MeanRatio", "MinRatio", "StdRatio"]] = metrics
    return render_paper_table(make_paper_frame(checked))


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def load_stage(
    trial_path: Path, summary_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[Path, tuple[int, int, int, int]]]:
    trial_path = Path(trial_path).resolve()
    summary_path = Path(summary_path).resolve()
    _require(
        trial_path.name == TRIAL_FILE and summary_path.name == SUMMARY_FILE,
        "Synthetic inputs must retain canonical filenames.",
    )
    _require(
        trial_path.parent == summary_path.parent,
        "Synthetic inputs must share one stage.",
    )
    states = {
        trial_path: _file_identity(trial_path),
        summary_path: _file_identity(summary_path),
    }
    trials = _validate_trials(pd.read_csv(trial_path))
    summary = _validate_summary(pd.read_csv(summary_path), trials)
    for path, state in states.items():
        _require(
            _file_identity(path) == state, f"Synthetic input changed while read: {path}"
        )
    return trials, summary, states


@contextmanager
def _publication_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / LOCK_FILE
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise ValueError(f"Publication lock is not a regular file: {lock_path}")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_targets(names: list[str], output_dir: Path) -> None:
    for name in names:
        target = output_dir / name
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            continue
        _require(
            stat.S_ISREG(mode), f"Publication target is not a regular file: {target}"
        )


def _restore(
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
        except BaseException as error:  # pragma: no cover
            errors.append(f"{name}: {error}")
    return errors


def publish_generation(
    trial_path: Path, summary_path: Path, output_dir: Path
) -> dict[str, Path]:
    trial_path = Path(trial_path).resolve()
    summary_path = Path(summary_path).resolve()
    trials, summary, states = load_stage(trial_path, summary_path)
    paper = make_paper_frame(summary)
    prepared = {
        TRIAL_FILE: trial_path.read_text(encoding="utf-8"),
        SUMMARY_FILE: summary_path.read_text(encoding="utf-8"),
        PAPER_FILE: paper.to_csv(
            index=False, float_format="%.17g", lineterminator="\n"
        ),
        TEX_FILE: render_paper_table(paper),
    }
    for path, state in states.items():
        _require(_file_identity(path) == state, f"Synthetic input changed: {path}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _require(
        output_dir.is_dir() and not output_dir.is_symlink(),
        "Output directory is invalid.",
    )
    names = list(prepared)
    _validate_targets(names, output_dir)
    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix=".synthetic-generation-"
    ) as staging:
        stage = Path(staging)
        for name, value in prepared.items():
            (stage / name).write_text(value, encoding="utf-8")
        with _publication_lock(output_dir):
            for path, state in states.items():
                _require(
                    _file_identity(path) == state, f"Synthetic input changed: {path}"
                )
            _validate_targets(names, output_dir)
            with tempfile.TemporaryDirectory(
                dir=output_dir, prefix=".synthetic-backup-"
            ) as backup:
                backup_dir = Path(backup)
                had_original = {name: (output_dir / name).is_file() for name in names}
                for name in names:
                    if had_original[name]:
                        shutil.copy2(output_dir / name, backup_dir / name)
                try:
                    for name in names:
                        os.replace(stage / name, output_dir / name)
                    for name, value in prepared.items():
                        _require(
                            (output_dir / name).read_text(encoding="utf-8") == value,
                            f"Committed output changed: {name}",
                        )
                    _validate_trials(pd.read_csv(output_dir / TRIAL_FILE))
                    _validate_summary(pd.read_csv(output_dir / SUMMARY_FILE), trials)
                except BaseException as publication_error:
                    rollback_errors = _restore(
                        names, output_dir, backup_dir, had_original
                    )
                    if rollback_errors:
                        publication_error.add_note(
                            "Rollback incomplete: " + " | ".join(rollback_errors)
                        )
                    raise
    return {name: output_dir / name for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = publish_generation(args.trials, args.summary, args.output_dir)
    print(f"Published synthetic table: {paths[TEX_FILE]}")


if __name__ == "__main__":
    main()
