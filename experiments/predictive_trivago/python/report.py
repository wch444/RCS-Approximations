#!/usr/bin/env python3
"""Validate Trivago predictive results and publish the paper-format table."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from common.python.predictive_reporting import (
    RESULT_METRICS,
    aggregate,
    paper_values,
    render_latex,
    validate_tasks,
)


TRIVAGO_TRIALS_FILE = "predictive_trivago_trials.csv"
TRIVAGO_MEANS_FILE = "predictive_trivago_means.csv"
TRIVAGO_PAPER_FILE = "predictive_trivago_paper.csv"
TRIVAGO_TEX_FILE = "predictive_trivago_table.tex"
TRIVAGO_TABLE_LOCK_FILE = ".predictive_trivago_tables.lock"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def _read_csv_stable(
    path: Path, **kwargs: Any
) -> tuple[pd.DataFrame, tuple[int, int, int, int]]:
    before = _file_identity(path)
    frame = pd.read_csv(path, **kwargs)
    if before != _file_identity(path):
        raise ValueError(f"CSV changed while being read: {path}")
    return frame, before


def _validate_evaluator_means(observed: pd.DataFrame, expected: pd.DataFrame) -> None:
    columns = ["Market", "Model", *RESULT_METRICS]
    if observed.columns.tolist() != columns:
        raise ValueError(f"Evaluator means columns must be exactly {columns}.")
    checked = observed.copy()
    checked["Market"] = checked["Market"].astype(str)
    checked["Model"] = checked["Model"].astype(str)
    checked[RESULT_METRICS] = checked[RESULT_METRICS].apply(
        pd.to_numeric, errors="coerce"
    )
    try:
        pd.testing.assert_frame_equal(
            checked,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=5e-15,
            atol=5e-12,
        )
    except AssertionError as error:
        raise ValueError(
            "Evaluator means do not equal the trial aggregation."
        ) from error


@contextmanager
def _table_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / TRIVAGO_TABLE_LOCK_FILE
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise ValueError(f"Table lock is not a regular file: {lock_path}")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _restore(
    names: list[str],
    output_dir: Path,
    backup_dir: Path,
    had_original: dict[str, bool],
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


def publish_trivago_tables(trials_path: Path, output_dir: Path) -> dict[str, Path]:
    """Validate a complete evaluation bundle and atomically publish its table."""

    trials_path = Path(trials_path).resolve()
    output_dir = Path(output_dir).resolve()
    _require(
        trials_path.name == TRIVAGO_TRIALS_FILE,
        "Trial filename is not canonical.",
    )
    _require(
        trials_path.parent == output_dir,
        "Trials and table outputs must share a directory.",
    )
    _require(
        output_dir.is_dir() and not output_dir.is_symlink(),
        "Output directory is invalid.",
    )
    means_path = output_dir / TRIVAGO_MEANS_FILE
    trials, trial_state = _read_csv_stable(trials_path, dtype={"Market": str})
    evaluator_means, means_state = _read_csv_stable(means_path, dtype={"Market": str})
    checked = validate_tasks(trials, "trivago")
    means = aggregate(checked, "trivago")
    _validate_evaluator_means(evaluator_means, means)
    paper = paper_values(means, "trivago")
    prepared = {
        TRIVAGO_PAPER_FILE: paper.to_csv(
            index=False, float_format="%.4f", lineterminator="\n"
        ),
        TRIVAGO_TEX_FILE: render_latex(paper, "trivago"),
    }
    names = list(prepared)
    states = {trials_path: trial_state, means_path: means_state}
    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix=".trivago-table-generation-"
    ) as staging:
        stage = Path(staging)
        for name, value in prepared.items():
            (stage / name).write_text(value, encoding="utf-8")
        with _table_lock(output_dir):
            for path, state in states.items():
                _require(
                    _file_identity(path) == state,
                    f"Evaluation input changed: {path}",
                )
            for name in names:
                target = output_dir / name
                _require(
                    not target.is_symlink()
                    and (not target.exists() or target.is_file()),
                    f"Invalid target: {target}",
                )
            with tempfile.TemporaryDirectory(
                dir=output_dir, prefix=".trivago-table-backup-"
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
                    for path, state in states.items():
                        _require(
                            _file_identity(path) == state,
                            f"Evaluation input changed: {path}",
                        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["trivago"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = publish_trivago_tables(args.input, args.output_dir)
    print(f"Published Trivago table: {paths[TRIVAGO_TEX_FILE]}")


if __name__ == "__main__":
    main()
