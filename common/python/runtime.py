"""Shared process, path, and fitted-grid helpers for experiment runners."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]


def environment() -> dict[str, str]:
    """Return the deterministic subprocess environment used by all runners."""

    environment_variables = os.environ.copy()
    environment_variables["PYTHONPATH"] = (
        str(ROOT) + os.pathsep + environment_variables.get("PYTHONPATH", "")
    )
    environment_variables["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment_variables


def run(command: Iterable[str]) -> None:
    """Run one repository command and fail immediately on a nonzero exit."""

    resolved = [str(argument) for argument in command]
    print("+", " ".join(resolved), flush=True)
    subprocess.run(
        resolved,
        cwd=ROOT,
        env=environment(),
        check=True,
    )


def run_module(module: str, *arguments: object) -> None:
    """Run a repository Python module with the pinned current interpreter."""

    run([sys.executable, "-m", module, *(str(argument) for argument in arguments)])


def matlab(experiment: str, expression: str) -> None:
    """Run a MATLAB entry point from one explicitly named experiment folder."""

    experiment_dir = ROOT / "experiments" / experiment / "matlab"
    if not experiment_dir.is_dir():
        raise FileNotFoundError(
            f"MATLAB experiment directory is missing: {experiment_dir}"
        )
    matlab_dir = str(experiment_dir).replace("'", "''")
    run(["matlab", "-batch", f"addpath('{matlab_dir}'); {expression}"])


def require_fit_grid(dataset: str) -> None:
    """Require the exact unified five-market by ten-trial fitted grid."""

    markets: dict[str, tuple[object, ...]] = {
        "expedia": (5, 14, 15, 24, 32),
        "trivago": ("BR", "US", "JP", "UK", "DE"),
    }
    if dataset not in markets:
        raise ValueError(f"Unknown unified-fit dataset: {dataset}")

    fit_dir = ROOT / "opt_data" / dataset
    expected = {
        f"fit_{market}_{trial}.mat"
        for market in markets[dataset]
        for trial in range(1, 11)
    }
    if not fit_dir.is_dir() or fit_dir.is_symlink():
        raise FileNotFoundError(f"Unified fit directory is invalid: {fit_dir}")

    entries = list(fit_dir.glob("*.mat"))
    actual = {path.name for path in entries}
    invalid = [path.name for path in entries if not path.is_file() or path.is_symlink()]
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected or invalid:
        details = []
        if missing:
            details.append(f"missing {len(missing)}")
        if unexpected:
            details.append(f"unexpected {len(unexpected)}")
        if invalid:
            details.append(f"nonregular {len(invalid)}")
        raise FileNotFoundError(
            f"Unified {dataset} fit grid is not the exact 5x10 canonical set ("
            + ", ".join(details)
            + ")."
        )
