"""Run Expedia fitting and predictive evaluation stages."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path

from common.python.runtime import (
    ROOT,
    matlab,
    require_fit_grid,
    run_module,
)


EXPERIMENT = "predictive_expedia"
RESULTS_DIR = ROOT / "experiments" / EXPERIMENT / "results"
EXPEDIA_MARKETS = ("5", "14", "15", "24", "32")


def _require_expedia(dataset: str) -> None:
    if dataset != "expedia":
        raise ValueError("The Expedia workflow supports only Expedia.")


def fit_mmnl(dataset: str = "expedia") -> None:
    """Refit only the MMNL fields in the unified Expedia fitted grid."""

    _require_expedia(dataset)
    require_fit_grid("expedia")
    matlab(EXPERIMENT, "fit_mmnl_only('expedia');")


def fit_all_models(
    dataset: str = "expedia",
    *,
    data_dir: Path,
    fit_dir: Path,
    markets: Sequence[str] = EXPEDIA_MARKETS,
    trials: Sequence[int] = tuple(range(1, 11)),
    shard_index: int = 1,
    shard_count: int = 1,
    resume: bool = False,
) -> None:
    """Fit a deterministic subset of all six models into a staging folder.

    The canonical data and fit trees are deliberately rejected.  A completed
    50-file staging grid can be promoted only after the separate data, fit,
    and evaluation checks have passed.
    """

    _require_expedia(dataset)
    requested_data_dir = Path(data_dir).expanduser()
    requested_fit_dir = Path(fit_dir).expanduser()
    if requested_data_dir.is_symlink() or requested_fit_dir.is_symlink():
        raise ValueError("Expedia data and fit directories may not be symlinks.")
    resolved_data_dir = requested_data_dir.resolve()
    resolved_fit_dir = requested_fit_dir.resolve()
    canonical_fit_dir = (ROOT / "opt_data" / "expedia").resolve()
    canonical_data_dir = (ROOT / "data" / "processed" / "expedia").resolve()
    if not resolved_data_dir.is_dir():
        raise FileNotFoundError(
            f"The explicit Expedia data directory is missing: {resolved_data_dir}"
        )
    for selected_path in (resolved_data_dir, resolved_fit_dir):
        if any(
            selected_path == protected or _is_within(selected_path, protected)
            for protected in (canonical_data_dir, canonical_fit_dir)
        ):
            raise ValueError(
                "Fresh Expedia data and fits must remain outside the canonical "
                "data/processed/expedia and opt_data/expedia trees."
            )
    if (
        resolved_data_dir == resolved_fit_dir
        or _is_within(resolved_data_dir, resolved_fit_dir)
        or _is_within(resolved_fit_dir, resolved_data_dir)
    ):
        raise ValueError(
            "Expedia data and fit directories must be separate and non-nested."
        )

    selected_markets = tuple(str(market) for market in markets)
    if not selected_markets or len(set(selected_markets)) != len(selected_markets):
        raise ValueError("Expedia markets must be a nonempty unique sequence.")
    unknown_markets = sorted(set(selected_markets) - set(EXPEDIA_MARKETS))
    if unknown_markets:
        raise ValueError(f"Unknown Expedia markets: {unknown_markets}")

    supplied_trials = tuple(trials)
    if not all(
        isinstance(trial, Integral) and not isinstance(trial, bool)
        for trial in supplied_trials
    ):
        raise ValueError("Expedia trials must be integers from 1 through 10.")
    selected_trials = tuple(int(trial) for trial in supplied_trials)
    if (
        not selected_trials
        or len(set(selected_trials)) != len(selected_trials)
        or any(trial < 1 or trial > 10 for trial in selected_trials)
    ):
        raise ValueError("Expedia trials must be unique integers from 1 through 10.")
    if shard_count < 1 or shard_index < 1 or shard_index > shard_count:
        raise ValueError(
            "shard_count must be positive and shard_index must lie in "
            "1,...,shard_count."
        )

    market_literal = "[" + ",".join(f'"{market}"' for market in selected_markets) + "]"
    trial_literal = "[" + ",".join(str(trial) for trial in selected_trials) + "]"
    expression = (
        f"fit_all_models('expedia',{market_literal},{trial_literal},"
        f"'DataDir','{_matlab_quote(resolved_data_dir)}',"
        f"'FitDir','{_matlab_quote(resolved_fit_dir)}',"
        f"'ShardIndex',{shard_index},'ShardCount',{shard_count},"
        f"'Resume',{'true' if resume else 'false'});"
    )
    matlab(EXPERIMENT, expression)


def evaluate_mmnl(dataset: str = "expedia") -> None:
    """Evaluate the MMNL component of the unified Expedia fitted grid."""

    _require_expedia(dataset)
    require_fit_grid("expedia")
    matlab(EXPERIMENT, "evaluate_mmnl_only('expedia');")


def evaluate_local_predictive() -> None:
    """Build the canonical 300-row Expedia predictive-result chain."""

    require_fit_grid("expedia")
    matlab(EXPERIMENT, "evaluate_expedia_archived_models;")
    matlab(EXPERIMENT, "evaluate_mmnl_only('expedia');")
    run_module(
        "experiments.predictive_expedia.python.report",
        "--legacy-trials",
        RESULTS_DIR / "predictive_expedia_legacy_trials.csv",
        "--mmnl-trials",
        RESULTS_DIR / "predictive_expedia_mmnl_trials.csv",
        "--data-dir",
        ROOT / "data" / "processed" / "expedia",
        "--fit-dir",
        ROOT / "opt_data" / "expedia",
        "--output-dir",
        RESULTS_DIR,
    )


def _matlab_quote(path: Path) -> str:
    """Escape one absolute filesystem path for a MATLAB char literal."""

    return str(path).replace("'", "''")


def _is_within(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is strictly nested below ``parent``."""

    return parent in path.parents
