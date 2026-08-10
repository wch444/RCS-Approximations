"""Run Trivago preprocessing, fitting, evaluation, and table stages."""

from __future__ import annotations

from common.python.runtime import (
    ROOT,
    matlab,
    require_fit_grid,
    run_module,
)


EXPERIMENT = "predictive_trivago"
RESULTS_DIR = ROOT / "experiments" / EXPERIMENT / "results"


def _require_trivago(dataset: str, stage: str) -> None:
    if dataset != "trivago":
        alternative = (
            "fit-expedia --dataset expedia"
            if stage == "fit"
            else "expedia-local-predictive"
        )
        raise ValueError(
            f"The {stage} stage supports only Trivago. "
            f"Use {alternative} for Expedia."
        )


def preprocess(dataset: str = "trivago") -> None:
    """Replace the canonical Trivago processed-data generation."""

    _require_trivago(dataset, "preprocess")
    run_module(
        "experiments.predictive_trivago.python.preprocess.pipeline",
        "--raw-dir",
        ROOT / "data" / "raw" / "trivago",
        "--output-dir",
        ROOT / "data" / "processed" / "trivago",
    )


def fit(dataset: str = "trivago") -> None:
    """Fit all six Trivago model estimates."""

    _require_trivago(dataset, "fit")
    matlab(EXPERIMENT, "fit_all_models('trivago');")
    require_fit_grid("trivago")


def evaluate(dataset: str = "trivago") -> None:
    """Evaluate the complete Trivago fitted grid."""

    _require_trivago(dataset, "evaluate")
    require_fit_grid("trivago")
    matlab(EXPERIMENT, "evaluate_all_models('trivago');")


def tables(dataset: str = "trivago") -> None:
    """Publish Trivago tables from the complete evaluation output."""

    _require_trivago(dataset, "tables")
    run_module(
        "experiments.predictive_trivago.python.report",
        "--dataset",
        "trivago",
        "--input",
        RESULTS_DIR / "predictive_trivago_trials.csv",
        "--output-dir",
        RESULTS_DIR,
    )
