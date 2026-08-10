"""Run the Trivago revenue counterfactual and publish its summary."""

from __future__ import annotations

from common.python.runtime import ROOT, matlab, require_fit_grid, run_module


EXPERIMENT = "counterfactual_trivago"
RESULTS_DIR = ROOT / "experiments" / EXPERIMENT / "results"


def run_counterfactual() -> None:
    """Evaluate all retained Trivago fits under the revenue counterfactual."""

    require_fit_grid("trivago")
    matlab(EXPERIMENT, "revenue_misspec_trivago;")
    run_module(
        "experiments.counterfactual_trivago.python.report",
        "--input",
        RESULTS_DIR / "counterfactual_trivago.csv",
        "--output-dir",
        RESULTS_DIR,
    )
