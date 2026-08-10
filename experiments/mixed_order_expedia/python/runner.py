"""Run and analyze the Expedia mixed-order counterfactual."""

from __future__ import annotations

from common.python.runtime import matlab, require_fit_grid, run_module


EXPERIMENT = "mixed_order_expedia"


def run_counterfactual() -> None:
    """Run the complete retained Expedia mixed-order design."""

    require_fit_grid("expedia")
    matlab(EXPERIMENT, "run_mixed_order(false);")
    run_module(
        "experiments.mixed_order_expedia.python.report",
        "--mode",
        "full",
    )
