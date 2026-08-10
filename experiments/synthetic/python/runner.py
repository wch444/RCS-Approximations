"""Run and publish the deterministic synthetic experiment."""

from __future__ import annotations

import tempfile
from pathlib import Path

from common.python.runtime import ROOT, matlab, run_module


EXPERIMENT = "synthetic"
RESULTS_DIR = ROOT / "experiments" / EXPERIMENT / "results"


def run_experiment() -> None:
    """Generate the fixed synthetic grid and publish its paper table."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=RESULTS_DIR, prefix=".synthetic-stage-"
    ) as temporary:
        stage = Path(temporary)
        matlab_stage = str(stage).replace("'", "''")
        matlab(
            EXPERIMENT,
            "run_synthetic_experiments(1000,20260808,true,"
            f"[10,15,20],[3,5,8],'{matlab_stage}');",
        )
        run_module(
            "experiments.synthetic.python.report",
            "--trials",
            stage / "synthetic_trial_level.csv",
            "--summary",
            stage / "synthetic_summary.csv",
            "--output-dir",
            RESULTS_DIR,
        )
