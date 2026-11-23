#!/usr/bin/env python3
"""Canonical command-line entry point for reproduction stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.counterfactual_trivago.python import (
    runner as trivago_counterfactual,
)
from experiments.mixed_order_expedia.python import (
    runner as expedia_mixed_order,
)
from experiments.predictive_expedia.python import (
    runner as expedia_predictive,
)
from experiments.predictive_trivago.python import (
    runner as trivago_predictive,
)
from experiments.synthetic.python import runner as synthetic_experiment


def evaluate_local() -> None:
    """Regenerate every local result stage without refitting models."""

    trivago_predictive.evaluate("trivago")
    trivago_predictive.tables("trivago")
    expedia_predictive.evaluate_local_predictive()
    trivago_counterfactual.run_counterfactual()
    expedia_mixed_order.run_counterfactual()
    synthetic_experiment.run_experiment()


def build_parser() -> argparse.ArgumentParser:
    """Build the stable public reproduction command tree."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    stage_help = {
        "preprocess": "replace the canonical Trivago processed-data generation",
        "fit": "refit and replace all six canonical Trivago model artifacts",
        "evaluate": "evaluate the retained Trivago six-model fit grid",
        "tables": "validate a Trivago evaluation bundle and publish paper tables",
        "fit-mmnl": "refit the MMNL fields in all 50 unified Expedia fits",
        "fit-expedia": "fit all six Expedia models into an isolated staging grid",
        "evaluate-mmnl": "evaluate MMNL from the 50 unified Expedia fits",
        "evaluate-local": "regenerate all result stages without refitting",
        "synthetic": "run and publish the deterministic synthetic experiment",
        "expedia-local-predictive": "evaluate and merge archived 250 + MMNL 50 tasks",
        "counterfactuals": "run one explicitly named counterfactual dataset",
    }
    for stage in ("preprocess", "fit", "evaluate", "tables"):
        child = subparsers.add_parser(
            stage,
            help=stage_help[stage],
            description=stage_help[stage],
        )
        child.add_argument("--dataset", choices=["trivago"], default="trivago")
    for stage in ("fit-mmnl", "evaluate-mmnl"):
        child = subparsers.add_parser(
            stage,
            help=stage_help[stage],
            description=stage_help[stage],
        )
        child.add_argument("--dataset", choices=["expedia"], default="expedia")
    expedia_fit_parser = subparsers.add_parser(
        "fit-expedia",
        help=stage_help["fit-expedia"],
        description=stage_help["fit-expedia"],
    )
    expedia_fit_parser.add_argument("--dataset", choices=["expedia"], default="expedia")
    expedia_fit_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="explicit corrected Expedia MAT staging directory",
    )
    expedia_fit_parser.add_argument(
        "--fit-dir",
        type=Path,
        required=True,
        help="isolated output directory outside the canonical data/fit trees",
    )
    expedia_fit_parser.add_argument(
        "--markets",
        nargs="+",
        choices=expedia_predictive.EXPEDIA_MARKETS,
        default=list(expedia_predictive.EXPEDIA_MARKETS),
    )
    expedia_fit_parser.add_argument(
        "--trials",
        nargs="+",
        type=int,
        choices=range(1, 11),
        default=list(range(1, 11)),
    )
    expedia_fit_parser.add_argument("--shard-index", type=int, default=1)
    expedia_fit_parser.add_argument("--shard-count", type=int, default=1)
    expedia_fit_parser.add_argument(
        "--resume",
        action="store_true",
        help="validate and skip already completed staging tasks",
    )
    for stage in (
        "evaluate-local",
        "synthetic",
        "expedia-local-predictive",
    ):
        subparsers.add_parser(
            stage,
            help=stage_help[stage],
            description=stage_help[stage],
        )
    counterfactual_parser = subparsers.add_parser(
        "counterfactuals",
        help=stage_help["counterfactuals"],
        description=stage_help["counterfactuals"],
    )
    counterfactual_parser.add_argument(
        "--dataset",
        choices=["expedia", "trivago"],
        required=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.stage == "evaluate-local":
        evaluate_local()
    elif args.stage == "preprocess":
        trivago_predictive.preprocess(args.dataset)
    elif args.stage == "fit":
        trivago_predictive.fit(args.dataset)
    elif args.stage == "fit-mmnl":
        expedia_predictive.fit_mmnl(args.dataset)
    elif args.stage == "fit-expedia":
        expedia_predictive.fit_all_models(
            args.dataset,
            data_dir=args.data_dir,
            fit_dir=args.fit_dir,
            markets=args.markets,
            trials=args.trials,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            resume=args.resume,
        )
    elif args.stage == "evaluate":
        trivago_predictive.evaluate(args.dataset)
    elif args.stage == "evaluate-mmnl":
        expedia_predictive.evaluate_mmnl(args.dataset)
    elif args.stage == "tables":
        trivago_predictive.tables(args.dataset)
    elif args.stage == "expedia-local-predictive":
        expedia_predictive.evaluate_local_predictive()
    elif args.stage == "counterfactuals" and args.dataset == "trivago":
        trivago_counterfactual.run_counterfactual()
    elif args.stage == "counterfactuals":
        expedia_mixed_order.run_counterfactual()
    elif args.stage == "synthetic":
        synthetic_experiment.run_experiment()


if __name__ == "__main__":
    main()
