from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from common.python import update_expedia_manuscript as updater
from common.python.predictive_reporting import (
    MARKET_ORDER,
    MODELS,
    RESULT_METRICS,
    TRIALS,
    aggregate as aggregate_predictive,
    latex_table as predictive_latex_table,
)
from experiments.mixed_order_expedia.python import report as mixed_report


class UpdateExpediaManuscriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.predictive = self.root / "experiments" / "predictive_expedia" / "results"
        self.mixed = self.root / "experiments" / "mixed_order_expedia" / "results"
        self.figures = self.root / "new" / "figures"
        self.predictive.mkdir(parents=True)
        self.mixed.mkdir(parents=True)
        self.figures.mkdir(parents=True)
        self.main_tex = self.root / "main_msom_v2.tex"
        self.appendix = self.root / "omitted analysis_v2.tex"
        self.main_tex.write_text(
            "\\documentclass{article}\n\\input{omitted analysis_v2}\n",
            encoding="utf-8",
        )
        self.appendix.write_text(self._appendix_template(), encoding="utf-8")
        self._write_predictive_generation()
        self.raw = self._write_mixed_generation()
        self._patches = ExitStack()
        self._patches.enter_context(patch.object(updater, "ROOT", self.root))
        self._patches.enter_context(patch.object(updater, "APPENDIX", self.appendix))
        self._patches.enter_context(patch.object(updater, "MAIN_TEX", self.main_tex))
        self._patches.enter_context(
            patch.object(updater, "PREDICTIVE", self.predictive)
        )
        self._patches.enter_context(patch.object(updater, "MIXED", self.mixed))
        self._patches.enter_context(
            patch.object(updater, "PAPER_FIGURES", self.figures)
        )
        self._patches.enter_context(
            patch.object(mixed_report, "EXPERIMENT_ROOT", self.mixed.parent)
        )

    def tearDown(self) -> None:
        self._patches.close()
        self.temporary.cleanup()

    @staticmethod
    def _appendix_template() -> str:
        return r"""
As shown in Table~\ref{Tab:compare_expedia}, this is an old predictive claim.
The two metrics emphasize different aspects of prediction.

\begin{table}[htb]
\begin{tabular}{c}
old predictive table
\end{tabular}
\caption{Predictive results.}
\label{Tab:compare_expedia}
\end{table}

Table~\ref{tab: realistic_orders} shows that this is an old mixed-order claim.
Figure~\ref{fig:realistic_orders_b} illustrates an old Dirichlet claim.
The last column of Table~\ref{tab: realistic_orders} is an old benchmark claim.
% The order-agnostic heuristic also improves in an old hidden comment.

\begin{table}[htb]
\begin{tabular}{c}
old mixed-order table
\end{tabular}
\caption{Mixed-order results.}
\label{tab: realistic_orders}
\end{table}
""".lstrip()

    def _write_predictive_generation(self) -> None:
        rows: list[dict[str, object]] = []
        for market_index, market in enumerate(MARKET_ORDER["expedia"]):
            for trial in TRIALS:
                for model_index, model in enumerate(MODELS):
                    base = 0.15 + 0.002 * market_index + 0.001 * model_index
                    rows.append(
                        {
                            "Market": market,
                            "Trial": trial,
                            "Model": model,
                            "NLL": (
                                np.nan
                                if model in {"CMNL", "SCM"}
                                else 1000 + 10 * market_index + trial + model_index
                            ),
                            "RMSE": base + trial / 100_000,
                            "WeightedSquaredError": base / 2 + trial / 100_000,
                            "MAPE": base / 3 + trial / 100_000,
                            "Hit1": 0.20 + 0.002 * model_index,
                            "Hit2": 0.40 + 0.002 * model_index,
                            "Hit3": 0.60 + 0.002 * model_index,
                        }
                    )
        trials = pd.DataFrame(
            rows, columns=["Market", "Trial", "Model", *RESULT_METRICS]
        )
        trials.to_csv(
            self.predictive / updater.PREDICTIVE_TRIALS,
            index=False,
            lineterminator="\n",
        )
        means = aggregate_predictive(trials, "expedia")
        means.to_csv(
            self.predictive / updater.PREDICTIVE_MEANS,
            index=False,
            lineterminator="\n",
        )
        (self.predictive / updater.PREDICTIVE_TABLE).write_text(
            predictive_latex_table(means, "expedia"), encoding="utf-8"
        )

    def _write_mixed_generation(self) -> Path:
        sites = np.repeat(
            np.asarray(mixed_report.SITES, dtype=int),
            len(mixed_report.FULL_TRIALS) * mixed_report.FULL_QUERIES_PER_TASK,
        )
        trials = np.tile(
            np.repeat(
                np.asarray(mixed_report.FULL_TRIALS, dtype=int),
                mixed_report.FULL_QUERIES_PER_TASK,
            ),
            len(mixed_report.SITES),
        )
        query = np.tile(
            np.arange(1, mixed_report.FULL_QUERIES_PER_TASK + 1, dtype=int),
            len(mixed_report.SITES) * len(mixed_report.FULL_TRIALS),
        )
        site_index = np.searchsorted(np.asarray(mixed_report.SITES), sites)
        base = 0.70 + site_index / 1000 + trials / 10_000 + (query % 11) / 100_000
        frame = pd.DataFrame(
            {
                "site": sites,
                "trial": trials,
                "srch_id": sites * 1_000_000 + trials * 1_000 + query,
                "n_prod": 20 + query % 5,
                "card_SqL": 10 + query % 3,
                "UB": 100.0 + query / 1000,
                "sampling_seed": (
                    mixed_report.SAMPLING_SEED_BASE + 100_000 * sites + 1_000 * trials
                ),
                "evaluation_seed": (
                    mixed_report.EVALUATION_SEED_BASE
                    + 100_000 * sites
                    + 1_000 * trials
                    + query
                ),
            }
        )
        for index, column in enumerate(mixed_report.H_COLUMNS):
            frame[column] = base + index * 0.015
        for index, column in enumerate(mixed_report.G_COLUMNS):
            frame[column] = base + 0.03 + index * 0.015
        for index, column in enumerate(mixed_report.ALL_COLUMNS):
            frame[column] = base - 0.04 + index * 0.015
        frame = frame.loc[:, mixed_report.EXPECTED_COLUMNS]
        raw_path = self.mixed / updater.MIXED_RAW
        frame.to_csv(raw_path, index=False, lineterminator="\n")

        overall = mixed_report.aggregate(frame, pooled=True)
        by_site = {
            int(site): mixed_report.aggregate(
                frame.loc[frame["site"].eq(site)], pooled=False
            )
            for site in mixed_report.SITES
        }
        mixed_report.build_summary(overall, by_site).to_csv(
            self.mixed / updater.MIXED_SUMMARY,
            index=False,
            na_rep="NA",
        )
        (self.mixed / updater.MIXED_TABLE).write_text(
            mixed_report.build_table(overall, by_site), encoding="utf-8"
        )
        source_bytes = {
            "a": b"%PDF-1.4\n/CreationDate (D:20260810010101)\nfigure-a\n%%EOF\n",
            "b": b"%PDF-1.4\n/ModDate (D:20260810010102)\nfigure-b\n%%EOF\n",
        }
        output_records: list[dict[str, object]] = []
        for name, content in source_bytes.items():
            source = self.mixed / f"fig_mixed_order_{name}.pdf"
            source.write_bytes(content)
            output_records.append({"path": str(source), "bytes": len(content)})
            (self.figures / f"mixed_order_{name}.pdf").write_bytes(
                b"%PDF-1.4\nstale\n%%EOF\n"
            )
        manifest = {
            "schema_version": 3,
            "artifact_type": "mixed_order_analysis",
            "mode": "full",
            "sites": list(mixed_report.SITES),
            "trials": list(mixed_report.FULL_TRIALS),
            "queries_per_task": mixed_report.FULL_QUERIES_PER_TASK,
            "expected_rows": len(frame),
            "actual_rows": len(frame),
            "orderings": list(mixed_report.ORDERINGS),
            "dirichlet_seed": mixed_report.DIRICHLET_SEED,
            "dirichlet_draws": mixed_report.DIRICHLET_DRAWS,
            "headline": updater._mixed_headline(overall),
            "outputs": output_records,
        }
        (self.mixed / updater.MIXED_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return raw_path

    def _targets(self) -> tuple[Path, Path, Path]:
        return (
            self.appendix,
            self.figures / "mixed_order_a.pdf",
            self.figures / "mixed_order_b.pdf",
        )

    def _target_bytes(self) -> dict[Path, bytes]:
        return {path: path.read_bytes() for path in self._targets()}

    def test_rejects_cross_generation_predictive_means(self) -> None:
        path = self.predictive / updater.PREDICTIVE_MEANS
        means = pd.read_csv(path, dtype={"Market": str})
        means.loc[0, "RMSE"] += 0.01
        means.to_csv(path, index=False, lineterminator="\n")
        before = self._target_bytes()
        with self.assertRaisesRegex(ValueError, "Predictive means"):
            updater.run(write=True)
        self.assertEqual(before, self._target_bytes())

    def test_rejects_cross_generation_mixed_manifest(self) -> None:
        path = self.mixed / updater.MIXED_MANIFEST
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["headline"]["H_uniform"] += 0.01
        path.write_text(json.dumps(manifest), encoding="utf-8")
        before = self._target_bytes()
        with self.assertRaisesRegex(ValueError, "manifest headline"):
            updater.run(write=True)
        self.assertEqual(before, self._target_bytes())

    def test_failure_after_second_commit_rolls_back_all_targets(self) -> None:
        before = self._target_bytes()
        real_commit = updater._commit_stage
        calls = 0

        def fail_after_second(stage: Path, target: Path) -> None:
            nonlocal calls
            real_commit(stage, target)
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("injected failure after second commit")

        with patch.object(updater, "_commit_stage", side_effect=fail_after_second):
            with self.assertRaises(KeyboardInterrupt):
                updater.run(write=True)
        self.assertEqual(before, self._target_bytes())

    def test_postverify_failure_rolls_back_all_targets(self) -> None:
        before = self._target_bytes()
        with patch.object(
            updater,
            "_postverify_generation",
            side_effect=SystemExit("injected postverify failure"),
        ):
            with self.assertRaises(SystemExit):
                updater.run(write=True)
        self.assertEqual(before, self._target_bytes())

    def test_input_drift_during_commit_is_detected_and_rolled_back(self) -> None:
        before = self._target_bytes()
        real_commit = updater._commit_stage
        drifted = False

        def drift_then_commit(stage: Path, target: Path) -> None:
            nonlocal drifted
            if not drifted:
                with self.raw.open("a", encoding="utf-8") as handle:
                    handle.write("\n")
                drifted = True
            real_commit(stage, target)

        with patch.object(updater, "_commit_stage", side_effect=drift_then_commit):
            with self.assertRaisesRegex(ValueError, "input changed"):
                updater.run(write=True)
        self.assertEqual(before, self._target_bytes())

    def test_write_is_idempotent_and_preserves_current_anchors(self) -> None:
        updater.run(write=True)
        text = self.appendix.read_text(encoding="utf-8")
        self.assertIn(r"the $0.375$ worst-case guarantee", text)
        self.assertIn(r"$W$-heuristic ratios", text)
        self.assertIn("pooled normalized ratio", text)
        identities = {path: updater._identity(path) for path in self._targets()}
        updater.run(write=True)
        updater.run(write=False)
        self.assertEqual(
            identities, {path: updater._identity(path) for path in self._targets()}
        )

    def test_duplicate_anchor_aborts_before_publication(self) -> None:
        token = r"As shown in Table~\ref{Tab:compare_expedia}"
        self.appendix.write_text(
            self.appendix.read_text(encoding="utf-8") + "\n" + token + "\n",
            encoding="utf-8",
        )
        figures_before = {
            path: path.read_bytes() for path in self._targets() if path != self.appendix
        }
        with self.assertRaisesRegex(ValueError, "Expected one start token"):
            updater.run(write=True)
        self.assertEqual(
            figures_before,
            {path: path.read_bytes() for path in figures_before},
        )


if __name__ == "__main__":
    unittest.main()
