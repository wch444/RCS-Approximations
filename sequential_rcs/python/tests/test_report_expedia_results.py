"""Focused contract tests for the canonical Expedia report generator."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from sequential_rcs.python import report_expedia_results as report


def _hash(site: int, trial: int) -> str:
    return hashlib.sha256(f"{site}:{trial}".encode()).hexdigest()


def _rows(*, imrcs_converged: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for site in report.SITES:
        for trial in report.TRIALS:
            booked = 3000
            for model_index, model_id in enumerate(report.MODELS, start=1):
                train_nll = Decimal(12000 + site + 10 * trial + model_index)
                test_nll = Decimal(8000 + site + 10 * trial + model_index)
                if model_id == "imrcs" and not imrcs_converged:
                    status = "hit_cap"
                    converged = "0"
                    iterations = "100"
                    max_iterations = "100"
                    hit_cap = "1"
                    final_gain = "0.1"
                    final_relative_gain = "0.00001"
                    exit_flag = "NaN"
                elif model_id == "imrcs":
                    status = "converged"
                    converged = "1"
                    iterations = "20"
                    max_iterations = "100"
                    hit_cap = "0"
                    final_gain = "0.0001"
                    final_relative_gain = "0.0000001"
                    exit_flag = "NaN"
                else:
                    status = "converged"
                    converged = "1"
                    iterations = "12"
                    max_iterations = "500"
                    hit_cap = "0"
                    final_gain = "NaN"
                    final_relative_gain = "NaN"
                    exit_flag = "1"
                rows.append(
                    {
                        "schema_version": report.RESULT_SCHEMA_VERSION,
                        "site": str(site),
                        "trial": str(trial),
                        "model_id": model_id,
                        "seed": str(report._expected_seed(site, trial, model_id)),
                        "data_file": f"/data/10k_sample_{site}_{trial}.mat",
                        "data_file_sha256": _hash(site, trial),
                        "feature_columns": "8:20",
                        "stage_size": "10",
                        "train_sessions": "6000",
                        "test_sessions": "4000",
                        "train_nll": str(train_nll),
                        "train_nll_per_query": str(train_nll / Decimal(6000)),
                        "test_nll": str(test_nll),
                        "test_nll_per_query": str(test_nll / Decimal(4000)),
                        "test_hit1": "0.1",
                        "test_hit2": "0.2",
                        "test_hit3": "0.3",
                        "test_booked_queries": str(booked),
                        "test_outside_queries": "1000",
                        "max_probability_sum_error": "1e-15",
                        "min_observed_probability": "1e-7",
                        "runtime_seconds": str(5 + model_index),
                        "convergence_status": status,
                        "converged": converged,
                        "iterations": iterations,
                        "max_iterations": max_iterations,
                        "hit_iteration_cap": hit_cap,
                        "final_loglik_gain": final_gain,
                        "final_relative_loglik_gain": final_relative_gain,
                        "monotonicity_violations": "0",
                        "worst_loglik_decrease": "0",
                        "inner_optimizer_failures": "0",
                        "optimizer_exit_flag": exit_flag,
                        "created_utc": "2026-08-16T00:00:00Z",
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=report.CANONICAL_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


class ExpediaReportTests(unittest.TestCase):
    def test_complete_grid_retains_nonconverged_and_blocks_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_csv = root / "canonical.csv"
            output_dir = root / "reports"
            _write_csv(input_csv, _rows())

            hashes = report.generate_reports(input_csv, output_dir)

            self.assertEqual(set(hashes), set(report.OUTPUT_FILENAMES))
            with (output_dir / "expedia_site_summary.csv").open(newline="") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 20)
            self.assertTrue(all(row["trials"] == "10" for row in summary))
            imrcs = [row for row in summary if row["model_id"] == "imrcs"]
            self.assertTrue(all(row["nonconverged_fits"] == "10" for row in imrcs))

            with (output_dir / "expedia_fit_diagnostics.csv").open(newline="") as handle:
                diagnostics = list(csv.DictReader(handle))
            self.assertEqual(len(diagnostics), 200)
            self.assertEqual(
                sum(row["converged"] == "0" for row in diagnostics), 50
            )

            claims = json.loads((output_dir / "expedia_claims.json").read_text())
            self.assertFalse(claims["publishable"])
            self.assertEqual(
                claims["publication_gate"]["reason_codes"],
                ["required_fit_not_converged"],
            )
            self.assertEqual(claims["publication_gate"]["nonconverged_fit_count"], 50)
            self.assertEqual(len(claims["publication_gate"]["nonconverged_keys"]), 50)
            self.assertEqual(len(claims["metric_claims"]), 20)

            table = (output_dir / "expedia_predictive_table.tex").read_text()
            self.assertIn("% Publication gate: BLOCKED.", table)
            self.assertIn(r"\multirow{4}{*}{5} & NLL", table)

    def test_all_converged_grid_passes_publication_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_csv = root / "canonical.csv"
            output_dir = root / "reports"
            _write_csv(input_csv, _rows(imrcs_converged=True))
            report.generate_reports(input_csv, output_dir)
            claims = json.loads((output_dir / "expedia_claims.json").read_text())
            self.assertTrue(claims["publishable"])
            self.assertEqual(claims["publication_gate"]["reason_codes"], [])

    def test_outputs_are_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_csv = root / "canonical.csv"
            first = root / "first"
            second = root / "second"
            _write_csv(input_csv, list(reversed(_rows())))
            first_hashes = report.generate_reports(input_csv, first)
            second_hashes = report.generate_reports(input_csv, second)
            self.assertEqual(first_hashes, second_hashes)
            for filename in report.OUTPUT_FILENAMES:
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_missing_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.csv"
            rows = _rows()
            rows.pop()
            _write_csv(path, rows)
            with self.assertRaisesRegex(report.ValidationError, "exactly 200"):
                report.validate_results_csv(path)

    def test_site_trial_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.csv"
            rows = _rows()
            rows[0]["data_file_sha256"] = "f" * 64
            _write_csv(path, rows)
            with self.assertRaisesRegex(report.ValidationError, "sharing exactly one data hash"):
                report.validate_results_csv(path)

    def test_nonfinite_common_metric_and_probability_failure_are_rejected(self) -> None:
        mutations = (
            ("test_nll", "NaN", "must be finite"),
            ("max_probability_sum_error", "1e-7", "probability-sum error"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "canonical.csv"
                rows = _rows()
                rows[0][field] = value
                _write_csv(path, rows)
                with self.assertRaisesRegex(report.ValidationError, message):
                    report.validate_results_csv(path)

    def test_status_diagnostic_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.csv"
            rows = _rows()
            rows[3]["convergence_status"] = "converged"
            _write_csv(path, rows)
            with self.assertRaisesRegex(report.ValidationError, "status/diagnostics mismatch"):
                report.validate_results_csv(path)


if __name__ == "__main__":
    unittest.main()
