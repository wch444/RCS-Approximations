"""Guard the paper's equally weighted session-level revenue improvement."""

import unittest

import pandas as pd

from experiments.counterfactual_trivago.python import report


class CounterfactualReportingTests(unittest.TestCase):
    def test_improvement_averages_sessions_without_revenue_weighting(self):
        rows = []
        for model in report.MODELS:
            for market in report.MARKETS:
                for trial in report.TRIALS:
                    # Each task has 4,000 sessions. High-baseline tasks also
                    # have higher gains, so pooling revenue gives a different
                    # estimand from the paper's equally weighted sessions.
                    low_baseline = trial % 2 == 1
                    baseline = 4_000.0 if low_baseline else 40_000.0
                    gain_g = 1.2 if low_baseline else 2.0
                    gain_w = 1.1 if low_baseline else 1.3
                    rows.append(
                        [model, market, trial, 4_000, 4_000, baseline,
                         baseline * gain_g, baseline * gain_w, gain_g, gain_w]
                    )
        tasks = pd.DataFrame(rows, columns=report.TASK_COLUMNS)
        summary, _ = report.build(tasks)

        self.assertEqual(len(summary), 36)
        for row in summary.itertuples(index=False):
            self.assertAlmostEqual(row.GImprovementPct, 60.0)
            self.assertAlmostEqual(row.WImprovementPct, 20.0)


if __name__ == "__main__":
    unittest.main()
