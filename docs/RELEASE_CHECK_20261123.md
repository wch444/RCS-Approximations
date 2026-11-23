# Public release check — 2026-11-23

The computational code was synchronized from the local research checkout's
`44ddaf3c63872001adb5ed6edeac63659bcb1817` version. All 135 tracked files under
`common/`, `experiments/`, `sequential_rcs/`, and the computational figures,
plus the root dispatcher and environment files, match their local versions
byte for byte. The public repository retains its 100 bundled empirical fits
and its synthetic reference outputs. Every final model parameter array in
the bundled fits matches the corresponding local fit.

Validation:

- `python -m pytest -q`: 34 passed.
- Recomputed five experimental tables from retained local result rows:
  all 414 displayed numeric result cells match the current manuscript targets.
- Recomputed 16 deterministic predictive model/task combinations, each using
  all 4,000 test queries in its selected task: maximum metric difference
  `4.996e-16`.
- Recomputed 36 synthetic instances spanning all 18 design cells: maximum
  revenue difference `5.329e-15`.
- Checked SCM and CMNL probabilities on 48 sampled sessions per model against
  independent numerical integration. Maximum absolute probability differences
  were `0.00451981` and `0.02141137`, respectively, with exact repeatability
  under fixed seeds.

Trivago counterfactual reporting uses the paper's equally weighted average of
session-level revenue ratios. It does not take a ratio of pooled revenues.
The regression test in `tests/test_counterfactual_reporting.py` distinguishes
these two estimands.

This is sampled evaluation from retained matrices and fits, not a complete
independent rerun from raw data through fresh estimation. The separate
theoretical numerical upper-bound construction curve is outside this
experimental check; its producer was not found in the local code.

The existing public-target repository's complete fetched branch and PR-ref
history was checked before publication. No manuscript TeX source was present;
the only historical `.tex` file was the synthetic experiment's generated
table fragment. Private manuscript sources and the private repository's
Git history were not imported.
