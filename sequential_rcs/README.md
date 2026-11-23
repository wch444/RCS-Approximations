# Sequential-RCS Expedia reproducibility package

This directory contains the frozen implementation for the revised Expedia
four-model comparison. The manuscript source and restored historical analyses
remain available for audit, but they are not inputs to the new results.

## Frozen experiment protocol

- Sites: `5, 14, 15, 24, 32`; trials: `1:10`.
- Every otherwise eligible query is retained regardless of click count.
- Each trial samples 10,000 query IDs with `random.Random(trial)` and splits
  them into 6,000 training and 4,000 test queries.
- Training and test covariates are standardized separately with their own
  sample means and sample standard deviations (`ddof=1`).
- Matrices contain exactly 20 columns. Model features are columns `8:20`;
  there are no position-encoding columns.
- Displayed position remains available only for stage construction, the
  `position_asc` IM-RCS ranking, and deterministic tie breaking.
- Stage size is fixed at 10.
- The models are SQ-MNL, NS-MNL, IML, and IM-RCS. All four use the same
  observed-choice evaluator: NLL over every test query and hit@1/2/3 over
  booked queries only.
- IM-RCS has a hard cap of 100 outer EM iterations. A cap hit is recorded as
  nonconverged rather than silently accepted.

## Rebuild the data

From the repository root, first rebuild the filtered HDF intermediate from the
raw Expedia training CSV:

```bash
python sequential_rcs/python/preprocess/build_preprocess_data_com.py \
  --input-csv /path/to/train.csv \
  --output-h5 data.nosync/preprocess_data_com.h5
```

Then build the 50 site--trial MAT files:

```bash
python sequential_rcs/python/preprocess/pipeline.py \
  --h5 data.nosync/preprocess_data_com.h5 \
  --output-dir data.nosync/expedia_no_click_filter_no_position_20260816
```

See `python/preprocess/README.md` and the generated
`expedia_preprocess_manifest.json` for the schema and hash contract.

## Validate the implementation

```bash
python -m pytest -q \
  sequential_rcs/python/preprocess
```

```matlab
addpath('sequential_rcs/matlab/tests');
run_model_core_tests;
run_model_fit_smoke_tests;
R = runtests('sequential_rcs/matlab/tests/test_experiment_layer.m');
assert(all([R.Passed]));
```

The model formulas and public interfaces are documented in
`matlab/models/README.md`. The guarded runner, smoke command, full-grid gate,
and canonical 200-row compiler are documented in
`matlab/experiment/README.md`.

## Audit and report the completed grid

After MATLAB has compiled the exactly 200-row canonical CSV, run the
dependency-free, noninteractive reporter:

```bash
python sequential_rcs/python/report_expedia_results.py \
  --input sequential_rcs/results/expedia_four_model_results.csv \
  --output-dir sequential_rcs/results
```

The reporter checks the complete key grid, frozen seeds and schema, common
data hash within each site/trial, finite common metrics, NLL/count identities,
convergence-status consistency, and the probability audit. It never removes
nonconverged fits. It produces the site-level mean/sample-SD CSV, a generated
LaTeX table, row-level diagnostic CSV/JSON, and structured claims JSON. The
claims file sets `publishable=false` whenever any required fit is not
converged.

Run its focused contract tests with:

```bash
python -m unittest discover -s sequential_rcs/python/tests -v
```

## Historical files

`matlab/IMRCS_EM_function.m`, `matlab/SMNL_impatience_function.m`,
`matlab/run_expedia_imrcs_em.m`, the three notebooks, and
`expedia_imrcs_figures/` are restored audit material. They encode the old data
paths, old likelihood/evaluation conventions, or old nonconverged results and
must not be used to generate the revised manuscript table.
