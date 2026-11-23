# Expedia four-model experiment runner

This layer freezes the approved Expedia comparison:

- sites `5, 14, 15, 24, 32`, trials `1:10`, and models `sqmnl`, `nsmnl`,
  `iml`, `imrcs`;
- exactly 20 matrix columns and model features `8:20` (no position encoding);
- stage size 10;
- separate train/test standardization, as recorded in the preprocessing
  manifest;
- all-query marginal observed-choice NLL, including outside choices;
- booked-query-only hit@1/2/3, excluding outside, with deterministic ties;
- IM-RCS outer EM cap 100, relative tolerance `1e-6`, and absolute
  monotonicity tolerance `1e-7`.

The runner validates the global preprocessing manifest, the live SHA-256 of
each MAT, the 20-column session schema, and train/test query disjointness.
Predicted probability vectors are never silently renormalized.

From the repository root, inspect the complete plan without fitting or
writing outputs:

```matlab
addpath('sequential_rcs/matlab/experiment');
run_expedia_four_model_grid( ...
    'DataDir','data.nosync/expedia_no_click_filter_no_position_20260816');
```

Run one explicitly bounded smoke fit (saved only under
`sequential_rcs/results/smoke_fits/`):

```matlab
run_expedia_four_model_grid( ...
    'DataDir','data.nosync/expedia_no_click_filter_no_position_20260816', ...
    'OutputDir','sequential_rcs/results', ...
    'Sites',5,'Trials',1,'Models','sqmnl', ...
    'DryRun',false,'MaxTrainSessions',50,'MaxTestSessions',40);
```

The complete grid remains blocked unless both fitting and the separate full
run confirmation are explicit. Use this only after user approval:

```matlab
run_expedia_four_model_grid( ...
    'DataDir','data.nosync/expedia_no_click_filter_no_position_20260816', ...
    'OutputDir','sequential_rcs/results', ...
    'DryRun',false,'RequireFullRunConfirmation',true,'Resume',true);
```

After all 200 non-smoke artifacts exist under `sequential_rcs/results/fits/`,
build the canonical CSV:

```matlab
compile_expedia_results('sequential_rcs/results');
```

The compiler requires exactly one valid artifact for every site/trial/model
key. It does not drop cap hits, inner-optimizer failures, monotonicity
violations, or any other nonconverged status; those diagnostics remain as
explicit columns in the 200-row CSV. Smoke artifacts are in a separate
directory and cannot enter the canonical compiler.
