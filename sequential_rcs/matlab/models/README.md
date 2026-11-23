# Expedia four-model mathematical core

This directory is the clean implementation used by the new Expedia
comparison.  It does not call the legacy `Seq_*`, `NS_MNL_function`, or
`SMNL_impatience_function` files.

Public fitting interfaces:

```matlab
[model, fitdiag] = fit_sqmnl_model(train_data, opts);
[model, fitdiag] = fit_nsmnl_model(train_data, opts);
[model, fitdiag] = fit_iml_model(train_data, opts);
[model, fitdiag] = fit_imrcs_model(train_data, opts);
```

Public prediction interface:

```matlab
[prob, detail] = predict_sequential_model(model, one_query_rows, opts);
```

`prob` contains products in canonical `(position,item,input-row)` order and
then the outside option.  Prediction validates nonnegativity and unit mass;
it never silently renormalizes.

The feature columns are fixed to `8:20`.  Position is retained only for
stage construction, the `position_asc` IM-RCS rule, and deterministic ties.
NS-MNL clicks are used only in `fit_nsmnl_model` to estimate the training
stop-page distribution.  Both its fitting objective and all predictions use
the stop-page-marginal observed-choice likelihood.

The frozen optimizer fields are deliberately unambiguous:
`directMaxIter=500`, `innerMaxIter=500`,
`optimizerOptimalityTolerance=1e-6`,
`imrcsOuterMaxIter=100`, `emRelativeTolerance=1e-6`, and an absolute
`monotonicityTolerance=1e-7` by default.

Run the exact probability-oracle tests with:

```matlab
addpath('sequential_rcs/matlab/tests');
run_model_core_tests;
```
