# Code layout

Code is organized by experiment:

```text
experiments/
  predictive_expedia/{matlab,python}
  predictive_trivago/{matlab,python}
  mixed_order_expedia/{matlab,python}
  counterfactual_trivago/{matlab,python}
  synthetic/{matlab,python}
```

Each experiment contains its numerical producer or evaluator under `matlab/`
and its runner or reporter under `python/`. Experiment-specific MATLAB helpers
live in that experiment's `matlab/private/` directory.

Shared code lives under:

- `common/matlab/models`: numerical model and prediction kernels;
- `common/matlab/workflow`: dataset, fixed-seed, and fit-schema helpers;
- `common/matlab/publishing`: result-table publishing;
- `common/python`: process and predictive-reporting helpers.

`reproduce.py` is the public command-line dispatcher. Do not use
`addpath(genpath(repo_root))`; each MATLAB entry point adds only its experiment
directory and the shared directories it needs.

The repository contains code, documentation, the two unified fitted grids
under `opt_data/`, and the published synthetic result bundle. At runtime,
generated matrices are placed under `data/processed/`, while empirical result
bundles are written under the corresponding experiment's `results/`
directory. Raw/processed data and generated empirical results remain ignored
by Git.
