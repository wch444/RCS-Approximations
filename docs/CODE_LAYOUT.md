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

The repository contains code and documentation only. At runtime, generated
matrices are placed under `data/processed/`, fitted parameters under
`opt_data/`, and result bundles under the corresponding experiment's
`results/` directory. These paths are ignored by Git.
