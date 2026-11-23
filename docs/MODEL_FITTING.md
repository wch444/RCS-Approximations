# Model fitting contract

## Unified fit files

Each empirical dataset has a five-market by ten-trial fitted grid under
`opt_data/<dataset>/`. Every MAT file contains one `fit` structure, and
`fit.models` stores `scm`, `mnl`, `mmnl_alpha`, `mmnl_beta`, `cmnl`, `seqinv`,
`nsmnl`, and `nsmnl_lambda`. The two MMNL arrays belong to one model, so there
are six fitted model specifications per task.

SCM vectors use `[beta; alpha]`, with attraction coefficients before attention
coefficients.

The fit files retain final parameter values, fixed seeds, and dimensions. They
do not retain per-iteration MMNL histories or termination-status records.

The complete Expedia and Trivago grids used for the reported experiments are
included in the repository. The default reproduction workflow evaluates these
bundled fits and does not repeat estimation.

## Expedia

`python reproduce.py fit-expedia --dataset expedia --data-dir ... --fit-dir ...`
fits all six models for corrected Expedia matrices. Both explicit staging
directories must be outside `data/processed/expedia/` and `opt_data/expedia/`;
fresh fits are written to the staged fit directory. The default run covers
all five sites and all ten trials; `--markets`, `--trials`, `--shard-index`, and
`--shard-count` select a deterministic subset. `--resume` validates and skips
completed staged tasks. Review complete data and fit grids before deliberately
promoting them for evaluation.

The five non-MMNL objectives are smooth and unconstrained and use MATLAB
`fminunc`. The three-component MMNL uses the maintained stable log-domain
EM/GEM implementation with analytic M-step gradients. Expedia MMNL uses a
deterministic position-reference parameterization and constrains position bins
that are absent from the corresponding training split, then expands the
coefficients back to the full matrix dimension.

## Trivago

`python reproduce.py fit --dataset trivago` fits the same six-model grid for
the Trivago matrices. Run it only after moving the bundled
`opt_data/trivago/` directory aside, because this command replaces canonical
task files.

All model fits use fixed task-keyed MATLAB `twister` seeds. Numerical model
kernels and seed formulas are shared with the local research implementation.
