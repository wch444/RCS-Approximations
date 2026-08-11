# Sequential Choice Model computational code

This code reproduces the Expedia, Trivago, and synthetic experiments used in
the SCM comparison study. It includes the complete 5-by-10 fitted grids for
Expedia and Trivago and the published synthetic result bundle. Raw data,
processed matrices, and generated empirical results are not bundled.

## Requirements

- MATLAB R2023b with Optimization Toolbox, Statistics and Machine Learning
  Toolbox, and Parallel Computing Toolbox.
- Conda and the pinned Python 3.11 environment in `environment.yml`.

From the repository root:

```bash
conda env create -f environment.yml
conda activate scm-repro-2026
python reproduce.py --help
```

All commands below must also be run from the repository root. Generated raw
and processed data and empirical result bundles are written only to paths
ignored by Git. The fitted grids and synthetic outputs described below are
versioned files.

## Expedia: exact run order

Download `train.csv` from the
[Expedia Personalized Sort competition](https://www.kaggle.com/c/expedia-personalized-sort/data)
after accepting the competition rules. The first command reconstructs the
historical fixed-format pandas HDF5 input, under key `data`; the second builds
the corrected 50-file MATLAB sample grid:

```bash
python reproduce.py build-expedia-h5 \
  --csv data/raw/expedia/train.csv

python reproduce.py preprocess-expedia \
  --h5 data/raw/expedia/preprocess_data_com.h5

python reproduce.py expedia-local-predictive

python reproduce.py counterfactuals --dataset expedia
```

The HDF5 build preserves the historical raw-data filter: keep rows with
`10 < price_usd < 5000`, `srch_room_count == 1`, and
`srch_length_of_stay <= 10`; retain the documented 20 columns; then drop rows
with a missing retained value. This stage contains no random sampling. With the
original Kaggle file, the logical HDF frame contains 8,938,178 rows in original
CSV order with the filtered CSV index retained. HDF5 bytes may differ across
pandas/PyTables versions even when the table values are the same.

The preprocessing command removes complete queries with more than four clicks
before it constructs position indicators or samples query IDs. Trial `i` uses
Python seed `i`, takes 10,000 valid queries, and writes 6,000 training and 4,000
test queries. The historical split-specific standardization is intentionally
preserved: train and test use their own sample means and sample standard
deviations (`ddof=1`).

The repository already contains the 50 unified Expedia fits used by the paper
under `opt_data/expedia/`. They cover SCM, MNL, MMNL, CMNL, SeqINV, and NSMNL
for all five sites and ten trials. The commands above therefore reproduce the
evaluation without repeating the long fitting stage.

## Trivago: exact run order

Download the official RecSys Challenge 2019 data from the
[challenge page](https://www.recsyschallenge.com/2019/) or its
[Trivago data portal](https://recsys.trivago.cloud/). Place these two files at:

```text
data/raw/trivago/train.csv
data/raw/trivago/item_metadata.csv
```

Then run:

```bash
python reproduce.py preprocess --dataset trivago

python reproduce.py evaluate --dataset trivago

python reproduce.py tables --dataset trivago

python reproduce.py counterfactuals --dataset trivago
```

The 50 unified Trivago fits used by the paper are bundled under
`opt_data/trivago/`, so the default sequence evaluates those retained fits.

## Optional full re-estimation

The two fitting commands remain available:

```bash
python reproduce.py fit-expedia --dataset expedia
python reproduce.py fit --dataset trivago
```

Run a full re-estimation in a disposable clone or worktree after moving the
corresponding bundled `opt_data/<dataset>/` directory aside. Expedia refuses to
overwrite an existing task; Trivago replaces its canonical task files.
Expedia `--resume` is intended only for continuing a newly started incomplete
fit grid, not for consuming the bundled grid.

## Synthetic experiment

The synthetic experiment needs no external data:

```bash
python reproduce.py synthetic
```

The checked-in outputs are under `experiments/synthetic/results/`: 18,000
trial rows, their aggregate summary, the paper-value CSV, and the generated
LaTeX table. Re-running the command should reproduce these files.

## Bundled inputs and generated outputs

- Processed empirical matrices: `data/processed/<dataset>/`
- Bundled unified fitted parameters: `opt_data/<dataset>/`
- Experiment outputs: `experiments/<experiment>/results/`

Each unified fit contains SCM, MNL, MMNL, CMNL, SeqINV, and NSMNL parameters
for one market/trial task. SCM parameters use `[beta; alpha]`, with attraction
coefficients first and attention coefficients second. All maintained fitting,
evaluation, counterfactual, and synthetic stages use fixed task-level seeds;
the formulas are listed in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Additional contracts are documented in [docs/DATA.md](docs/DATA.md),
[docs/MODEL_FITTING.md](docs/MODEL_FITTING.md), and
[docs/CODE_LAYOUT.md](docs/CODE_LAYOUT.md).
