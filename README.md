# Sequential Choice Model computational code

This code reproduces the Expedia, Trivago, and synthetic experiments used in
the SCM comparison study. Raw data, processed matrices, fitted parameters, and
generated results are not bundled.

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

All commands below must also be run from the repository root. Generated data,
fits, and results are written only to paths ignored by Git.

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

python reproduce.py fit-expedia --dataset expedia

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

`fit-expedia` freshly estimates SCM, MNL, MMNL, CMNL, SeqINV, and NSMNL for all
five sites and ten trials. The resulting unified files are written to
`opt_data/expedia/`. For deterministic parallel sharding, launch distinct
`--shard-index` values with one common `--shard-count`; add `--resume` when
restarting an interrupted grid.

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

python reproduce.py fit --dataset trivago

python reproduce.py evaluate --dataset trivago

python reproduce.py tables --dataset trivago

python reproduce.py counterfactuals --dataset trivago
```

The unified six-model fit grid is written to `opt_data/trivago/`.

## Synthetic experiment

The synthetic experiment needs no external data:

```bash
python reproduce.py synthetic
```

## Generated outputs

- Processed empirical matrices: `data/processed/<dataset>/`
- Unified six-model fitted parameters: `opt_data/<dataset>/`
- Experiment outputs: `experiments/<experiment>/results/`

Each unified fit contains SCM, MNL, MMNL, CMNL, SeqINV, and NSMNL parameters
for one market/trial task. SCM parameters use `[beta; alpha]`, with attraction
coefficients first and attention coefficients second. All maintained fitting,
evaluation, counterfactual, and synthetic stages use fixed task-level seeds;
the formulas are listed in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Additional contracts are documented in [docs/DATA.md](docs/DATA.md),
[docs/MODEL_FITTING.md](docs/MODEL_FITTING.md), and
[docs/CODE_LAYOUT.md](docs/CODE_LAYOUT.md).
