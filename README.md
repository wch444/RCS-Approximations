# Sequential Choice Model computational code

This code reproduces the Expedia, Trivago, and synthetic experiments used in
the SCM comparison study. It includes the complete 5-by-10 fitted grids for
Expedia and Trivago and the published synthetic result bundle. Raw data,
processed matrices, and generated empirical results are not bundled.

## Paper

1. Pin Gao, Yicheng Liu, Chenhao Wang, Zizhuo Wang. 2026+. **RCS Approximations for Assortment Optimization with Sequential Consumer Behavior.** *Manufacturing & Service Operations Management.*

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

## Expedia: exact run order

Download `train.csv` from the
[Expedia Personalized Sort competition](https://www.kaggle.com/c/expedia-personalized-sort/data).
The first command reconstructs the
historical fixed-format pandas HDF5 input, under key `data`; the second builds
the corrected 50-file MATLAB sample grid:

```bash
python -m experiments.predictive_expedia.python.preprocess.raw_to_h5 \
  --csv data/raw/expedia/train.csv \
  --output data/raw/expedia/preprocess_data_com.h5

python -m experiments.predictive_expedia.python.preprocess.pipeline \
  --h5 data/raw/expedia/preprocess_data_com.h5 \
  --output-dir data/processed/expedia

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
python reproduce.py fit-expedia --dataset expedia \
  --data-dir /absolute/path/to/expedia_data_stage \
  --fit-dir /absolute/path/to/expedia_fit_stage
python reproduce.py fit --dataset trivago
```

For Expedia, first generate the complete processed matrix grid in the explicit
data staging directory above. Both Expedia staging directories must be outside
`data/processed/expedia/` and `opt_data/expedia/`; fresh fitting writes to the
staged fit directory. Review a complete staged grid before deliberately
promoting it for evaluation. `--resume` validates completed staged tasks.
For Trivago, use a disposable clone or worktree and move the bundled
`opt_data/trivago/` directory aside because fitting replaces its task files.

## Synthetic experiment

The synthetic experiment needs no external data:

```bash
python reproduce.py synthetic
```

The checked-in outputs are under `experiments/synthetic/results/`: 18,000
trial rows, their aggregate summary, the paper-value CSV, and the generated
LaTeX table. Re-running the command should reproduce these files.
