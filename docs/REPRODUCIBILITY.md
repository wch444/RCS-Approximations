# Reproducibility workflow

Create the pinned environment and run every command from the repository root:

```bash
conda env create -f environment.yml
conda activate scm-repro-2026
```

MATLAB R2023b with Optimization, Statistics and Machine Learning, and Parallel
Computing toolboxes is the tested numerical environment.

## Expedia

Starting from the Kaggle `train.csv`, run:

```bash
python reproduce.py build-expedia-h5 \
  --csv data/raw/expedia/train.csv
python reproduce.py preprocess-expedia \
  --h5 data/raw/expedia/preprocess_data_com.h5
python reproduce.py expedia-local-predictive
python reproduce.py counterfactuals --dataset expedia
```

The source data are available from the
[Expedia competition page](https://www.kaggle.com/c/expedia-personalized-sort/data).
The CSV-to-HDF5 stage contains no randomness. HDF5-to-MAT preprocessing uses
the trial number as the Python shuffle seed. Full fitting
uses MATLAB `twister` with base seed
`42 + 1000*trial + market_index`, where the MMNL market order is
`15,14,5,24,32`. SCM, SeqINV, NSMNL, MNL, and CMNL use offsets `+10`, `+20`,
`+30`, `+40`, and `+50`; MMNL uses the base seed.

Predictive evaluation uses
`20260808 + 10000*trial + 100*market_index + model_index`. The mixed-order
experiment samples with `20260809 + 100000*site + 1000*trial`, evaluates each
query with `120260809 + 100000*site + 1000*trial + query_index`, and uses seed
`2026` for its Dirichlet summary.

## Trivago

After downloading the official RecSys Challenge 2019 `train.csv` and
`item_metadata.csv`, run:

```bash
python reproduce.py preprocess --dataset trivago
python reproduce.py evaluate --dataset trivago
python reproduce.py tables --dataset trivago
python reproduce.py counterfactuals --dataset trivago
```

Trivago fitting uses task seed `42 + 1000*trial + market_index`, with market
order `BR,US,JP,UK,DE`; the five non-MMNL models use the same offsets listed
above. Predictive evaluation uses the same `20260808` task-keyed formula as
Expedia. Counterfactual seeds are defined per model, market, trial, and query
inside `revenue_misspec_trivago.m`.

## Synthetic

```bash
python reproduce.py synthetic
```

The synthetic grid uses `N in {10,15,20}`, `B in {3,5,8}`, 1,000 replications,
and fixed task-keyed seeds rooted at `20260808`.

## Output contracts

- Expedia and Trivago predictive evaluation each produce 300 trial rows for
  the five-market by ten-trial by six-model grid.
- The Trivago counterfactual covers the same 300 fitted tasks.
- The Expedia mixed-order study covers five sites, ten trials, and 500 eligible
  queries per task. Standard errors use the ten trial means as the replication
  unit.
- The synthetic grid produces 18,000 trial rows before aggregation.

Generated matrices and empirical CSV, table, and figure bundles remain in
ignored runtime directories. The source distribution includes the two
50-file fitted grids and the four synthetic result files; raw/processed data
and generated empirical outputs remain excluded.
