# Expedia preprocessing protocol

Protocol version: `expedia_no_click_filter_no_position_v1`.

This is the canonical entry point for the sequential-RCS Expedia experiment.
It invokes the project-local implementation in `implementation.py` and writes
the legacy 20-column MATLAB schema. Keeping this implementation inside
`sequential_rcs` prevents changes to the separate SMC preprocessing protocol
from changing this frozen experiment. It applies no click-count filter, adds
no position-encoding columns, retains dense displayed position and raw click,
uses `random.Random(trial)` for the 10,000-query sample, splits it 6,000/4,000,
and standardizes the two splits separately with their own sample SD (`ddof=1`).

To rebuild the HDF intermediate from the raw Expedia training CSV, run:

```bash
python sequential_rcs/python/preprocess/build_preprocess_data_com.py \
  --input-csv /path/to/train.csv \
  --output-h5 data.nosync/preprocess_data_com.h5
```

This stage retains rows with `10 < price_usd < 5000`, single-room searches,
and stays of at most 10 nights, then drops rows missing any retained field. It
refuses to replace an existing HDF file unless `--overwrite` is explicit.

Next, run the complete five-site by ten-trial MAT build:

```bash
python sequential_rcs/python/preprocess/pipeline.py \
  --h5 data.nosync/preprocess_data_com.h5 \
  --output-dir data.nosync/expedia_no_click_filter_no_position_20260816
```

The output directory is intentionally new. Do not use `--overwrite` when
building the approved grid. The manifest records the raw-input SHA-256, fixed
schema, split protocol, per-file and matrix hashes, query-ID hashes, and the
numbers of sampled zero-click and more-than-four-click queries in each split.
