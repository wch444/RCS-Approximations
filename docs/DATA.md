# Data contract

No raw observations, processed matrices, fitted parameters, or generated
results are distributed with this code.

## Expedia

Obtain `train.csv` from the
[Expedia Personalized Sort competition](https://www.kaggle.com/c/expedia-personalized-sort/data)
and place it at:

```text
data/raw/expedia/train.csv
```

`python reproduce.py build-expedia-h5 --csv data/raw/expedia/train.csv`
reconstructs `data/raw/expedia/preprocess_data_com.h5`. It applies the recovered
historical row filters `10 < price_usd < 5000`, `srch_room_count == 1`, and
`srch_length_of_stay <= 10`, retains the following columns in order, and then
drops rows with a missing retained value:

```text
srch_id, prop_id, site_id, position, date_time, booking_bool, random_bool,
click_bool, price_usd, prop_starrating, prop_location_score1,
prop_review_score, srch_length_of_stay, prop_log_historical_price,
promotion_flag, srch_booking_window, srch_adults_count, prop_brand_bool,
srch_children_count, srch_saturday_night_bool
```

Rows are not restricted by `random_bool`; both values remain in the HDF5
frame, matching the historical build.

The output is a pandas fixed-format HDF5 frame under key `data`. The original
competition file produces 8,938,178 logical rows in original CSV order, with
the filtered CSV index retained. Binary HDF5 serialization is library-version
dependent and is not used as the reproducibility target.

For each site, the pipeline removes every complete query with more than four
clicks before position-bin construction and query sampling. Trial `i` uses
Python seed `i`, shuffles only valid query IDs, samples 10,000, and splits them
into 6,000 train and 4,000 test queries. Historical train/test-specific sample
standardization (`ddof=1`) is preserved. One outside-option row is appended to
each query.

The command writes 50 files named `10k_sample_<site>_<trial>.mat` plus
`expedia_preprocess_manifest.json` under `data/processed/expedia/`. Sites are
`5,14,15,24,32`; trials are `1,...,10`. Site 24 has 27 matrix columns and the
other sites have 28 because of the realized position-bin support.

## Trivago

Download the official RecSys Challenge 2019 data from the
[challenge page](https://www.recsyschallenge.com/2019/) or
[Trivago portal](https://recsys.trivago.cloud/), then place:

```text
data/raw/trivago/train.csv
data/raw/trivago/item_metadata.csv
```

The pipeline parses exact official star-rating and spa tags, uses the final
observed click-out as the chosen outcome, removes sessions whose chosen hotel
is lost to filtering, and preserves the historical split-specific population
standardization. It writes the five-market by ten-trial grid under
`data/processed/trivago/`.

## Fitted parameters

Both empirical workflows write `fit_<market>_<trial>.mat` files under
`opt_data/<dataset>/`. Expedia uses its numeric site as the market key. Each
file contains one top-level `fit` structure whose `models` field stores the six
model specifications.
