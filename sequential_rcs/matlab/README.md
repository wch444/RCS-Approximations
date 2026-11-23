# Archived Impatient Mixed-RCS EM MATLAB Code

> **Legacy audit material.** Do not use this README or the adjacent legacy
> runner to produce the revised Expedia table. The frozen four-model
> implementation is documented in `models/README.md`, and the guarded
> experiment workflow is documented in `experiment/README.md`.

This folder implements the EM estimator described in `sequential.tex`, Section
`Estimation` and Appendix `Implementation Details of the EM Algorithm`.

## Main Entry Point

```matlab
[model, metrics] = IMRCS_EM_function(trainData, testData, opts);
```

Default Expedia layout:

- `data(:,1)`: query/session id
- `data(:,2)`: product id, with `0` for the outside/no-purchase row
- `data(:,4)`: displayed position, with `100` for the outside/no-purchase row
- `data(:,5)`: chosen indicator, including the outside row on no-purchase
- `data(:,8:end)`: product features in `q = sigmoid(x * theta)`

Default model choices:

- stage size `b = opts.stageSize = 10`
- patience support `Y in {1,...,max_t K_t}`
- candidate ranking support induced by display position, price, star rating,
  location score, and review score, each with sensible ascending/descending
  variants

Custom ranking support can be passed as a struct array:

```matlab
opts.rankings = struct( ...
    'name', {'position', 'price_low', 'star_high'}, ...
    'column', {4, 9, 10}, ...
    'direction', {'ascend', 'ascend', 'descend'});
```

## Expedia Runner

```matlab
run("sequential_rcs/matlab/run_expedia_imrcs_em.m")
```

The runner reads `data.nosync/10k_sample_<site>_<trial>.mat`, saves the fitted
model to `opt_data/imrcs_em_<site>_<trial>.mat`, and appends one result line to
`record/imrcs_em_expedia.txt`.

For a quick smoke test, set `sampleTrainSessions` and `sampleTestSessions` in
`run_expedia_imrcs_em.m` to small integers. Set them back to `[]` for full data.

## Full Expedia 5-Site Batch

```matlab
run("test/test_10k_expedia_imrcs_em.m")
```

This matches the loop structure of `test/test_10k_expedia.m`:

- sites: `[15,14,5,24,32]`
- trials: `1:10`
- page/stage size: `10`
- Expedia feature block: `data(:, 1:20)` with features from column `8`
- result text file: `record/10k_sample_expedia_imrcs_em_k10.txt`
- per-run model files: `opt_data/imrcs_em_k10_<site>_<trial>.mat`

Recorded measurements include train/test negative log-likelihood, RMSE,
weighted squared error, `ChiSq100`, MAPE, hit@1/2/3, purchase rate, EM
iterations, final likelihood improvement, number of rankings, and number of
patience support points.

## TeX-to-Code Mapping

- `p_y`: `model.p`
- `lambda_k = sum_{y>=k} p_y`: `model.lambda`
- `theta`: `model.theta`
- `beta_sigma`: `model.beta`
- `Z_hat`: posterior patience weights computed in the E-step
- `H_hat`: posterior ranking weights for purchase sessions
- Problem `(convex)`: the weighted logistic objective solved by `fminunc`
