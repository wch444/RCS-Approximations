# Numerical upper-bound curve

`matlab/worst_ratio.m` computes the numerical construction behind the paper's
upper-bound curve for product counts `n = 2,...,50`, including the reported
value of approximately `0.407` at `n = 15`.
An `n = 15` check in MATLAB R2023b returned `0.406907899421`.

Run in MATLAB from the repository root:

```matlab
run('experiments/worst_ratio/matlab/worst_ratio.m')
```

The script requires Optimization Toolbox. It uses `fmincon` with SQP,
an all-ones starting point, lower bound `epsilon = 1e-5`, constraint tolerance
`1e-10`, at most `1e6` function evaluations, and at most `1e4` iterations.
For a single-size run, set `n_list = 15` at the top of the script.

The decision vector is `[r, v]`, with `r(1) = 1` and attention probabilities
equal to one. The numerator is the singleton revenue `r(1)*v(1)/(1+v(1))`.
The denominator is the RCS cascade revenue in index order `1,...,n`.
The nonlinear constraints bound each prefix's reverse-order SCM revenue by
the singleton revenue. The formulas are annotated beside their implementation.

Running the script plots the curve and leaves `n_list` and `ratio_list` in the
workspace. `xx` and `fval` contain the solution for the final product count.
SQP returns a local solution, so the numerical values depend on initialization,
solver tolerances, and MATLAB version.
