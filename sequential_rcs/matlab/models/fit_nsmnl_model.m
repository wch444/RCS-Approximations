function [model, fitdiag] = fit_nsmnl_model(train_data, opts)
%FIT_NSMNL_MODEL Fit stop-page-marginalized NS-MNL.
%
% Clicks from TRAIN_DATA alone estimate the stopping-page distribution:
% the proxy is the page of the last click, or the final page when there is
% no click.  Theta is then fitted to the marginal observed-choice likelihood,
% integrating over that distribution.  Test clicks are never an input.
if nargin < 2
    opts = struct();
end
opts = sequential_model_options(opts);
parsed = prepare_sequential_sessions(train_data, opts);
[lambda, stop_counts] = stopping_distribution(parsed, opts);
d = parsed.numFeatures;
theta0 = initialize_theta(opts.theta0, d);

previous_rng = rng;
restore_rng = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
rng(opts.rngSeed, 'twister');
starts = zeros(d, opts.numStarts);
starts(:, 1) = theta0;
for j = 2:opts.numStarts
    starts(:, j) = theta0 + 0.05 * randn(d, 1);
end

optim_options = optimizer_options(opts);
start_objective = Inf(opts.numStarts, 1);
start_exitflag = nan(opts.numStarts, 1);
outputs = cell(opts.numStarts, 1);
solutions = cell(opts.numStarts, 1);
objective = @(theta) nsmnl_objective(theta, parsed, lambda);
gradient_audit = [];
if opts.gradientCheck
    gradient_audit = finite_difference_gradient_check(objective, theta0, ...
        opts.gradientCheckStep, opts.gradientCheckTolerance, 'NS-MNL theta');
end
for j = 1:opts.numStarts
    [solutions{j}, start_objective(j), start_exitflag(j), outputs{j}] = ...
        fminunc(objective, starts(:, j), optim_options);
end
[fval, best] = min(start_objective);
if ~isfinite(fval)
    error('NSMNL:OptimizationFailed', 'All NS-MNL starts returned nonfinite objectives.');
end
theta = solutions{best};
output = outputs{best};

model = struct('id', 'nsmnl', 'theta', theta(:), 'lambda', lambda(:), ...
    'stopCounts', stop_counts(:), 'opts', opts, 'numFeatures', d, ...
    'trainNegLogLik', fval);
fitdiag = struct('trainNegLogLik', fval, 'exitFlag', start_exitflag(best), ...
    'iterations', output.iterations, 'functionCount', output.funcCount, ...
    'firstOrderOptimality', output.firstorderopt, 'bestStart', best, ...
    'selectedRestart', best, 'restartCount', opts.numStarts, ...
    'startObjectives', start_objective, 'startExitFlags', start_exitflag, ...
    'numSessions', parsed.numSessions, 'numProducts', parsed.numProducts, ...
    'stopCounts', stop_counts(:), 'lambda', lambda(:), ...
    'converged', start_exitflag(best) > 0, 'gradientCheck', gradient_audit, ...
    'maxIterations', opts.directMaxIter);
if fitdiag.converged
    fitdiag.status = 'converged';
else
    fitdiag.status = 'optimizer_not_converged';
end
end

function [lambda, counts] = stopping_distribution(parsed, opts)
m = parsed.maxStage;
stop_page = zeros(parsed.numSessions, 1);
for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    rows = parsed.productData(s.idx, :);
    last_click = find(rows(:, opts.clickCol) == 1, 1, 'last');
    if isempty(last_click)
        stop_page(t) = s.K;
    else
        stop_page(t) = s.stageOf(last_click);
    end
end
counts = accumarray(stop_page, 1, [m, 1]);
lambda = counts ./ sum(counts);
end

function [nll, grad] = nsmnl_objective(theta, parsed, lambda)
log_lambda = log(lambda(:));
m = numel(lambda);
log_lik = 0;
grad_log_lik = zeros(size(theta));

for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    X = parsed.X(s.idx, :);
    eta = X * theta;
    log_denom = zeros(m, 1);
    mean_x = zeros(size(X, 2), m);
    for y = 1:m
        viewed = s.stageOf <= min(y, s.K);
        log_denom(y) = sequential_logsumexp([0; eta(viewed)]);
        product_share = exp(eta(viewed) - log_denom(y));
        mean_x(:, y) = X(viewed, :)' * product_share;
    end

    log_component = log_lambda - log_denom;
    if s.isPurchase
        k = s.stageOf(s.choiceLocal);
        log_component(1:min(k - 1, m)) = -Inf;
    end
    log_mix = sequential_logsumexp(log_component);
    if ~isfinite(log_mix)
        nll = Inf;
        grad = zeros(size(theta));
        return
    end
    posterior = exp(log_component - log_mix);
    log_lik = log_lik + log_mix;
    grad_log_lik = grad_log_lik - mean_x * posterior;
    if s.isPurchase
        log_lik = log_lik + eta(s.choiceLocal);
        grad_log_lik = grad_log_lik + X(s.choiceLocal, :)';
    end
end

nll = -log_lik;
grad = -grad_log_lik;
end

function theta = initialize_theta(theta0, d)
if isempty(theta0)
    theta = zeros(d, 1);
else
    theta = theta0(:);
    if numel(theta) ~= d
        error('NSMNL:Theta0', 'theta0 has %d entries; expected %d.', numel(theta), d);
    end
end
end

function options = optimizer_options(opts)
if ~isempty(opts.optimOptions)
    options = opts.optimOptions;
else
    options = optimoptions('fminunc', 'Display', display_value(opts.verbose), ...
        'Algorithm', 'quasi-newton', 'SpecifyObjectiveGradient', true, ...
        'MaxIterations', opts.directMaxIter, ...
        'MaxFunctionEvaluations', opts.maxFunctionEvaluations, ...
        'OptimalityTolerance', opts.optimizerOptimalityTolerance, ...
        'FunctionTolerance', opts.optimizerOptimalityTolerance, ...
        'StepTolerance', 1e-10);
end
end

function value = display_value(verbose)
if verbose, value = 'iter'; else, value = 'off'; end
end
