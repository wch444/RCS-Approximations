function [model, fitdiag] = fit_iml_model(train_data, opts)
%FIT_IML_MODEL Fit cumulative-denominator IML by observed-choice MLE.
%
% Patience p is parameterized by m-1 free logits with the last category as
% the reference.  This removes the additive-logit nonidentification while
% retaining an unconstrained optimization problem.
if nargin < 2
    opts = struct();
end
opts = sequential_model_options(opts);
parsed = prepare_sequential_sessions(train_data, opts);
d = parsed.numFeatures;
m = parsed.maxStage;
theta0 = initialize_theta(opts.theta0, d);
p0 = initialize_simplex(opts.p0, m);
gamma0 = log(p0(1:end - 1) ./ p0(end));
x0 = [theta0; gamma0];

previous_rng = rng;
restore_rng = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
rng(opts.rngSeed, 'twister');
starts = zeros(numel(x0), opts.numStarts);
starts(:, 1) = x0;
for j = 2:opts.numStarts
    starts(:, j) = x0 + 0.05 * randn(numel(x0), 1);
end

optim_options = optimizer_options(opts);
start_objective = Inf(opts.numStarts, 1);
start_exitflag = nan(opts.numStarts, 1);
outputs = cell(opts.numStarts, 1);
solutions = cell(opts.numStarts, 1);
objective = @(x) iml_objective(x, parsed, d, m);
gradient_audit = [];
if opts.gradientCheck
    gradient_audit = finite_difference_gradient_check(objective, x0, ...
        opts.gradientCheckStep, opts.gradientCheckTolerance, ...
        'IML theta and patience reference logits');
end
for j = 1:opts.numStarts
    [solutions{j}, start_objective(j), start_exitflag(j), outputs{j}] = ...
        fminunc(objective, starts(:, j), optim_options);
end
[fval, best] = min(start_objective);
if ~isfinite(fval)
    error('IML:OptimizationFailed', 'All IML starts returned nonfinite objectives.');
end
x = solutions{best};
theta = x(1:d);
gamma = x(d + 1:end);
p = reference_softmax(gamma);
output = outputs{best};

model = struct('id', 'iml', 'theta', theta(:), 'gamma', gamma(:), ...
    'p', p(:), 'lambda', flipud(cumsum(flipud(p(:)))), 'opts', opts, ...
    'numFeatures', d, 'numPatience', m, 'trainNegLogLik', fval);
fitdiag = struct('trainNegLogLik', fval, 'exitFlag', start_exitflag(best), ...
    'iterations', output.iterations, 'functionCount', output.funcCount, ...
    'firstOrderOptimality', output.firstorderopt, 'bestStart', best, ...
    'selectedRestart', best, 'restartCount', opts.numStarts, ...
    'startObjectives', start_objective, 'startExitFlags', start_exitflag, ...
    'numSessions', parsed.numSessions, 'numProducts', parsed.numProducts, ...
    'converged', start_exitflag(best) > 0, 'gradientCheck', gradient_audit, ...
    'maxIterations', opts.directMaxIter);
if fitdiag.converged
    fitdiag.status = 'converged';
else
    fitdiag.status = 'optimizer_not_converged';
end
end

function [nll, grad] = iml_objective(x, parsed, d, m)
theta = x(1:d);
gamma = x(d + 1:end);
p = reference_softmax(gamma);
log_p = log(p);
lambda = flipud(cumsum(flipud(p)));
log_lik = 0;
grad_theta = zeros(d, 1);
grad_gamma = zeros(max(m - 1, 0), 1);

for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    X = parsed.X(s.idx, :);
    eta = X * theta;
    log_denom = zeros(s.K, 1);
    mean_x = zeros(d, s.K);
    for k = 1:s.K
        viewed = s.stageOf <= k;
        log_denom(k) = sequential_logsumexp([0; eta(viewed)]);
        product_share = exp(eta(viewed) - log_denom(k));
        mean_x(:, k) = X(viewed, :)' * product_share;
    end

    if s.isPurchase
        k = s.stageOf(s.choiceLocal);
        previous_log_denom = 0;
        previous_mean = zeros(d, 1);
        if k > 1
            previous_log_denom = log_denom(k - 1);
            previous_mean = mean_x(:, k - 1);
        end
        log_lik = log_lik + log(lambda(k)) + eta(s.choiceLocal) ...
            - previous_log_denom - log_denom(k);
        grad_theta = grad_theta + X(s.choiceLocal, :)' ...
            - previous_mean - mean_x(:, k);
        posterior_patience = zeros(m, 1);
        posterior_patience(k:m) = p(k:m) ./ lambda(k);
    else
        log_component = zeros(m, 1);
        for y = 1:m
            log_component(y) = log_p(y) - log_denom(min(y, s.K));
        end
        session_log_prob = sequential_logsumexp(log_component);
        posterior_patience = exp(log_component - session_log_prob);
        log_lik = log_lik + session_log_prob;
        for y = 1:m
            grad_theta = grad_theta ...
                - posterior_patience(y) * mean_x(:, min(y, s.K));
        end
    end
    if m > 1
        grad_gamma = grad_gamma + posterior_patience(1:m - 1) - p(1:m - 1);
    end
end

nll = -log_lik;
grad = -[grad_theta; grad_gamma];
end

function p = reference_softmax(gamma)
logits = [gamma(:); 0];
normalizer = sequential_logsumexp(logits);
p = exp(logits - normalizer);
end

function theta = initialize_theta(theta0, d)
if isempty(theta0)
    theta = zeros(d, 1);
else
    theta = theta0(:);
    if numel(theta) ~= d
        error('IML:Theta0', 'theta0 has %d entries; expected %d.', numel(theta), d);
    end
end
end

function p = initialize_simplex(p0, m)
if isempty(p0)
    p = ones(m, 1) ./ m;
else
    p = p0(:);
    if numel(p) ~= m || any(~isfinite(p)) || any(p <= 0)
        error('IML:P0', 'p0 must contain %d strictly positive finite entries.', m);
    end
    p = p ./ sum(p);
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
