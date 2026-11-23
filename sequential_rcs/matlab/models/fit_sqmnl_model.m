function [model, fitdiag] = fit_sqmnl_model(train_data, opts)
%FIT_SQMNL_MODEL Fit independent-stage SQ-MNL by exact observed-choice MLE.
if nargin < 2
    opts = struct();
end
opts = sequential_model_options(opts);
parsed = prepare_sequential_sessions(train_data, opts);
d = parsed.numFeatures;
theta0 = initialize_theta(opts.theta0, d);
optim_options = optimizer_options(opts);

objective = @(theta) sqmnl_objective(theta, parsed);
gradient_audit = [];
if opts.gradientCheck
    gradient_audit = finite_difference_gradient_check(objective, theta0, ...
        opts.gradientCheckStep, opts.gradientCheckTolerance, 'SQ-MNL theta');
end
[theta, fval, exitflag, output] = fminunc(objective, theta0, optim_options);

model = struct('id', 'sqmnl', 'theta', theta(:), 'opts', opts, ...
    'numFeatures', d, 'trainNegLogLik', fval);
fitdiag = base_diagnostics(fval, exitflag, output, parsed);
fitdiag.gradientCheck = gradient_audit;
fitdiag.maxIterations = opts.directMaxIter;
if fitdiag.converged
    fitdiag.status = 'converged';
else
    fitdiag.status = 'optimizer_not_converged';
end
end

function [nll, grad] = sqmnl_objective(theta, parsed)
log_lik = 0;
grad_log_lik = zeros(size(theta));

for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    X = parsed.X(s.idx, :);
    eta = X * theta;
    log_denom = zeros(s.K, 1);
    mean_x = zeros(size(X, 2), s.K);
    for k = 1:s.K
        in_stage = s.stageOf == k;
        log_denom(k) = sequential_logsumexp([0; eta(in_stage)]);
        product_share = exp(eta(in_stage) - log_denom(k));
        mean_x(:, k) = X(in_stage, :)' * product_share;
    end

    if s.isPurchase
        k = s.stageOf(s.choiceLocal);
        log_lik = log_lik + eta(s.choiceLocal) - sum(log_denom(1:k));
        grad_log_lik = grad_log_lik + X(s.choiceLocal, :)' ...
            - sum(mean_x(:, 1:k), 2);
    else
        log_lik = log_lik - sum(log_denom);
        grad_log_lik = grad_log_lik - sum(mean_x, 2);
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
        error('SQMNL:Theta0', 'theta0 has %d entries; expected %d.', numel(theta), d);
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
if verbose
    value = 'iter';
else
    value = 'off';
end
end

function diag = base_diagnostics(fval, exitflag, output, parsed)
diag = struct('trainNegLogLik', fval, 'exitFlag', exitflag, ...
    'iterations', output.iterations, 'functionCount', output.funcCount, ...
    'firstOrderOptimality', output.firstorderopt, ...
    'numSessions', parsed.numSessions, 'numProducts', parsed.numProducts, ...
    'converged', exitflag > 0);
end
