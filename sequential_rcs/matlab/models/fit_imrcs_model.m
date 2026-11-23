function [model, fitdiag] = fit_imrcs_model(train_data, opts)
%FIT_IMRCS_MODEL Fit IM-RCS by the approved 100-iteration EM algorithm.
%
% Matching features are exactly columns 8:20.  Displayed position enters
% only through stage formation, deterministic tie breaking, and the fixed
% position_asc ranking rule.  The returned history reports cap hits,
% likelihood improvements, monotonicity violations, and inner optimizer
% exit flags; hitting iteration 100 is never labeled convergence.
if nargin < 2
    opts = struct();
end
opts = sequential_model_options(opts);
if isempty(opts.rankings)
    opts.rankings = default_imrcs_rankings(opts);
end
parsed = prepare_sequential_sessions(train_data, opts);
T = parsed.numSessions;
d = parsed.numFeatures;
m = parsed.maxStage;
R = numel(opts.rankings);

rank_pos = cell(T, 1);
for t = 1:T
    s = parsed.sessions(t);
    rank_pos{t} = imrcs_rank_positions(parsed.productData(s.idx, :), ...
        opts.rankings, opts);
end

theta = initialize_theta(opts.theta0, d);
p = normalize_simplex(initialize_simplex(opts.p0, m, 'p0'), opts.probFloor);
beta = normalize_simplex(initialize_simplex(opts.beta0, R, 'beta0'), opts.probFloor);
inner_options = inner_optimizer_options(opts);

history.logLik = nan(opts.imrcsOuterMaxIter + 1, 1);
history.improvement = nan(opts.imrcsOuterMaxIter, 1);
history.relativeImprovement = nan(opts.imrcsOuterMaxIter, 1);
history.thetaObjective = nan(opts.imrcsOuterMaxIter, 1);
history.thetaExitFlags = nan(opts.imrcsOuterMaxIter, 1);
history.p = nan(opts.imrcsOuterMaxIter + 1, m);
history.beta = nan(opts.imrcsOuterMaxIter + 1, R);
history.logLik(1) = observed_loglik(parsed, rank_pos, theta, p, beta);
history.p(1, :) = p;
history.beta(1, :) = beta;

converged = false;
stopped_on_tolerance = false;
monotonicity_violations = 0;
iter_done = 0;
gradient_audit = [];
for iter = 1:opts.imrcsOuterMaxIter
    [Z, H] = expectation_step(parsed, rank_pos, theta, p, beta);
    p = normalize_simplex(mean(Z, 1), opts.probFloor);
    if parsed.numPurchases > 0
        beta = normalize_simplex(sum(H, 1) ./ parsed.numPurchases, opts.probFloor);
    end

    [exposure_weight, success_feature_sum] = ...
        theta_sufficient_statistics(parsed, rank_pos, Z, H, R);
    theta_objective = @(value) logistic_m_step(value, parsed.X, ...
        exposure_weight, success_feature_sum, opts.ridge);
    if opts.gradientCheck && iter == 1
        gradient_audit = finite_difference_gradient_check(theta_objective, theta, ...
            opts.gradientCheckStep, opts.gradientCheckTolerance, ...
            'IM-RCS logistic M-step theta');
    end
    [theta, theta_fval, theta_exitflag] = ...
        fminunc(theta_objective, theta, inner_options);

    new_log_lik = observed_loglik(parsed, rank_pos, theta, p, beta);
    improvement = new_log_lik - history.logLik(iter);
    relative_improvement = improvement / max(1, abs(history.logLik(iter)));
    if improvement < -opts.monotonicityTolerance
        monotonicity_violations = monotonicity_violations + 1;
    end

    history.logLik(iter + 1) = new_log_lik;
    history.improvement(iter) = improvement;
    history.relativeImprovement(iter) = relative_improvement;
    history.thetaObjective(iter) = theta_fval;
    history.thetaExitFlags(iter) = theta_exitflag;
    history.p(iter + 1, :) = p;
    history.beta(iter + 1, :) = beta;
    iter_done = iter;

    if opts.verbose
        fprintf(['IM-RCS iter %3d logLik %.9f improvement %.6g ' ...
            'relative %.3g innerExit %d\n'], iter, new_log_lik, ...
            improvement, relative_improvement, theta_exitflag);
    end
    if iter >= 2 && abs(relative_improvement) <= opts.emRelativeTolerance && ...
            improvement >= -opts.monotonicityTolerance
        stopped_on_tolerance = true;
        no_inner_failures = all(history.thetaExitFlags(1:iter) > 0);
        converged = theta_exitflag > 0 && no_inner_failures && ...
            monotonicity_violations == 0;
        break
    end
end

history.logLik = history.logLik(1:iter_done + 1);
history.improvement = history.improvement(1:iter_done);
history.relativeImprovement = history.relativeImprovement(1:iter_done);
history.thetaObjective = history.thetaObjective(1:iter_done);
history.thetaExitFlags = history.thetaExitFlags(1:iter_done);
history.p = history.p(1:iter_done + 1, :);
history.beta = history.beta(1:iter_done + 1, :);

model = struct();
model.id = 'imrcs';
model.theta = theta(:);
model.p = p(:);
model.lambda = flipud(cumsum(flipud(p(:))));
model.beta = beta(:);
model.rankings = opts.rankings;
model.opts = opts;
model.numFeatures = d;
model.numPatience = m;
model.numRankings = R;
model.trainNegLogLik = -history.logLik(end);
model.history = history;

final_improvement = NaN;
final_relative_improvement = NaN;
if iter_done > 0
    final_improvement = history.improvement(end);
    final_relative_improvement = history.relativeImprovement(end);
end
fitdiag = struct();
fitdiag.trainNegLogLik = model.trainNegLogLik;
fitdiag.numIterations = iter_done;
fitdiag.maxIterations = opts.imrcsOuterMaxIter;
fitdiag.converged = converged;
fitdiag.hitCap = ~converged && iter_done == opts.imrcsOuterMaxIter;
fitdiag.stoppedOnTolerance = stopped_on_tolerance;
fitdiag.finalImprovement = final_improvement;
fitdiag.finalRelativeImprovement = final_relative_improvement;
fitdiag.monotonicityViolations = monotonicity_violations;
fitdiag.innerExitFlags = history.thetaExitFlags;
fitdiag.numInnerFailures = sum(history.thetaExitFlags <= 0);
fitdiag.numSessions = parsed.numSessions;
fitdiag.numProducts = parsed.numProducts;
fitdiag.history = history;
fitdiag.gradientCheck = gradient_audit;
if converged
    fitdiag.status = 'converged';
elseif monotonicity_violations > 0
    fitdiag.status = 'monotonicity_violation';
elseif fitdiag.numInnerFailures > 0
    fitdiag.status = 'inner_optimizer_failure';
elseif fitdiag.hitCap
    fitdiag.status = 'hit_cap';
else
    fitdiag.status = 'stopped_nonconverged';
end
model.convergenceStatus = fitdiag.status;
end

function [Z, H] = expectation_step(parsed, rank_pos, theta, p, beta)
T = parsed.numSessions;
m = numel(p);
R = numel(beta);
Z = zeros(T, m);
H = zeros(parsed.numPurchases, R);
purchase_count = 0;
log_p = log(p(:)');
log_beta = log(beta(:)');

for t = 1:T
    s = parsed.sessions(t);
    eta = parsed.X(s.idx, :) * theta;
    log_one_minus_q = -softplus_stable(eta);
    if s.isPurchase
        purchase_count = purchase_count + 1;
        k = s.stageOf(s.choiceLocal);
        Z(t, k:m) = p(k:m) ./ sum(p(k:m));
        in_stage = find(s.stageOf == k);
        log_rank_weight = log_beta;
        for r = 1:R
            before = in_stage(rank_pos{t}(in_stage, r) ...
                < rank_pos{t}(s.choiceLocal, r));
            log_rank_weight(r) = log_rank_weight(r) ...
                + sum(log_one_minus_q(before));
        end
        H(purchase_count, :) = softmax_log(log_rank_weight);
    else
        log_patience_weight = log_p;
        for y = 1:m
            inspected = s.stageOf <= min(y, s.K);
            log_patience_weight(y) = log_patience_weight(y) ...
                + sum(log_one_minus_q(inspected));
        end
        Z(t, :) = softmax_log(log_patience_weight);
    end
end
end

function [exposure_weight, success_feature_sum] = ...
        theta_sufficient_statistics(parsed, rank_pos, Z, H, R)
N = parsed.numProducts;
d = parsed.numFeatures;
m = size(Z, 2);
exposure_weight = zeros(N, 1);
success_feature_sum = zeros(d, 1);
purchase_count = 0;

for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    idx = s.idx;
    if s.isPurchase
        purchase_count = purchase_count + 1;
        k = s.stageOf(s.choiceLocal);
        choice_global = idx(s.choiceLocal);
        exposure_weight(choice_global) = exposure_weight(choice_global) + 1;
        success_feature_sum = success_feature_sum + parsed.X(choice_global, :)';
        earlier = idx(s.stageOf < k);
        exposure_weight(earlier) = exposure_weight(earlier) + 1;
        in_stage = find(s.stageOf == k);
        for r = 1:R
            before = in_stage(rank_pos{t}(in_stage, r) ...
                < rank_pos{t}(s.choiceLocal, r));
            exposure_weight(idx(before)) = exposure_weight(idx(before)) ...
                + H(purchase_count, r);
        end
    else
        inspect_probability = zeros(s.K, 1);
        for k = 1:s.K
            inspect_probability(k) = sum(Z(t, k:m));
        end
        exposure_weight(idx) = exposure_weight(idx) ...
            + inspect_probability(s.stageOf);
    end
end
end

function [objective, gradient] = logistic_m_step(theta, X, exposure, success_x, ridge)
eta = X * theta;
objective = sum(exposure .* softplus_stable(eta)) - success_x' * theta;
if ridge > 0
    objective = objective + 0.5 * ridge * sum(theta .^ 2);
end
if nargout > 1
    q = sigmoid_stable(eta);
    gradient = X' * (exposure .* q) - success_x;
    if ridge > 0
        gradient = gradient + ridge * theta;
    end
end
end

function log_lik = observed_loglik(parsed, rank_pos, theta, p, beta)
m = numel(p);
R = numel(beta);
log_p = log(p(:)');
log_beta = log(beta(:)');
lambda = flipud(cumsum(flipud(p(:))));
log_lik = 0;

for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    eta = parsed.X(s.idx, :) * theta;
    log_q = -softplus_stable(-eta);
    log_one_minus_q = -softplus_stable(eta);
    if s.isPurchase
        k = s.stageOf(s.choiceLocal);
        in_stage = find(s.stageOf == k);
        log_rank_terms = -Inf(1, R);
        for r = 1:R
            before = in_stage(rank_pos{t}(in_stage, r) ...
                < rank_pos{t}(s.choiceLocal, r));
            log_rank_terms(r) = log_beta(r) + log_q(s.choiceLocal) ...
                + sum(log_one_minus_q(before));
        end
        log_lik = log_lik + log(lambda(k)) ...
            + sum(log_one_minus_q(s.stageOf < k)) ...
            + sequential_logsumexp(log_rank_terms);
    else
        log_terms = log_p;
        for y = 1:m
            inspected = s.stageOf <= min(y, s.K);
            log_terms(y) = log_terms(y) + sum(log_one_minus_q(inspected));
        end
        log_lik = log_lik + sequential_logsumexp(log_terms);
    end
end
end

function theta = initialize_theta(theta0, d)
if isempty(theta0)
    theta = zeros(d, 1);
else
    theta = theta0(:);
    if numel(theta) ~= d
        error('IMRCS:Theta0', 'theta0 has %d entries; expected %d.', numel(theta), d);
    end
end
end

function value = initialize_simplex(initial, n, name)
if isempty(initial)
    value = ones(1, n) ./ n;
else
    value = initial(:)';
    if numel(value) ~= n || any(~isfinite(value)) || any(value < 0) || sum(value) <= 0
        error('IMRCS:SimplexInit', '%s must contain %d nonnegative entries.', name, n);
    end
end
end

function value = normalize_simplex(value, floor_value)
value = max(value(:)', floor_value);
value = value ./ sum(value);
end

function options = inner_optimizer_options(opts)
if ~isempty(opts.innerOptimOptions)
    options = opts.innerOptimOptions;
else
    options = optimoptions('fminunc', 'Display', display_value(opts.verbose), ...
        'Algorithm', 'quasi-newton', 'SpecifyObjectiveGradient', true, ...
        'MaxIterations', opts.innerMaxIter, ...
        'MaxFunctionEvaluations', opts.maxFunctionEvaluations, ...
        'OptimalityTolerance', opts.optimizerOptimalityTolerance, ...
        'FunctionTolerance', opts.optimizerOptimalityTolerance, ...
        'StepTolerance', 1e-10);
end
end

function value = display_value(verbose)
if verbose, value = 'iter'; else, value = 'off'; end
end

function value = softplus_stable(x)
value = max(x, 0) + log1p(exp(-abs(x)));
end

function value = sigmoid_stable(x)
value = zeros(size(x));
positive = x >= 0;
value(positive) = 1 ./ (1 + exp(-x(positive)));
exp_x = exp(x(~positive));
value(~positive) = exp_x ./ (1 + exp_x);
end

function value = softmax_log(log_weight)
value = exp(log_weight - sequential_logsumexp(log_weight));
end
