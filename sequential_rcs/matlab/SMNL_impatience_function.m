function [model, metrics] = SMNL_impatience_function(train_data, test_data, opts)
%SMNL_IMPATIENCE_FUNCTION Direct MLE for the Sequential MNL with stage patience.
%
%   [MODEL, METRICS] = SMNL_IMPATIENCE_FUNCTION(TRAIN_DATA, TEST_DATA, OPTS)
%   maximizes the observed log-likelihood of the impatient SMNL over
%   (theta, p) jointly via fminunc. Customers traverse displayed products
%   stage by stage (stage size opts.stageSize). The number of stages a
%   customer is willing to inspect follows a discrete patience distribution
%   p on {1, ..., maxStage}, with lambda_k = sum_{j>=k} p_j the probability
%   of reaching stage k and lambda_1 = 1 by construction. Within each
%   inspected stage the customer makes one MNL choice over the stage's
%   products plus an outside option (utility zero), giving stage-level
%   no-purchase probability 1 / (1 + sum_{j in stage} exp(x_j'theta)).
%
%   p is parameterized via softmax logits alpha so the optimization is
%   unconstrained: p = softmax(alpha). The latent patience Y is marginalized
%   analytically; no EM is used.
%
%   Default data layout follows the Expedia .mat files in data.nosync:
%     column 1: query/session id
%     column 2: product id, zero for the outside/no-purchase row
%     column 4: displayed position, 100 for the outside/no-purchase row
%     column 5: chosen indicator, including the outside row on no-purchase
%     columns opts.featureStart:end: product features for x_i'theta

if nargin < 2
    test_data = [];
end
if nargin < 3
    opts = struct();
end

opts = smnli_default_opts(opts);
if ~isempty(opts.rngSeed)
    rng(opts.rngSeed, 'twister');
end

train = smnli_prepare_data(train_data, opts);
num_features = size(train.X, 2);
num_patience = train.maxStage;

theta0 = smnli_init_theta(opts.theta0, num_features);
p0 = smnli_normalize_simplex(smnli_init_simplex(opts.p0, num_patience), opts.probFloor);
alpha0 = log(max(p0(:), opts.probFloor));

x0 = [theta0(:); alpha0];

optim_options = smnli_optim_options(opts);

if opts.verbose
    fprintf('SMNL-Impatience MLE: T=%d, products=%d, features=%d, Kmax=%d, params=%d\n', ...
        train.numSessions, train.numProducts, num_features, num_patience, ...
        num_features + num_patience);
end

objfun = @(x) smnli_neg_loglik_and_grad(x, train, num_features, opts);

[x_opt, fval, exitflag, output] = fminunc(objfun, x0, optim_options);

theta = x_opt(1:num_features);
alpha = x_opt(num_features + 1:end);
p = smnli_softmax_simplex(alpha, opts.probFloor);

if opts.verbose
    fprintf('Optim done: exit=%d, iters=%d, funcCount=%d, negLogLik=%.6f\n', ...
        exitflag, output.iterations, output.funcCount, fval);
end

model = struct();
model.theta = theta;
model.p = p(:);
model.alpha = alpha(:);
model.lambda = flipud(cumsum(flipud(p(:))));
model.opts = opts;
model.numFeatures = num_features;
model.numPatience = num_patience;
model.optimExitFlag = exitflag;
model.optimOutput = output;
model.objective = fval;
model.trainNegLogLik = fval;
model.trainLogLik = -fval;

metrics = struct();
metrics.train = smnli_evaluate_parsed(train, theta, p, opts);
if ~isempty(test_data)
    test = smnli_prepare_data(test_data, opts);
    metrics.test = smnli_evaluate_parsed(test, theta, p, opts);
else
    metrics.test = [];
end
end

function opts = smnli_default_opts(opts)
defaults = struct();
defaults.queryCol = 1;
defaults.itemCol = 2;
defaults.positionCol = 4;
defaults.choiceCol = 5;
defaults.featureStart = 8;
defaults.noPurchasePosition = 100;
defaults.stageSize = 10;
defaults.maxIter = 200;
defaults.tol = 1e-6;
defaults.probFloor = 1e-10;
defaults.theta0 = [];
defaults.p0 = [];
defaults.optimOptions = [];
defaults.verbose = true;
defaults.rngSeed = 42;
defaults.maxSessions = [];

names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = defaults.(name);
    end
end
end

function theta = smnli_init_theta(theta0, num_features)
if isempty(theta0)
    theta = zeros(num_features, 1);
else
    theta = theta0(:);
    if numel(theta) ~= num_features
        error('SMNLI:BadTheta0', 'opts.theta0 has %d entries, expected %d.', ...
            numel(theta), num_features);
    end
end
end

function p = smnli_init_simplex(p0, n)
if isempty(p0)
    p = ones(1, n) ./ n;
else
    p = p0(:)';
    if numel(p) ~= n
        error('SMNLI:BadSimplexInit', 'Simplex initializer has %d entries, expected %d.', ...
            numel(p), n);
    end
end
end

function p = smnli_normalize_simplex(p, floor_value)
p = max(p(:)', floor_value);
p = p ./ sum(p);
end

function p = smnli_softmax_simplex(alpha, floor_value)
M = max(alpha);
ea = exp(alpha(:) - M);
p = ea / sum(ea);
p = max(p, floor_value);
p = p / sum(p);
end

function optim_options = smnli_optim_options(opts)
if ~isempty(opts.optimOptions)
    optim_options = opts.optimOptions;
    return
end
optim_options = optimoptions('fminunc', ...
    'Display', 'off', ...
    'Algorithm', 'quasi-newton', ...
    'SpecifyObjectiveGradient', true, ...
    'MaxIterations', opts.maxIter, ...
    'MaxFunctionEvaluations', 1e5, ...
    'OptimalityTolerance', opts.tol, ...
    'StepTolerance', 1e-10, ...
    'FunctionTolerance', 1e-8);
end

function parsed = smnli_prepare_data(data, opts)
if isempty(data)
    error('SMNLI:EmptyData', 'Input data is empty.');
end
if opts.featureStart > size(data, 2)
    error('SMNLI:BadFeatureStart', 'featureStart=%d exceeds data width %d.', ...
        opts.featureStart, size(data, 2));
end

session_start = [1; find(diff(data(:, opts.queryCol)) ~= 0) + 1];
session_end = [session_start(2:end) - 1; size(data, 1)];
if ~isempty(opts.maxSessions)
    keep = 1:min(opts.maxSessions, numel(session_start));
    session_start = session_start(keep);
    session_end = session_end(keep);
end

num_sessions = numel(session_start);
sessions(num_sessions, 1) = struct( ...
    'id', [], ...
    'idx', [], ...
    'stageOf', [], ...
    'K', [], ...
    'choiceLocal', [], ...
    'choiceStage', [], ...
    'isPurchase', []);

X_cell = cell(num_sessions, 1);
num_products = 0;
num_purchases = 0;
max_stage = 1;

for t = 1:num_sessions
    rows = data(session_start(t):session_end(t), :);
    outside = rows(:, opts.positionCol) >= opts.noPurchasePosition;
    if opts.itemCol > 0 && opts.itemCol <= size(rows, 2)
        outside = outside | rows(:, opts.itemCol) == 0;
    end
    product_rows = rows(~outside, :);
    if isempty(product_rows)
        product_rows = rows(1:max(size(rows, 1) - 1, 0), :);
    end

    [~, order] = sort(product_rows(:, opts.positionCol), 'ascend');
    product_rows = product_rows(order, :);

    n = size(product_rows, 1);
    if n == 0
        error('SMNLI:EmptySession', 'Session %g has no displayed products.', rows(1, opts.queryCol));
    end

    idx = (num_products + 1:num_products + n)';
    num_products = num_products + n;
    stage_of = ceil((1:n)' ./ opts.stageSize);
    K = max(stage_of);
    max_stage = max(max_stage, K);

    choice_local = find(product_rows(:, opts.choiceCol) == 1, 1, 'first');
    is_purchase = ~isempty(choice_local);
    if is_purchase
        choice_stage = stage_of(choice_local);
        num_purchases = num_purchases + 1;
    else
        choice_local = 0;
        choice_stage = 0;
    end

    sessions(t).id = rows(1, opts.queryCol);
    sessions(t).idx = idx;
    sessions(t).stageOf = stage_of;
    sessions(t).K = K;
    sessions(t).choiceLocal = choice_local;
    sessions(t).choiceStage = choice_stage;
    sessions(t).isPurchase = is_purchase;

    X_cell{t} = product_rows(:, opts.featureStart:end);
end

parsed = struct();
parsed.sessions = sessions;
parsed.X = cell2mat(X_cell);
parsed.numSessions = num_sessions;
parsed.numProducts = num_products;
parsed.numPurchases = num_purchases;
parsed.maxStage = max_stage;
end

function [f, g] = smnli_neg_loglik_and_grad(x, parsed, d, opts)
% x = [theta(d); alpha(m)]; p = softmax(alpha).
m = numel(x) - d;
theta = x(1:d);
alpha = x(d + 1:end);

% softmax(alpha) with stability
M_a = max(alpha);
ea = exp(alpha - M_a);
Z_a = sum(ea);
p = ea / Z_a;
log_p = alpha - M_a - log(Z_a);    % = log(p), stable

floor_p = opts.probFloor;
logLik = 0;
need_grad = nargout > 1;
if need_grad
    dL_dtheta = zeros(d, 1);
    dL_dp = zeros(m, 1);
end

for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    idx = s.idx;
    X = parsed.X(idx, :);
    eta = X * theta;

    [stage_log_denom, stage_prob] = smnli_stage_mnl(s, eta);

    if s.isPurchase
        kt = s.choiceStage;
        if kt > m
            logLik = logLik + log(floor_p);
            continue
        end

        lambda_kt = sum(p(kt:m));
        log_tail = log(max(lambda_kt, floor_p));
        log_before = -sum(stage_log_denom(1:kt - 1));
        log_buy = eta(s.choiceLocal) - stage_log_denom(kt);
        logLik = logLik + log_tail + log_before + log_buy;

        if need_grad
            grad_eta = zeros(numel(eta), 1);
            for k = 1:kt - 1
                in_stage = (s.stageOf == k);
                grad_eta(in_stage) = -stage_prob{k};
            end
            in_kt = (s.stageOf == kt);
            grad_eta(in_kt) = -stage_prob{kt};
            grad_eta(s.choiceLocal) = grad_eta(s.choiceLocal) + 1;
            dL_dtheta = dL_dtheta + X' * grad_eta;

            dL_dp(kt:m) = dL_dp(kt:m) + 1 / max(lambda_kt, floor_p);
        end
    else
        K = s.K;
        cum_neg_log = -cumsum(stage_log_denom);  % K x 1
        w = log_p;                               % m x 1
        for y = 1:m
            w(y) = w(y) + cum_neg_log(min(y, K));
        end
        wmax = max(w);
        ew = exp(w - wmax);
        Sw = sum(ew);
        logLik = logLik + wmax + log(Sw);

        if need_grad
            r = ew / Sw;                         % m x 1 posterior over patience Y
            lambda_post = flipud(cumsum(flipud(r)));   % lambda_post(k) = sum_{m>=k} r_m

            grad_eta = zeros(numel(eta), 1);
            kmax = min(K, m);
            for k = 1:kmax
                in_stage = (s.stageOf == k);
                grad_eta(in_stage) = -stage_prob{k} * lambda_post(k);
            end
            dL_dtheta = dL_dtheta + X' * grad_eta;

            dL_dp = dL_dp + r ./ max(p, floor_p);
        end
    end
end

f = -logLik;

if need_grad
    weighted_sum = dL_dp(:)' * p(:);
    dL_dalpha = p(:) .* (dL_dp(:) - weighted_sum);
    g = -[dL_dtheta; dL_dalpha];
end
end

function [stage_log_denom, stage_prob] = smnli_stage_mnl(s, eta)
% Per-stage MNL denominator and within-stage softmax probs (excluding the
% zero-utility outside option).
K = s.K;
stage_log_denom = zeros(K, 1);
stage_prob = cell(K, 1);
for k = 1:K
    eta_k = eta(s.stageOf == k);
    M = max([0; eta_k]);
    exp_k = exp(eta_k - M);
    exp_0 = exp(-M);
    denom = exp_0 + sum(exp_k);
    stage_log_denom(k) = M + log(denom);
    stage_prob{k} = exp_k / denom;
end
end

function logLik = smnli_observed_loglik(parsed, theta, p, opts)
logLik = 0;
m = numel(p);
log_p = log(max(p, opts.probFloor));

for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    eta = parsed.X(s.idx, :) * theta;
    stage_log_denom = smnli_stage_log_denom(s, eta);

    if s.isPurchase
        kt = s.choiceStage;
        if kt > m
            logLik = logLik + log(opts.probFloor);
            continue
        end
        log_tail = log(max(sum(p(kt:m)), opts.probFloor));
        log_before = -sum(stage_log_denom(1:kt - 1));
        log_buy = eta(s.choiceLocal) - stage_log_denom(kt);
        logLik = logLik + log_tail + log_before + log_buy;
    else
        cum_neg_log = -cumsum(stage_log_denom);
        log_terms = log_p;
        for y = 1:m
            log_terms(y) = log_terms(y) + cum_neg_log(min(y, s.K));
        end
        logLik = logLik + smnli_logsumexp(log_terms);
    end
end
end

function stage_log_denom = smnli_stage_log_denom(s, eta)
K = s.K;
stage_log_denom = zeros(K, 1);
for k = 1:K
    eta_k = eta(s.stageOf == k);
    stage_log_denom(k) = smnli_logsumexp([0; eta_k]);
end
end

function metrics = smnli_evaluate_parsed(parsed, theta, p, opts)
logLik = smnli_observed_loglik(parsed, theta, p, opts);
sq = 0;
weighted_sq = 0;
abs_scaled = 0;
num_outcomes = 0;
hit = zeros(3, 1);
hit_base = 0;

for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    prob = smnli_session_probability(parsed.X(s.idx, :), s, theta, p);
    y = zeros(numel(prob), 1);
    if s.isPurchase
        y(s.choiceLocal) = 1;
        product_prob = prob(1:end - 1);
        [~, order] = sort(product_prob, 'descend');
        hit_base = hit_base + 1;
        for k = 1:3
            hit(k) = hit(k) + ismember(s.choiceLocal, order(1:min(k, numel(order))));
        end
    else
        y(end) = 1;
    end
    sq = sq + sum((prob - y) .^ 2);
    weighted_sq = weighted_sq + sum((prob - y) .^ 2 ./ (0.5 + y));
    abs_scaled = abs_scaled + sum(abs(prob - y) ./ (0.5 + y));
    num_outcomes = num_outcomes + numel(prob);
end

metrics = struct();
metrics.logLik = logLik;
metrics.negLogLik = -logLik;
metrics.rmse = sqrt(sq / num_outcomes);
metrics.weightedSq = weighted_sq / num_outcomes;
metrics.chiSq100 = 100 * metrics.weightedSq;
metrics.mape = abs_scaled / num_outcomes;
if hit_base > 0
    metrics.hit = hit ./ hit_base;
else
    metrics.hit = nan(3, 1);
end
metrics.purchaseRate = hit_base / parsed.numSessions;
metrics.numSessions = parsed.numSessions;
metrics.numProducts = parsed.numProducts;
metrics.numOutcomes = num_outcomes;
end

function prob = smnli_session_probability(X, s, theta, p)
m = numel(p);
n = size(X, 1);
eta = X * theta;
stage_log_denom = smnli_stage_log_denom(s, eta);
lambda = flipud(cumsum(flipud(p(:))));
prob_products = zeros(n, 1);

for i = 1:n
    kt = s.stageOf(i);
    if kt > m
        continue
    end
    log_before = -sum(stage_log_denom(1:kt - 1));
    log_buy = eta(i) - stage_log_denom(kt);
    prob_products(i) = lambda(kt) * exp(log_before + log_buy);
end

cum_neg_log = -cumsum(stage_log_denom);
log_terms = log(max(p(:), realmin));
for y = 1:m
    log_terms(y) = log_terms(y) + cum_neg_log(min(y, s.K));
end
prob_no_purchase = exp(smnli_logsumexp(log_terms));

prob = [prob_products; prob_no_purchase];
prob = max(prob, 0);
total = sum(prob);
if total > 1 + 1e-8
    prob = prob ./ total;
end
end

function s = smnli_logsumexp(x)
xmax = max(x);
if isinf(xmax)
    s = xmax;
else
    s = xmax + log(sum(exp(x - xmax)));
end
end
