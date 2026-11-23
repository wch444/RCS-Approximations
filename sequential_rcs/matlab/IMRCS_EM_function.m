function [model, metrics] = IMRCS_EM_function(train_data, test_data, opts)
%IMRCS_EM_FUNCTION Estimate the Impatient Mixed-RCS model by EM.
%
%   [MODEL, METRICS] = IMRCS_EM_FUNCTION(TRAIN_DATA, TEST_DATA, OPTS)
%   implements the EM algorithm in sequential.tex, Section "Estimation" and
%   Appendix "Implementation Details of the EM Algorithm".
%
%   The default data layout follows the Expedia .mat files in data.nosync:
%     column 1: query/session id
%     column 2: product id, zero for the outside/no-purchase row
%     column 4: displayed position, 100 for the outside/no-purchase row
%     column 5: chosen indicator, including the outside row on no-purchase
%     columns opts.featureStart:end: product features for q = sigmoid(x theta)
%
%   The finite ranking support Sigma_0 is represented by ranking rules. Each
%   rule induces a ranking inside each observed session from displayed product
%   attributes, which is the natural Expedia adaptation of a fixed support over
%   global product permutations.

if nargin < 2
    test_data = [];
end
if nargin < 3
    opts = struct();
end

opts = imrcs_default_opts(opts);
if isempty(opts.rankings)
    opts.rankings = imrcs_default_rankings(opts);
end
if ~isempty(opts.rngSeed)
    rng(opts.rngSeed, 'twister');
end

train = imrcs_prepare_data(train_data, opts);
num_rankings = numel(opts.rankings);
num_features = size(train.X, 2);
num_patience = train.maxStage;

theta = imrcs_init_theta(opts.theta0, num_features);
p = imrcs_normalize_simplex(imrcs_init_simplex(opts.p0, num_patience), opts.probFloor);
beta = imrcs_normalize_simplex(imrcs_init_simplex(opts.beta0, num_rankings), opts.probFloor);

optim_options = imrcs_optim_options(opts);

history.logLik = nan(opts.maxIter + 1, 1);
history.negLogLik = nan(opts.maxIter + 1, 1);
history.improvement = nan(opts.maxIter + 1, 1);
history.thetaObjective = nan(opts.maxIter, 1);
history.thetaExitFlag = nan(opts.maxIter, 1);
history.beta = nan(opts.maxIter + 1, num_rankings);
history.p = nan(opts.maxIter + 1, num_patience);

history.logLik(1) = imrcs_observed_loglik(train, theta, p, beta, opts);
history.negLogLik(1) = -history.logLik(1);
history.beta(1, :) = beta;
history.p(1, :) = p;

if opts.verbose
    fprintf('IMRCS EM: T=%d, products=%d, features=%d, Kmax=%d, rankings=%d\n', ...
        train.numSessions, train.numProducts, num_features, num_patience, num_rankings);
    fprintf('iter %3d logLik %.6f\n', 0, history.logLik(1));
end

iter_done = 0;
for iter = 1:opts.maxIter
    [Z, H] = imrcs_e_step(train, theta, p, beta, opts);

    p = imrcs_normalize_simplex(mean(Z, 1), opts.probFloor);
    if train.numPurchases > 0
        beta = imrcs_normalize_simplex(sum(H, 1) ./ train.numPurchases, opts.probFloor);
    end

    [exposureWeight, successFeatureSum] = imrcs_theta_weights(train, Z, H, num_rankings);
    theta_objective = @(b) imrcs_theta_objective(b, train.X, exposureWeight, ...
        successFeatureSum, opts.ridge);
    [theta, fval, exitflag] = fminunc(theta_objective, theta, optim_options);

    history.thetaObjective(iter) = fval;
    history.thetaExitFlag(iter) = exitflag;
    history.logLik(iter + 1) = imrcs_observed_loglik(train, theta, p, beta, opts);
    history.negLogLik(iter + 1) = -history.logLik(iter + 1);
    history.improvement(iter + 1) = history.logLik(iter + 1) - history.logLik(iter);
    history.beta(iter + 1, :) = beta;
    history.p(iter + 1, :) = p;

    iter_done = iter;
    if opts.verbose
        fprintf('iter %3d logLik %.6f improvement %.6g thetaExit %d\n', ...
            iter, history.logLik(iter + 1), history.improvement(iter + 1), exitflag);
    end

    if iter > 1 && history.improvement(iter + 1) <= opts.tol
        break
    end
end

keep = 1:(iter_done + 1);
history.logLik = history.logLik(keep);
history.negLogLik = history.negLogLik(keep);
history.improvement = history.improvement(keep);
history.beta = history.beta(keep, :);
history.p = history.p(keep, :);
history.thetaObjective = history.thetaObjective(1:iter_done);
history.thetaExitFlag = history.thetaExitFlag(1:iter_done);

model = struct();
model.theta = theta;
model.p = p(:);
model.lambda = flipud(cumsum(flipud(p(:))));
model.beta = beta(:);
model.rankings = opts.rankings;
model.opts = opts;
model.numFeatures = num_features;
model.numPatience = num_patience;
model.numRankings = num_rankings;
model.history = history;

metrics = struct();
metrics.train = imrcs_evaluate_parsed(train, theta, p, beta, opts);
if ~isempty(test_data)
    test = imrcs_prepare_data(test_data, opts);
    metrics.test = imrcs_evaluate_parsed(test, theta, p, beta, opts);
else
    metrics.test = [];
end
end

function opts = imrcs_default_opts(opts)
defaults = struct();
defaults.queryCol = 1;
defaults.itemCol = 2;
defaults.positionCol = 4;
defaults.choiceCol = 5;
defaults.featureStart = 8;
defaults.noPurchasePosition = 100;
defaults.stageSize = 10;
defaults.rankings = [];
defaults.maxIter = 50;
defaults.tol = 1e-4;
defaults.probFloor = 1e-10;
defaults.ridge = 0;
defaults.theta0 = [];
defaults.p0 = [];
defaults.beta0 = [];
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

function rankings = imrcs_default_rankings(opts)
fs = opts.featureStart;
rankings = struct('name', {}, 'column', {}, 'direction', {});
rankings(end + 1) = struct('name', 'position_asc', 'column', opts.positionCol, 'direction', 'ascend');
rankings(end + 1) = struct('name', 'price_low', 'column', fs + 1, 'direction', 'ascend');
rankings(end + 1) = struct('name', 'price_high', 'column', fs + 1, 'direction', 'descend');
rankings(end + 1) = struct('name', 'star_high', 'column', fs + 2, 'direction', 'descend');
rankings(end + 1) = struct('name', 'star_low', 'column', fs + 2, 'direction', 'ascend');
rankings(end + 1) = struct('name', 'location_high', 'column', fs + 3, 'direction', 'descend');
rankings(end + 1) = struct('name', 'location_low', 'column', fs + 3, 'direction', 'ascend');
rankings(end + 1) = struct('name', 'review_high', 'column', fs + 4, 'direction', 'descend');
rankings(end + 1) = struct('name', 'review_low', 'column', fs + 4, 'direction', 'ascend');
end

function theta = imrcs_init_theta(theta0, num_features)
if isempty(theta0)
    theta = zeros(num_features, 1);
else
    theta = theta0(:);
    if numel(theta) ~= num_features
        error('IMRCS:BadTheta0', 'opts.theta0 has %d entries, expected %d.', ...
            numel(theta), num_features);
    end
end
end

function p = imrcs_init_simplex(p0, n)
if isempty(p0)
    p = ones(1, n) ./ n;
else
    p = p0(:)';
    if numel(p) ~= n
        error('IMRCS:BadSimplexInit', 'Simplex initializer has %d entries, expected %d.', ...
            numel(p), n);
    end
end
end

function p = imrcs_normalize_simplex(p, floor_value)
p = max(p(:)', floor_value);
p = p ./ sum(p);
end

function optim_options = imrcs_optim_options(opts)
if ~isempty(opts.optimOptions)
    optim_options = opts.optimOptions;
    return
end

optim_options = optimoptions('fminunc', ...
    'Display', 'off', ...
    'Algorithm', 'quasi-newton', ...
    'SpecifyObjectiveGradient', true, ...
    'MaxIterations', 100, ...
    'MaxFunctionEvaluations', 5e4, ...
    'OptimalityTolerance', 1e-5);
end

function parsed = imrcs_prepare_data(data, opts)
if isempty(data)
    error('IMRCS:EmptyData', 'Input data is empty.');
end
if opts.featureStart > size(data, 2)
    error('IMRCS:BadFeatureStart', 'featureStart=%d exceeds data width %d.', ...
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
    'isPurchase', [], ...
    'rankPos', []);

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
        error('IMRCS:EmptySession', 'Session %g has no displayed products.', rows(1, opts.queryCol));
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
    sessions(t).rankPos = imrcs_rank_positions(product_rows, opts);

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

function rank_pos = imrcs_rank_positions(product_rows, opts)
n = size(product_rows, 1);
R = numel(opts.rankings);
rank_pos = zeros(n, R);
tie_break = product_rows(:, opts.positionCol);

for r = 1:R
    col = opts.rankings(r).column;
    if col < 1 || col > size(product_rows, 2)
        score = tie_break;
    else
        score = product_rows(:, col);
    end

    if strcmpi(opts.rankings(r).direction, 'descend')
        sort_key = [-score, tie_break];
    else
        sort_key = [score, tie_break];
    end
    [~, order] = sortrows(sort_key, [1 2]);
    rank_pos(order, r) = (1:n)';
end
end

function [Z, H] = imrcs_e_step(parsed, theta, p, beta, opts)
T = parsed.numSessions;
m = numel(p);
R = numel(beta);
Z = zeros(T, m);
H = zeros(parsed.numPurchases, R);
purchase_count = 0;

for t = 1:T
    s = parsed.sessions(t);
    X = parsed.X(s.idx, :);
    eta = X * theta;
    log_one_minus_q = -imrcs_softplus(eta);

    if s.isPurchase
        purchase_count = purchase_count + 1;
        kt = s.choiceStage;
        tail = sum(p(kt:m));
        if tail <= 0
            Z(t, kt:m) = 1 / (m - kt + 1);
        else
            Z(t, kt:m) = p(kt:m) ./ tail;
        end

        stage_idx = find(s.stageOf == kt);
        choice_local = s.choiceLocal;
        log_num = log(max(beta(:)', opts.probFloor));
        for r = 1:R
            before = stage_idx(s.rankPos(stage_idx, r) < s.rankPos(choice_local, r));
            log_num(r) = log_num(r) + sum(log_one_minus_q(before));
        end
        H(purchase_count, :) = imrcs_softmax_log(log_num);
    else
        stage_log = imrcs_stage_log_no_match(s, log_one_minus_q);
        log_num = log(max(p, opts.probFloor));
        for y = 1:m
            log_num(y) = log_num(y) + stage_log(min(y, s.K));
        end
        Z(t, :) = imrcs_softmax_log(log_num);
    end
end
end

function [exposureWeight, successFeatureSum] = imrcs_theta_weights(parsed, Z, H, num_rankings)
N = parsed.numProducts;
d = size(parsed.X, 2);
m = size(Z, 2);
exposureWeight = zeros(N, 1);
successFeatureSum = zeros(d, 1);
purchase_count = 0;

for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    idx = s.idx;

    if s.isPurchase
        purchase_count = purchase_count + 1;
        kt = s.choiceStage;
        choice_local = s.choiceLocal;

        choice_global = idx(choice_local);
        exposureWeight(choice_global) = exposureWeight(choice_global) + 1;
        successFeatureSum = successFeatureSum + parsed.X(choice_global, :)';

        earlier = idx(s.stageOf < kt);
        exposureWeight(earlier) = exposureWeight(earlier) + 1;

        stage_idx = find(s.stageOf == kt);
        for r = 1:num_rankings
            before = stage_idx(s.rankPos(stage_idx, r) < s.rankPos(choice_local, r));
            exposureWeight(idx(before)) = exposureWeight(idx(before)) + H(purchase_count, r);
        end
    else
        inspect_prob_by_stage = zeros(s.K, 1);
        for k = 1:s.K
            inspect_prob_by_stage(k) = sum(Z(t, k:m));
        end
        exposureWeight(idx) = exposureWeight(idx) + inspect_prob_by_stage(s.stageOf);
    end
end
end

function [f, g] = imrcs_theta_objective(theta, X, exposureWeight, successFeatureSum, ridge)
eta = X * theta;
f = sum(exposureWeight .* imrcs_softplus(eta)) - successFeatureSum' * theta;
if ridge > 0
    f = f + 0.5 * ridge * sum(theta .^ 2);
end

if nargout > 1
    q = imrcs_sigmoid(eta);
    g = X' * (exposureWeight .* q) - successFeatureSum;
    if ridge > 0
        g = g + ridge * theta;
    end
end
end

function logLik = imrcs_observed_loglik(parsed, theta, p, beta, opts)
logLik = 0;
m = numel(p);
R = numel(beta);
log_p = log(max(p, opts.probFloor));
log_beta = log(max(beta(:)', opts.probFloor));

for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    X = parsed.X(s.idx, :);
    eta = X * theta;
    log_q = eta - imrcs_softplus(eta);
    log_one_minus_q = -imrcs_softplus(eta);

    if s.isPurchase
        kt = s.choiceStage;
        if kt > m
            logLik = logLik + log(opts.probFloor);
            continue
        end
        log_tail = log(max(sum(p(kt:m)), opts.probFloor));
        log_before_stages = sum(log_one_minus_q(s.stageOf < kt));
        stage_idx = find(s.stageOf == kt);
        choice_local = s.choiceLocal;
        log_rank_terms = zeros(1, R);
        for r = 1:R
            before = stage_idx(s.rankPos(stage_idx, r) < s.rankPos(choice_local, r));
            log_rank_terms(r) = log_beta(r) + log_q(choice_local) + sum(log_one_minus_q(before));
        end
        logLik = logLik + log_tail + log_before_stages + imrcs_logsumexp(log_rank_terms);
    else
        stage_log = imrcs_stage_log_no_match(s, log_one_minus_q);
        log_terms = log_p;
        for y = 1:m
            log_terms(y) = log_terms(y) + stage_log(min(y, s.K));
        end
        logLik = logLik + imrcs_logsumexp(log_terms);
    end
end
end

function metrics = imrcs_evaluate_parsed(parsed, theta, p, beta, opts)
logLik = imrcs_observed_loglik(parsed, theta, p, beta, opts);
sq = 0;
weighted_sq = 0;
abs_scaled = 0;
num_outcomes = 0;
hit = zeros(3, 1);
hit_base = 0;

for t = 1:parsed.numSessions
    s = parsed.sessions(t);
    prob = imrcs_session_probability(parsed.X(s.idx, :), s, theta, p, beta);
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

function prob = imrcs_session_probability(X, s, theta, p, beta)
m = numel(p);
R = numel(beta);
n = size(X, 1);
eta = X * theta;
log_q = eta - imrcs_softplus(eta);
log_one_minus_q = -imrcs_softplus(eta);
prob_products = zeros(n, 1);

for i = 1:n
    kt = s.stageOf(i);
    if kt > m
        continue
    end
    log_tail = log(max(sum(p(kt:m)), realmin));
    log_before_stages = sum(log_one_minus_q(s.stageOf < kt));
    stage_idx = find(s.stageOf == kt);
    log_rank_terms = zeros(1, R);
    for r = 1:R
        before = stage_idx(s.rankPos(stage_idx, r) < s.rankPos(i, r));
        log_rank_terms(r) = log(max(beta(r), realmin)) + log_q(i) + sum(log_one_minus_q(before));
    end
    prob_products(i) = exp(log_tail + log_before_stages + imrcs_logsumexp(log_rank_terms));
end

stage_log = imrcs_stage_log_no_match(s, log_one_minus_q);
log_terms = log(max(p(:)', realmin));
for y = 1:m
    log_terms(y) = log_terms(y) + stage_log(min(y, s.K));
end
prob_no_purchase = exp(imrcs_logsumexp(log_terms));

prob = [prob_products; prob_no_purchase];
prob = max(prob, 0);
total_prob = sum(prob);
if total_prob > 1 + 1e-8
    prob = prob ./ total_prob;
end
end

function stage_log = imrcs_stage_log_no_match(s, log_one_minus_q)
stage_log = zeros(s.K, 1);
running = 0;
for k = 1:s.K
    running = running + sum(log_one_minus_q(s.stageOf == k));
    stage_log(k) = running;
end
end

function y = imrcs_softplus(x)
y = max(x, 0) + log1p(exp(-abs(x)));
end

function y = imrcs_sigmoid(x)
y = zeros(size(x));
pos = x >= 0;
y(pos) = 1 ./ (1 + exp(-x(pos)));
exp_x = exp(x(~pos));
y(~pos) = exp_x ./ (1 + exp_x);
y = min(max(y, realmin), 1 - eps);
end

function s = imrcs_logsumexp(x)
xmax = max(x);
if isinf(xmax)
    s = xmax;
else
    s = xmax + log(sum(exp(x - xmax)));
end
end

function p = imrcs_softmax_log(log_num)
den = imrcs_logsumexp(log_num);
if isinf(den)
    p = ones(size(log_num)) ./ numel(log_num);
else
    p = exp(log_num - den);
end
end
