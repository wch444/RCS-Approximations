function prob = nsmnl_session_probability(eta, stage_of, lambda, tolerance)
%NSMNL_SESSION_PROBABILITY Stop-page-marginalized MNL probabilities.
%
% The stopping page is latent at prediction time.  No clicks enter this
% function.  Conditional on stop page y, the choice set contains stages
% 1:min(y,K) plus the outside option.

if nargin < 4 || isempty(tolerance)
    tolerance = 1e-8;
end
eta = eta(:);
stage_of = stage_of(:);
lambda = lambda(:);
if numel(eta) ~= numel(stage_of) || isempty(eta) || isempty(lambda)
    error('NSMNL:BadSession', 'eta, stage_of, and lambda must be nonempty.');
end
assert_simplex(lambda, tolerance, 'NS-MNL stopping distribution');

K = max(stage_of);
m = numel(lambda);
log_lambda = log(lambda);
log_denom = zeros(m, 1);
for y = 1:m
    viewed = stage_of <= min(y, K);
    log_denom(y) = sequential_logsumexp([0; eta(viewed)]);
end

log_prob_product = -Inf(numel(eta), 1);
for i = 1:numel(eta)
    k = stage_of(i);
    if k <= m
        mix = log_lambda(k:m) - log_denom(k:m);
        log_prob_product(i) = eta(i) + sequential_logsumexp(mix);
    end
end
log_prob_outside = sequential_logsumexp(log_lambda - log_denom);
prob = [exp(log_prob_product); exp(log_prob_outside)];
assert_probability_vector(prob, tolerance, 'NS-MNL');
end

function assert_simplex(p, tolerance, context)
if any(~isfinite(p)) || any(p < 0) || abs(sum(p) - 1) > tolerance
    error('SequentialModels:InvalidSimplex', ...
        '%s must be a finite nonnegative vector summing to one.', context);
end
end
