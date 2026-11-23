function prob = imrcs_session_probability(eta, stage_of, rank_pos, p, beta, tolerance)
%IMRCS_SESSION_PROBABILITY Impatient mixed-RCS probabilities.
%
% This function evaluates the model formula directly and deliberately does
% not renormalize the resulting vector.  Invalid mass is surfaced by the
% probability audit.

if nargin < 6 || isempty(tolerance)
    tolerance = 1e-8;
end
eta = eta(:);
stage_of = stage_of(:);
p = p(:);
beta = beta(:);
if isempty(eta) || numel(eta) ~= numel(stage_of) || ...
        size(rank_pos, 1) ~= numel(eta) || size(rank_pos, 2) ~= numel(beta)
    error('IMRCS:BadSession', 'Session arrays have incompatible dimensions.');
end
assert_simplex(p, tolerance, 'IM-RCS patience distribution');
assert_simplex(beta, tolerance, 'IM-RCS ranking distribution');

K = max(stage_of);
m = numel(p);
R = numel(beta);
lambda = flipud(cumsum(flipud(p)));
log_beta = log(beta);
log_q = -softplus_stable(-eta);
log_one_minus_q = -softplus_stable(eta);
log_prob_product = -Inf(numel(eta), 1);

for i = 1:numel(eta)
    k = stage_of(i);
    if k > m || lambda(k) <= 0
        continue
    end
    earlier_stages = sum(log_one_minus_q(stage_of < k));
    in_stage = find(stage_of == k);
    rank_terms = -Inf(R, 1);
    for r = 1:R
        before = in_stage(rank_pos(in_stage, r) < rank_pos(i, r));
        rank_terms(r) = log_beta(r) + log_q(i) + sum(log_one_minus_q(before));
    end
    log_prob_product(i) = log(lambda(k)) + earlier_stages ...
        + sequential_logsumexp(rank_terms);
end

log_p = log(p);
outside_terms = -Inf(m, 1);
for y = 1:m
    inspected = stage_of <= min(y, K);
    outside_terms(y) = log_p(y) + sum(log_one_minus_q(inspected));
end
prob = [exp(log_prob_product); exp(sequential_logsumexp(outside_terms))];
assert_probability_vector(prob, tolerance, 'IM-RCS');
end

function y = softplus_stable(x)
y = max(x, 0) + log1p(exp(-abs(x)));
end

function assert_simplex(p, tolerance, context)
if any(~isfinite(p)) || any(p < 0) || abs(sum(p) - 1) > tolerance
    error('SequentialModels:InvalidSimplex', ...
        '%s must be a finite nonnegative vector summing to one.', context);
end
end
