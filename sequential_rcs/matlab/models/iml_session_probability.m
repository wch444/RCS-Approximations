function prob = iml_session_probability(eta, stage_of, p, tolerance)
%IML_SESSION_PROBABILITY TeX-authoritative impatient MNL probabilities.
%
% Let C_k be cumulative utility mass through stage k and lambda_k the
% patience tail.  For i in stage k,
%   P(i) = lambda_k v_i / ((1+C_{k-1})(1+C_k)).
% The outside probability is
%   sum_y p_y / (1+C_{min(y,K)}).

if nargin < 4 || isempty(tolerance)
    tolerance = 1e-8;
end
eta = eta(:);
stage_of = stage_of(:);
p = p(:);
if numel(eta) ~= numel(stage_of) || isempty(eta) || isempty(p)
    error('IML:BadSession', 'eta, stage_of, and p must be nonempty.');
end
if any(~isfinite(p)) || any(p < 0) || abs(sum(p) - 1) > tolerance
    error('SequentialModels:InvalidSimplex', ...
        'IML patience probabilities must be finite, nonnegative, and sum to one.');
end

K = max(stage_of);
m = numel(p);
log_p = log(p);
lambda = flipud(cumsum(flipud(p)));
log_denom = zeros(K, 1);
for k = 1:K
    log_denom(k) = sequential_logsumexp([0; eta(stage_of <= k)]);
end

log_prob_product = -Inf(numel(eta), 1);
for i = 1:numel(eta)
    k = stage_of(i);
    if k <= m && lambda(k) > 0
        previous = 0;
        if k > 1
            previous = log_denom(k - 1);
        end
        log_prob_product(i) = log(lambda(k)) + eta(i) ...
            - previous - log_denom(k);
    end
end
outside_terms = zeros(m, 1);
for y = 1:m
    outside_terms(y) = log_p(y) - log_denom(min(y, K));
end
prob = [exp(log_prob_product); exp(sequential_logsumexp(outside_terms))];
assert_probability_vector(prob, tolerance, 'IML');
end
