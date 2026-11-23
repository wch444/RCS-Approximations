function prob = sqmnl_session_probability(eta, stage_of, tolerance)
%SQMNL_SESSION_PROBABILITY Independent-stage sequential MNL probabilities.
%
% For product i in stage k,
%   P(i) = prod_{r<k} 1/(1+V_r) * v_i/(1+V_k),
% and P(0) is the product of all stage-level outside probabilities.

if nargin < 3 || isempty(tolerance)
    tolerance = 1e-8;
end
eta = eta(:);
stage_of = stage_of(:);
if numel(eta) ~= numel(stage_of) || isempty(eta) || ...
        any(stage_of < 1) || any(stage_of ~= floor(stage_of))
    error('SQMNL:BadSession', 'eta and stage_of must describe a nonempty session.');
end

K = max(stage_of);
log_denom = zeros(K, 1);
for k = 1:K
    log_denom(k) = sequential_logsumexp([0; eta(stage_of == k)]);
end
log_reach = [0; -cumsum(log_denom(1:end - 1))];
log_prob_product = eta + log_reach(stage_of) - log_denom(stage_of);
prob = [exp(log_prob_product); exp(-sum(log_denom))];
assert_probability_vector(prob, tolerance, 'SQ-MNL');
end
