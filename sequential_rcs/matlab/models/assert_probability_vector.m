function audit = assert_probability_vector(prob, tolerance, context)
%ASSERT_PROBABILITY_VECTOR Fail on invalid probabilities without renormalizing.
if nargin < 2 || isempty(tolerance)
    tolerance = 1e-8;
end
if nargin < 3 || isempty(context)
    context = 'sequential model';
end
prob = prob(:);
if isempty(prob) || any(~isfinite(prob))
    error('SequentialModels:NonfiniteProbability', ...
        '%s produced an empty or nonfinite probability vector.', context);
end
min_prob = min(prob);
total = sum(prob);
if min_prob < -tolerance
    error('SequentialModels:NegativeProbability', ...
        '%s produced a minimum probability %.17g.', context, min_prob);
end
if abs(total - 1) > tolerance
    error('SequentialModels:ProbabilityMass', ...
        '%s probabilities sum to %.17g (tolerance %.3g).', ...
        context, total, tolerance);
end
audit = struct('sum', total, 'minimum', min_prob, ...
    'maximum', max(prob), 'tolerance', tolerance);
end
