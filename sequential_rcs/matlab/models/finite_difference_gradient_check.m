function audit = finite_difference_gradient_check(objective, x, step, tolerance, label)
%FINITE_DIFFERENCE_GRADIENT_CHECK Central-difference analytic-gradient audit.
if nargin < 5 || isempty(label)
    label = 'objective';
end
x = x(:);
[base_value, analytic] = objective(x);
analytic = analytic(:);
if ~isfinite(base_value) || numel(analytic) ~= numel(x) ...
        || any(~isfinite(analytic))
    error('SequentialModels:GradientCheckInput', ...
        '%s returned a nonfinite value or invalid analytic gradient.', label);
end

finite_difference = zeros(size(x));
for j = 1:numel(x)
    direction = zeros(size(x));
    direction(j) = step;
    plus = objective(x + direction);
    minus = objective(x - direction);
    finite_difference(j) = (plus - minus) / (2 * step);
end
absolute_error = abs(analytic - finite_difference);
scale = max([ones(size(x)), abs(analytic), abs(finite_difference)], [], 2);
relative_error = absolute_error ./ scale;

audit = struct();
audit.label = label;
audit.step = step;
audit.tolerance = tolerance;
audit.maxAbsoluteError = max(absolute_error, [], 'omitnan');
audit.maxRelativeError = max(relative_error, [], 'omitnan');
audit.passed = audit.maxRelativeError <= tolerance;
if ~audit.passed
    error('SequentialModels:GradientCheckFailed', ...
        '%s gradient max relative error %.6g exceeds %.6g.', ...
        label, audit.maxRelativeError, tolerance);
end
end
