function value = sequential_logsumexp(x)
%SEQUENTIAL_LOGSUMEXP Stable log(sum(exp(x))) for a vector.
x = x(:);
xmax = max(x);
if isempty(x)
    value = -Inf;
elseif isinf(xmax)
    value = xmax;
else
    value = xmax + log(sum(exp(x - xmax)));
end
end
