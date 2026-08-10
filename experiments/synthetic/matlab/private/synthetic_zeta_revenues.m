function expected_revenue = synthetic_zeta_revenues( ...
        model_index, attention, revenue, attraction, cache)
%SYNTHETIC_ZETA_REVENUES Exact revenues for all cached offer masks.
%   MODEL_INDEX=1 is CMNL and MODEL_INDEX=2 is fixed-order SCM.  Conditional
%   on a realized consideration set A, each model's revenue depends only on A.
%   Starting from those conditional revenues, the in-place recurrence
%
%       F(S) = (1-q_i) F(S\{i}) + q_i F(S),  i in S,
%
%   applies the independent-attention expectation along each product
%   dimension.  It is the weighted subset-zeta transform, restricted exactly
%   to masks |S| <= b.  Unlike an odds-based implementation, this convex
%   combination remains stable when q_i is equal or arbitrarily close to one.

if ~isscalar(model_index) || ~ismember(model_index, [1, 2])
    error('synthetic_zeta_revenues:InvalidModel', ...
        'model_index must be 1 (CMNL) or 2 (SCM).');
end
n = cache.n;
attention = validate_vector(attention, n, 'attention');
revenue = validate_vector(revenue, n, 'revenue');
attraction = validate_vector(attraction, n, 'attraction');
if any(attention < 0) || any(attention > 1) ...
        || any(revenue < 0) || any(attraction < 0)
    error('synthetic_zeta_revenues:InvalidPrimitive', ...
        'Attention must lie in [0,1], and revenues/attractions must be nonnegative.');
end

row_count = size(cache.offers, 1);
attraction_sum = zeros(row_count, 1);
if model_index == 1
    revenue_weighted_sum = zeros(row_count, 1);
else
    conditional = zeros(row_count, 1);
end

% Rows are grouped by cardinality.  Each construction parent removes the
% largest-index item, so all parent values are complete before the next
% cardinality is processed.  This computes every conditional revenue in
% O(number of feasible masks), without rebuilding an offer-by-item matrix.
for cardinality = 1:cache.b
    rows = cache.level_rows{cardinality};
    parents = cache.construction_parent_rows(rows);
    items = cache.construction_items(rows);
    item_attraction = attraction(items);
    parent_sum = attraction_sum(parents);
    attraction_sum(rows) = parent_sum + item_attraction;
    if model_index == 1
        revenue_weighted_sum(rows) = revenue_weighted_sum(parents) ...
            + revenue(items) .* item_attraction;
    else
        through = 1 + parent_sum + item_attraction;
        conditional(rows) = conditional(parents) ...
            + revenue(items) .* item_attraction ...
            ./ ((1 + parent_sum) .* through);
    end
end
if model_index == 1
    conditional = revenue_weighted_sum ./ (1 + attraction_sum);
end
conditional(1) = 0;

expected_revenue = conditional;
for item = 1:n
    selected = cache.selected_rows{item};
    parent = cache.parent_rows{item};
    expected_revenue(selected) = (1 - attention(item)) ...
        .* expected_revenue(parent) ...
        + attention(item) .* expected_revenue(selected);
end

if any(~isfinite(expected_revenue)) || any(expected_revenue < -1e-12)
    error('synthetic_zeta_revenues:InvalidRevenue', ...
        'The weighted subset-zeta transform returned invalid revenues.');
end
end


function value = validate_vector(value, n, name)
if ~isnumeric(value) || ~isvector(value) || numel(value) ~= n ...
        || any(~isfinite(value))
    error('synthetic_zeta_revenues:InvalidPrimitive', ...
        '%s must be a finite vector with n entries.', name);
end
value = double(value(:));
end
