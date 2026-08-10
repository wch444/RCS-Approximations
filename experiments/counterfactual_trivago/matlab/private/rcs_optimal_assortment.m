function [offer, optimum] = rcs_optimal_assortment(revenue, q, capacity)
%RCS_OPTIMAL_ASSORTMENT Dynamic program for an ordered RCS assortment.

n_product = numel(revenue);
if n_product == 0 || capacity < 1 || capacity > n_product
    error('rcs_optimal_assortment:InvalidSize', ...
        'Require 1 <= capacity <= number of products.');
end

revenue = flip(revenue);
q = flip(q);
dp = zeros(n_product, capacity);
dp(1, :) = revenue(1) * q(1);

for item = 2:n_product
    for cardinality = 1:capacity
        if cardinality == 1
            dp(item, cardinality) = max( ...
                revenue(item) * q(item), dp(item - 1, cardinality));
        else
            dp(item, cardinality) = max( ...
                (1 - q(item)) * dp(item - 1, cardinality - 1) ...
                + q(item) * revenue(item), ...
                dp(item - 1, cardinality));
        end
    end
end

offer = false(1, n_product);
item = n_product;
cardinality = capacity;
while item >= 1 && cardinality >= 1
    if item == 1
        offer(item) = true;
    elseif dp(item, cardinality) ~= dp(item - 1, cardinality)
        offer(item) = true;
        cardinality = cardinality - 1;
    end
    item = item - 1;
end

offer = flip(offer);
optimum = dp(n_product, capacity);
end
