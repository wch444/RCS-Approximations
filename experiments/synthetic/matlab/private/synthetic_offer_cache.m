function cache = synthetic_offer_cache(n, b)
%SYNTHETIC_OFFER_CACHE Cache every offer mask with cardinality at most b.
%   The first row is the empty offer.  For every product dimension, the
%   cache stores both the rows containing that product and the corresponding
%   rows with that product removed.  These indices support an exact weighted
%   subset-zeta transform without odds q/(1-q).

if ~isnumeric(n) || ~isscalar(n) || ~isfinite(n) || n ~= fix(n) ...
        || n < 1 || n > 20
    error('synthetic_offer_cache:InvalidN', ...
        ['n must be an integer in [1, 20].  The dense mask lookup is ' ...
        'bounded at the canonical production maximum.']);
end
if ~isnumeric(b) || ~isscalar(b) || ~isfinite(b) || b ~= fix(b) ...
        || b < 1 || b > n
    error('synthetic_offer_cache:InvalidB', ...
        'b must be an integer in [1, n].');
end
n = double(n);
b = double(b);

n_nonempty = sum(arrayfun(@(cardinality) nchoosek(n, cardinality), 1:b));
offers = false(n_nonempty + 1, n);
powers = uint32(2 .^ (0:n-1));
masks = zeros(n_nonempty + 1, 1, 'uint32');
construction_parent_masks = zeros(n_nonempty + 1, 1, 'uint32');
construction_items = zeros(n_nonempty + 1, 1, 'uint8');
level_rows = cell(1, b);
row = 1;
for cardinality = 1:b
    combinations = nchoosek(1:n, cardinality);
    block = false(size(combinations, 1), n);
    for column = 1:cardinality
        linear = sub2ind(size(block), ...
            (1:size(block, 1))', combinations(:, column));
        block(linear) = true;
    end
    index = row + (1:size(block, 1));
    offers(index, :) = block;
    combination_powers = reshape(powers(combinations), size(combinations));
    masks(index) = uint32(sum(double(combination_powers), 2));
    if cardinality > 1
        construction_parent_masks(index) = uint32(sum( ...
            double(combination_powers(:, 1:end-1)), 2));
    end
    construction_items(index) = uint8(combinations(:, end));
    level_rows{cardinality} = uint32(index(:));
    row = row + size(block, 1);
end

if numel(unique(masks)) ~= numel(masks) || masks(1) ~= 0
    error('synthetic_offer_cache:MaskConstructionFailed', ...
        'Offer masks are not unique or the empty mask is missing.');
end

mask_to_row = zeros(2^n, 1, 'uint32');
mask_to_row(double(masks) + 1) = uint32((1:numel(masks))');
construction_parent_rows = zeros(size(masks), 'uint32');
construction_parent_rows(2:end) = mask_to_row( ...
    double(construction_parent_masks(2:end)) + 1);
if any(construction_parent_rows(2:end) == 0)
    error('synthetic_offer_cache:MissingConstructionParent', ...
        'A feasible offer mask is missing its construction parent.');
end
selected_rows = cell(1, n);
parent_rows = cell(1, n);
for item = 1:n
    selected = uint32(find(offers(:, item)));
    parent_mask = bitset(masks(selected), item, 0);
    parent = mask_to_row(double(parent_mask) + 1);
    if any(parent == 0)
        error('synthetic_offer_cache:MissingParent', ...
            'A feasible offer mask is missing its parent mask.');
    end
    selected_rows{item} = selected;
    parent_rows{item} = parent;
end

cache.schema_version = 1;
cache.n = n;
cache.b = b;
cache.offers = offers;
cache.masks = masks;
cache.powers = powers;
cache.mask_to_row = mask_to_row;
cache.construction_items = construction_items;
cache.construction_parent_rows = construction_parent_rows;
cache.level_rows = level_rows;
cache.selected_rows = selected_rows;
cache.parent_rows = parent_rows;
end
