function parsed = prepare_sequential_sessions(data, opts)
%PREPARE_SEQUENTIAL_SESSIONS Validate and canonicalize Expedia sessions.
%
% Products are sorted by displayed position, then item id, then their input
% row.  The outside row is not included in parsed.X/productData.  Every
% session must contain exactly one observed outcome across products and the
% outside row.

opts = sequential_model_options(opts);
if isempty(data) || ~isnumeric(data) || ndims(data) ~= 2
    error('SequentialModels:EmptyData', 'Input data must be a nonempty numeric matrix.');
end
required_col = max([opts.queryCol, opts.itemCol, opts.positionCol, ...
    opts.choiceCol, opts.clickCol, opts.featureCols]);
if size(data, 2) < required_col
    error('SequentialModels:DataWidth', ...
        'Input has %d columns, but column %d is required.', size(data, 2), required_col);
end
if any(~isfinite(data(:, [opts.queryCol, opts.itemCol, opts.positionCol, ...
        opts.choiceCol, opts.clickCol, opts.featureCols])), 'all')
    error('SequentialModels:NonfiniteData', 'Required data columns must be finite.');
end

input_row = (1:size(data, 1))';
[~, grouped_order] = sortrows([data(:, opts.queryCol), input_row], [1 2]);
grouped_data = data(grouped_order, :);
session_start = [1; find(diff(grouped_data(:, opts.queryCol)) ~= 0) + 1];
session_end = [session_start(2:end) - 1; size(grouped_data, 1)];
query_ids = grouped_data(session_start, opts.queryCol);
T = numel(session_start);
sessions(T, 1) = struct('id', [], 'idx', [], 'stageOf', [], 'K', [], ...
    'choiceLocal', [], 'isPurchase', [], 'outsideChosen', [], ...
    'canonicalInputRows', []);

X_cell = cell(T, 1);
product_cell = cell(T, 1);
num_products = 0;
num_purchases = 0;
max_stage = 1;

for t = 1:T
    grouped_idx = session_start(t):session_end(t);
    rows = grouped_data(grouped_idx, :);
    input_idx = grouped_order(grouped_idx);
    % Item id zero is the authoritative outside-row marker.  Do not classify
    % a real product as outside merely because a long query has position 100
    % or larger.
    outside = rows(:, opts.itemCol) == 0;
    if sum(outside) ~= 1
        error('SequentialModels:OutsideRow', ...
            'Session %.17g must have exactly one outside row; found %d.', ...
            query_ids(t), sum(outside));
    end
    if sum(rows(:, opts.choiceCol) == 1) ~= 1 || ...
            any(rows(:, opts.choiceCol) ~= 0 & rows(:, opts.choiceCol) ~= 1)
        error('SequentialModels:ObservedChoice', ...
            'Session %.17g must have exactly one binary chosen outcome.', query_ids(t));
    end

    local_product_idx = find(~outside);
    if isempty(local_product_idx)
        error('SequentialModels:EmptySession', ...
            'Session %.17g contains no displayed products.', query_ids(t));
    end
    product_rows = rows(local_product_idx, :);
    row_tie = (1:size(product_rows, 1))';
    sort_key = [product_rows(:, opts.positionCol), ...
        product_rows(:, opts.itemCol), row_tie];
    [~, order] = sortrows(sort_key, [1 2 3]);
    product_rows = product_rows(order, :);
    canonical_input_rows = input_idx(local_product_idx(order));

    n = size(product_rows, 1);
    idx = (num_products + 1:num_products + n)';
    stage_of = ceil((1:n)' ./ opts.stageSize);
    K = stage_of(end);
    choice_local = find(product_rows(:, opts.choiceCol) == 1, 1, 'first');
    outside_chosen = rows(outside, opts.choiceCol) == 1;
    is_purchase = ~isempty(choice_local);
    if is_purchase == outside_chosen
        error('SequentialModels:ChoiceMapping', ...
            'Session %.17g has an inconsistent product/outside choice.', query_ids(t));
    end

    sessions(t).id = query_ids(t);
    sessions(t).idx = idx;
    sessions(t).stageOf = stage_of;
    sessions(t).K = K;
    sessions(t).choiceLocal = choice_local;
    sessions(t).isPurchase = is_purchase;
    sessions(t).outsideChosen = outside_chosen;
    sessions(t).canonicalInputRows = canonical_input_rows;

    X_cell{t} = product_rows(:, opts.featureCols);
    product_cell{t} = product_rows;
    num_products = num_products + n;
    num_purchases = num_purchases + is_purchase;
    max_stage = max(max_stage, K);
end

parsed = struct();
parsed.sessions = sessions;
parsed.X = cell2mat(X_cell);
parsed.productData = cell2mat(product_cell);
parsed.queryIds = query_ids;
parsed.numSessions = T;
parsed.numProducts = num_products;
parsed.numPurchases = num_purchases;
parsed.numFeatures = numel(opts.featureCols);
parsed.maxStage = max_stage;
parsed.opts = opts;
end
