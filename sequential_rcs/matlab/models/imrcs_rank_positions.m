function rank_pos = imrcs_rank_positions(product_rows, rankings, opts)
%IMRCS_RANK_POSITIONS Ranking positions with deterministic display/item ties.
opts = sequential_model_options(opts);
n = size(product_rows, 1);
R = numel(rankings);
rank_pos = zeros(n, R);
row_tie = (1:n)';
tie_key = [product_rows(:, opts.positionCol), ...
    product_rows(:, opts.itemCol), row_tie];

for r = 1:R
    col = rankings(r).column;
    if col < 1 || col > size(product_rows, 2)
        error('IMRCS:RankingColumn', ...
            'Ranking %s requests unavailable column %d.', rankings(r).name, col);
    end
    score = product_rows(:, col);
    if strcmpi(rankings(r).direction, 'descend')
        key = [-score, tie_key];
    elseif strcmpi(rankings(r).direction, 'ascend')
        key = [score, tie_key];
    else
        error('IMRCS:RankingDirection', ...
            'Ranking %s has invalid direction %s.', ...
            rankings(r).name, rankings(r).direction);
    end
    [~, order] = sortrows(key, 1:size(key, 2));
    rank_pos(order, r) = (1:n)';
end
end
