function rankings = default_imrcs_rankings(opts)
%DEFAULT_IMRCS_RANKINGS Fixed nine-rule Expedia ranking support.
opts = sequential_model_options(opts);
rankings = struct('name', {}, 'column', {}, 'direction', {});
rankings(end + 1) = rule('position_asc', opts.positionCol, 'ascend');
rankings(end + 1) = rule('price_low', 9, 'ascend');
rankings(end + 1) = rule('price_high', 9, 'descend');
rankings(end + 1) = rule('star_high', 10, 'descend');
rankings(end + 1) = rule('star_low', 10, 'ascend');
rankings(end + 1) = rule('location_high', 11, 'descend');
rankings(end + 1) = rule('location_low', 11, 'ascend');
rankings(end + 1) = rule('review_high', 12, 'descend');
rankings(end + 1) = rule('review_low', 12, 'ascend');
end

function value = rule(name, column, direction)
value = struct('name', name, 'column', column, 'direction', direction);
end
