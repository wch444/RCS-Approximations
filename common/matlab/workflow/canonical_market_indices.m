function indices = canonical_market_indices(markets, config)
%CANONICAL_MARKET_INDICES Stable seed keys for a market subset or reordering.

markets = string(markets);
if numel(unique(markets, 'stable')) ~= numel(markets)
    error('canonical_market_indices:DuplicateMarket', ...
        'A market may be requested at most once per run.');
end

[known, indices] = ismember(markets, config.markets);
if any(~known)
    unknown = strjoin(markets(~known), ', ');
    error('canonical_market_indices:UnknownMarket', ...
        'Market(s) %s are not configured for %s.', unknown, config.name);
end
end
