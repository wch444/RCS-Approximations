function seed = paper_mmnl_seed(dataset_name,market,trial)
%PAPER_MMNL_SEED Reproduce the initialization used by the paper MMNL runs.

name = validatestring(char(dataset_name),{'expedia','trivago'});
switch name
    case 'expedia'
        % Historical fitting order in test_10k_expedia_mmnl.m.
        order = ["15","14","5","24","32"];
    case 'trivago'
        order = ["BR","US","JP","UK","DE"];
end
market_index = find(order == string(market),1);
if isempty(market_index)
    error('paper_mmnl_seed:UnknownMarket', ...
        'Unknown %s market for the paper MMNL seed: %s.',name,market);
end
if ~isnumeric(trial) || ~isscalar(trial) || ~isfinite(trial) ...
        || trial < 1 || trial ~= fix(trial)
    error('paper_mmnl_seed:InvalidTrial','trial must be a positive integer.');
end
seed = 42 + 1000*trial + market_index;
end
