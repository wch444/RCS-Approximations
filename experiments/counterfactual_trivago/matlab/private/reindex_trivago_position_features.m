function rebuilt = reindex_trivago_position_features(features, positions)
%REINDEX_TRIVAGO_POSITION_FEATURES Rebuild five relative-position indicators.
%
% Trivago full-model rows contain eleven non-position features followed by
% five one-hot indicators for relative display positions 0:4, 5:9, ...,
% 20:24.  A counterfactual assortment preserves the original display order,
% but its retained products occupy consecutive positions starting from zero.
% This helper leaves the eleven non-position columns unchanged and replaces
% the last five columns using the supplied zero-based counterfactual positions.

non_position_count = 11;
position_bin_count = 5;
position_bin_width = 5;
expected_feature_count = non_position_count + position_bin_count;

if ~isnumeric(features) || ~ismatrix(features) ...
        || size(features,2) ~= expected_feature_count ...
        || ~isreal(features) || any(~isfinite(features),'all')
    error('reindex_trivago_position_features:InvalidFeatures', ...
        'Features must be a finite real numeric matrix with 16 columns.');
end
if ~isnumeric(positions) || ~isvector(positions) || ~isreal(positions) ...
        || numel(positions) ~= size(features,1) ...
        || any(~isfinite(positions)) || any(positions ~= fix(positions)) ...
        || any(positions < 0 | positions >= position_bin_count*position_bin_width)
    error('reindex_trivago_position_features:InvalidPositions', ...
        'Positions must be zero-based integers from 0 through 24, one per row.');
end

rebuilt = features;
rebuilt(:,non_position_count+1:end) = 0;
if isempty(positions)
    return
end
positions = double(positions(:));
bins = floor(positions/position_bin_width) + 1;
rows = (1:size(rebuilt,1))';
columns = non_position_count + bins;
indices = sub2ind(size(rebuilt),rows,columns);
rebuilt(indices) = 1;
end
