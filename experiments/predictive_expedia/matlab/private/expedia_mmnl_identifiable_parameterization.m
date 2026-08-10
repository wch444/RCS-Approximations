function [train_reduced,test_reduced,beta0_reduced,metadata] = ...
        expedia_mmnl_identifiable_parameterization( ...
            train_full,test_full,beta0_full,feature_start)
%EXPEDIA_MMNL_IDENTIFIABLE_PARAMETERIZATION Identify the Expedia MMNL design.
%   The Expedia matrices contain an item intercept followed by a
%   mutually exclusive collection of position-bin indicators.  This helper
%   uses position_bin_0 as the reference category and removes any other
%   position bin that is identically zero in the training split.  The full
%   initialization is transformed before columns are removed, so the
%   reference-category change is utility-equivalent on every training row.

parameterization = 'position_bin_0_reference_and_train_zero_v1';
expected_feature_start = 8;
position_matrix_start = 21;
reference_bin = 0;
intercept_feature_index = 1;

if ~isnumeric(feature_start) || ~isscalar(feature_start) ...
        || ~isreal(feature_start) || ~isfinite(feature_start) ...
        || feature_start ~= expected_feature_start
    error('expedia_mmnl_identifiable_parameterization:InvalidFeatureStart', ...
        'The Expedia feature_start must be 8.');
end
validate_matrix_pair(train_full,test_full,position_matrix_start);

full_feature_dimension = size(train_full,2)-feature_start+1;
position_feature_start = position_matrix_start-feature_start+1;
position_feature_indices = position_feature_start:full_feature_dimension;
reference_feature_index = position_feature_indices(reference_bin+1);
if ~isnumeric(beta0_full) || ~isreal(beta0_full) ...
        || size(beta0_full,1) ~= full_feature_dimension ...
        || isempty(beta0_full) || any(~isfinite(beta0_full),'all')
    error('expedia_mmnl_identifiable_parameterization:InvalidInitialization', ...
        ['beta0_full must be a finite full-feature-by-component ' ...
         'numeric matrix.']);
end

validate_position_contract(train_full,feature_start,position_matrix_start, ...
    'data_train');
validate_position_contract(test_full,feature_start,position_matrix_start, ...
    'data_test');
train_position = train_full(:,position_matrix_start:end);
if ~any(train_position(:,reference_bin+1) == 1)
    error('expedia_mmnl_identifiable_parameterization:MissingReferenceBin', ...
        'position_bin_0 must occur in the training split.');
end

zero_position_mask = all(train_position == 0,1);
zero_position_mask(reference_bin+1) = false;
constrained_zero_bins = find(zero_position_mask)-1;
constrained_zero_feature_indices = ...
    position_feature_indices(zero_position_mask);
removed_feature_indices = [reference_feature_index, ...
    constrained_zero_feature_indices];
reduced_feature_indices = setdiff( ...
    1:full_feature_dimension,removed_feature_indices,'stable');
reduced_matrix_indices = [1:feature_start-1, ...
    feature_start-1+reduced_feature_indices];
train_reduced = train_full(:,reduced_matrix_indices);
test_reduced = test_full(:,reduced_matrix_indices);

% First perform the pure reference-category transformation on the full
% initialization.  Only then remove the reference and unidentified columns.
beta0_equivalent_full = beta0_full;
reference_coefficient = beta0_full(reference_feature_index,:);
beta0_equivalent_full(intercept_feature_index,:) = ...
    beta0_equivalent_full(intercept_feature_index,:) ...
    + reference_coefficient;
beta0_equivalent_full(position_feature_indices,:) = ...
    beta0_equivalent_full(position_feature_indices,:) ...
    - reference_coefficient;
beta0_equivalent_full(reference_feature_index,:) = 0;
beta0_reduced = beta0_equivalent_full(reduced_feature_indices,:);

full_train_rank = rank(train_full(:,feature_start:end));
train_rank = rank(train_reduced(:,feature_start:end));
reduced_feature_dimension = numel(reduced_feature_indices);
if train_rank ~= reduced_feature_dimension
    error('expedia_mmnl_identifiable_parameterization:RankDeficient', ...
        ['The reduced Expedia training design has rank %d but %d ' ...
         'retained features.'],train_rank,reduced_feature_dimension);
end

full_initial_utility = train_full(:,feature_start:end)*beta0_full;
reduced_initial_utility = ...
    train_reduced(:,feature_start:end)*beta0_reduced;
mapping_error = max(abs( ...
    full_initial_utility-reduced_initial_utility),[],'all');
mapping_tolerance = 64*eps(max(1,max(abs(full_initial_utility),[],'all')));
if mapping_error > mapping_tolerance
    error('expedia_mmnl_identifiable_parameterization:NonEquivalentMapping', ...
        'The reference-category mapping changed a training utility.');
end

metadata = struct();
metadata.parameterization = parameterization;
metadata.reference_bin = reference_bin;
metadata.reference_feature_index = reference_feature_index;
metadata.position_feature_indices = double(position_feature_indices);
metadata.constrained_zero_bins = double(constrained_zero_bins);
metadata.constrained_zero_feature_indices = ...
    double(constrained_zero_feature_indices);
metadata.reduced_feature_indices = double(reduced_feature_indices);
metadata.full_feature_dimension = full_feature_dimension;
metadata.reduced_feature_dimension = reduced_feature_dimension;
metadata.full_train_rank = full_train_rank;
metadata.train_rank = train_rank;
end


function validate_matrix_pair(train_full,test_full,position_matrix_start)
for item = {train_full,test_full}
    data = item{1};
    if ~isnumeric(data) || ~isreal(data) || ~ismatrix(data) ...
            || isempty(data) || size(data,2) < position_matrix_start ...
            || any(~isfinite(data),'all')
        error('expedia_mmnl_identifiable_parameterization:InvalidData', ...
            ['Training and test inputs must be finite nonempty real ' ...
             'Expedia matrices with position-bin columns.']);
    end
end
if size(train_full,2) ~= size(test_full,2)
    error('expedia_mmnl_identifiable_parameterization:InvalidData', ...
        'Training and test Expedia matrices must have the same width.');
end
end


function validate_position_contract( ...
        data,feature_start,position_matrix_start,label)
intercept = data(:,feature_start);
position = data(:,position_matrix_start:end);
if any(~ismember(intercept,[0,1])) ...
        || any(~ismember(position,[0,1]),'all') ...
        || any(sum(position,2) ~= intercept)
    error('expedia_mmnl_identifiable_parameterization:InvalidPositionDesign', ...
        ['%s must use a binary item intercept and exactly one position ' ...
         'bin for each inside option (none for the outside option).'],label);
end
end
