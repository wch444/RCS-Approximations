function fit_mmnl_only(dataset_name, markets, trials)
%FIT_MMNL_ONLY Fit only Expedia MMNL from archived processed matrices.
%   The direct API requires the exact fixed 5-site by 10-trial matrix grid.
%   It never preprocesses data or fits/overwrites the other five benchmark
%   models.  It atomically updates MMNL inside each unified fit artifact.

here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(here)));
addpath(fullfile(root,'common','matlab','models'),'-begin');
addpath(fullfile(root,'common','matlab','workflow'),'-begin');
if nargin < 1 || isempty(dataset_name), dataset_name = "expedia"; end
require_expedia(dataset_name);
config = dataset_config("expedia");
schema = unified_fit_schema();
if nargin < 2 || isempty(markets), markets = config.markets; end
if nargin < 3 || isempty(trials), trials = 1:10; end
markets = string(markets);
canonical_market_indices(markets, config); % Validate membership/uniqueness.
markets = markets(:).';
trials = validate_trials(trials);

data_dir = fullfile(root, 'data', 'processed', char(config.name));
fit_dir = fullfile(root,'opt_data',char(config.name));
validate_data_grid(data_dir,config);
validate_fit_grid(fit_dir,config);
preflight_data(data_dir, config, markets, trials);

optimizer_options = optimoptions('fminunc', ...
    'Display', 'off', ...
    'MaxIterations', 100, ...
    'MaxFunctionEvaluations', 5e4, ...
    'UseParallel', false, ...
    'OptimalityTolerance', 1e-4);
em_options = struct('MaxIterations', 100, 'RelativeTolerance', 1e-7);
n_component = 3;

for trial = trials
    for market = markets
        data_name = sprintf(config.data_pattern, char(market), trial);
        data_file = fullfile(data_dir, data_name);
        data = load(data_file, 'data_train', 'data_test');
        if ~all(isfield(data, {'data_train', 'data_test'}))
            error('fit_mmnl_only:InvalidData', ...
                'Archived MAT must contain data_train and data_test.');
        end
        validate_matrix(data.data_train, config, 'data_train');
        validate_matrix(data.data_test, config, 'data_test');
        if size(data.data_train, 2) ~= size(data.data_test, 2)
            error('fit_mmnl_only:InconsistentData', ...
                'Training and test matrices must have the same width.');
        end
        output_name = sprintf('fit_%s_%d.mat',char(market),trial);
        output_file = fullfile(fit_dir,output_name);
        stored = load(output_file,'fit');
        if ~isfield(stored,'fit') || ~isstruct(stored.fit) ...
                || ~isscalar(stored.fit)
            error('fit_mmnl_only:InvalidFit', ...
                'Unified MAT must contain one scalar fit: %s.',output_file);
        end
        models = validate_unified_fit(stored.fit,'expedia',market,trial, ...
            data.data_train,data.data_test);

        full_feature_dimension = ...
            size(data.data_train, 2) - config.feature_start + 1;
        alpha0 = ones(1, n_component) / n_component;
        seed = paper_mmnl_seed(config.name, market, trial);
        rng(seed, 'twister');
        % Preserve the paper's full-dimensional random stream exactly, then
        % map that draw into the identified parameterization.
        beta0_full = rand(full_feature_dimension, n_component);
        [train_reduced,test_reduced,beta0_reduced,parameterization] = ...
            expedia_mmnl_identifiable_parameterization( ...
                data.data_train,data.data_test,beta0_full, ...
                config.feature_start);

        fprintf('[%s %s trial %d] fitting MMNL only\n', ...
            config.name, market, trial);
        [alpha, beta_reduced] = ...
            Mix_MNL_EM_function(alpha0, beta0_reduced, ...
                config.feature_start,train_reduced,test_reduced, ...
                optimizer_options, false, ...
                em_options);
        validate_parameters(alpha,beta_reduced, ...
            parameterization.reduced_feature_dimension,n_component);
        beta = zeros(full_feature_dimension,n_component);
        beta(parameterization.reduced_feature_indices,:) = beta_reduced;
        validate_parameters(alpha,beta,full_feature_dimension,n_component);

        models.mmnl_alpha = alpha;
        models.mmnl_beta = beta;
        models = orderfields(models,schema.model_fields);

        fit = struct();
        fit.schema_version = schema.version;
        fit.artifact_type = schema.artifact_type;
        fit.dataset = char(config.name);
        fit.market = char(market);
        fit.trial = trial;
        fit.seed = seed;
        fit.rng = 'twister';
        fit.mmnl_seed = seed;
        fit.mmnl_seed_rule = 'legacy_paper_mmnl_v1';
        fit.feature_start = config.feature_start;
        fit.feature_end = config.feature_end;
        fit.data_schema_version = config.data_schema_version;
        fit.matrix_columns = size(data.data_train, 2);
        fit.page_size = 10;
        fit.data_file = fullfile( ...
            'data', 'processed', char(config.name), data_name);
        fit.train_rows = size(data.data_train,1);
        fit.test_rows = size(data.data_test,1);
        fit.train_sessions = numel(unique(data.data_train(:,1)));
        fit.test_sessions = numel(unique(data.data_test(:,1)));
        fit.mmnl_components = n_component;
        fit.mmnl_estimator_version = schema.mmnl_estimator_version;
        fit.scm_parameter_order = schema.scm_parameter_order;
        fit.parameterization = parameterization.parameterization;
        fit.reference_bin = parameterization.reference_bin;
        fit.reference_feature_index = ...
            parameterization.reference_feature_index;
        fit.position_feature_indices = ...
            parameterization.position_feature_indices;
        fit.constrained_zero_bins = ...
            parameterization.constrained_zero_bins;
        fit.constrained_zero_feature_indices = ...
            parameterization.constrained_zero_feature_indices;
        fit.reduced_feature_indices = ...
            parameterization.reduced_feature_indices;
        fit.full_feature_dimension = ...
            parameterization.full_feature_dimension;
        fit.reduced_feature_dimension = ...
            parameterization.reduced_feature_dimension;
        fit.full_train_rank = parameterization.full_train_rank;
        fit.train_rank = parameterization.train_rank;
        fit.mmnl_beta0_full = beta0_full;
        fit.mmnl_beta0_reduced = beta0_reduced;
        fit.mmnl_beta_reduced = beta_reduced;
        fit.models = models;
        fit = orderfields(fit,schema.expedia_fields);
        validate_unified_fit(fit,'expedia',market,trial, ...
            data.data_train,data.data_test);
        atomic_save(output_file, fit);
        fprintf('saved %s\n',output_file);
    end
end
end


function require_expedia(dataset_name)
valid_text = (ischar(dataset_name) && isrow(dataset_name)) ...
    || (isstring(dataset_name) && isscalar(dataset_name) ...
        && ~ismissing(dataset_name));
if ~valid_text || ~strcmp(char(dataset_name), 'expedia')
    error('fit_mmnl_only:UnsupportedDataset', ...
        'The MMNL-only fitting workflow supports only Expedia.');
end
end


function validate_data_grid(data_dir,config)
[~,~,expected_filenames] = expected_archive_grid(config);
mat_entries = dir(fullfile(data_dir, '*.mat'));
mat_names = sort(string({mat_entries.name}).');
if numel(mat_names) ~= 50 ...
        || ~isequal(mat_names, sort(expected_filenames))
    error('fit_mmnl_only:ArchiveMismatch', ...
        'The Expedia archive directory must contain exactly the canonical 50 MAT files.');
end
end


function validate_fit_grid(fit_dir,config)
if ~isfolder(fit_dir)
    error('fit_mmnl_only:FitGridMismatch', ...
        'Missing unified Expedia fit directory: %s.',fit_dir);
end
expected = strings(numel(config.markets)*10,1);
row = 0;
for market = config.markets
    for trial = 1:10
        row = row+1;
        expected(row) = sprintf('fit_%s_%d.mat',char(market),trial);
    end
end
entries = dir(fullfile(fit_dir,'*.mat'));
names = sort(string({entries.name}).');
if numel(names) ~= numel(expected) || ~isequal(names,sort(expected))
    error('fit_mmnl_only:FitGridMismatch', ...
        'The unified Expedia fit directory must contain exactly 50 MAT files.');
end
for index = 1:numel(expected)
    variables = whos('-file',fullfile(fit_dir,char(expected(index))));
    if numel(variables) ~= 1 || ~strcmp(variables.name,'fit') ...
            || ~strcmp(variables.class,'struct')
        error('fit_mmnl_only:InvalidFit', ...
            'Every unified Expedia MAT must contain only the fit struct.');
    end
end
end


function [markets, trials, filenames] = expected_archive_grid(config)
n_task = numel(config.markets) * 10;
markets = strings(n_task,1);
trials = zeros(n_task,1);
filenames = strings(n_task,1);
row = 0;
for market = config.markets
    for trial = 1:10
        row = row + 1;
        markets(row) = market;
        trials(row) = trial;
        filenames(row) = sprintf(config.data_pattern, char(market), trial);
    end
end
end


function trials = validate_trials(trials)
if ~isnumeric(trials) || ~isvector(trials) || ~isreal(trials) ...
        || any(~isfinite(trials)) || any(trials ~= fix(trials)) ...
        || any(trials < 1 | trials > 10) ...
        || numel(unique(trials)) ~= numel(trials)
    error('fit_mmnl_only:InvalidTrials', ...
        'Trials must be unique integers from 1 through 10.');
end
trials = double(trials(:).');
end


function preflight_data(data_dir, config, markets, trials)
for trial = trials
    for market = markets
        data_name = sprintf(config.data_pattern, char(market), trial);
        require_file(fullfile(data_dir, data_name));
    end
end
end


function require_file(path)
if ~isfile(path)
    error('fit_mmnl_only:MissingFile', ...
        'Missing archived processed matrix: %s', path);
end
end


function validate_matrix(data, config, label)
if ~isnumeric(data) || ~ismatrix(data) || isempty(data) ...
        || size(data, 2) < config.feature_end
    error('fit_mmnl_only:InvalidData', ...
        '%s must be a nonempty numeric matrix with at least %d columns.', ...
        label, config.feature_end);
end
if ~isempty(config.matrix_columns) ...
        && size(data, 2) ~= config.matrix_columns
    error('fit_mmnl_only:InvalidData', ...
        '%s must have exactly %d columns for %s.', ...
        label, config.matrix_columns, config.name);
end
if any(~isfinite(data), 'all') || any(diff(data(:, 1)) < 0)
    error('fit_mmnl_only:InvalidData', ...
        '%s must be finite and sorted by session.', label);
end
end


function validate_parameters(alpha, beta, n_feature, n_component)
valid = isnumeric(alpha) && isreal(alpha) ...
    && isequal(size(alpha), [1, n_component]) ...
    && all(isfinite(alpha), 'all') && all(alpha >= 0) ...
    && abs(sum(alpha) - 1) <= 1e-10 ...
    && isnumeric(beta) && isreal(beta) ...
    && isequal(size(beta), [n_feature, n_component]) ...
    && all(isfinite(beta), 'all');
if ~valid
    error('fit_mmnl_only:InvalidEstimate', ...
        'MMNL parameters are nonfinite, malformed, or outside the simplex.');
end
end


function atomic_save(output_file, fit)
folder = fileparts(output_file);
temporary_file = string(tempname(folder)) + ".mat";
cleanup = onCleanup(@() delete_if_present(temporary_file));
save(char(temporary_file), 'fit', '-v7');
[ok, message] = movefile(char(temporary_file), output_file, 'f');
if ~ok
    error('fit_mmnl_only:MoveFailed', '%s', message);
end
clear cleanup
end


function delete_if_present(path)
if isfile(path), delete(path); end
end
