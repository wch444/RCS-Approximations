function fit_all_models(dataset_name, markets, trials)
%FIT_ALL_MODELS Fit the six models reported in the predictive tables.
%   Outputs one versioned MAT artifact per market/trial.  Fresh SCM fits are
%   always stored as [beta; alpha] and carry explicit order metadata.

here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(here)));
addpath(fullfile(root,'common','matlab','models'),'-begin');
addpath(fullfile(root,'common','matlab','workflow'),'-begin');

if nargin < 1 || isempty(dataset_name), dataset_name = "trivago"; end
config = dataset_config(dataset_name);
schema = unified_fit_schema();
if config.name ~= "trivago"
    error('fit_all_models:UnsupportedDataset', ...
        ['This entry point supports only Trivago. Use the Expedia ' ...
         'fit-expedia workflow to estimate all six Expedia models.']);
end
if nargin < 2 || isempty(markets), markets = config.markets; end
if nargin < 3 || isempty(trials), trials = 1:10; end
markets = string(markets);
canonical_indices = canonical_market_indices(markets, config);
trials = validate_trials(trials);

data_dir = fullfile(root, 'data', 'processed', char(config.name));
fit_dir = fullfile(root, 'opt_data', char(config.name));
if ~exist(fit_dir, 'dir'), mkdir(fit_dir); end
preflight_data(data_dir, config, markets, trials);

options = optimoptions('fminunc', ...
    'Display', 'off', ...
    'MaxIterations', 100, ...
    'MaxFunctionEvaluations', 5e4, ...
    'UseParallel', false, ...
    'OptimalityTolerance', 1e-4);
mmnl_em_options = struct( ...
    'MaxIterations',100, ...
    'RelativeTolerance',1e-7);

for trial = trials
    for market_index = 1:numel(markets)
        market = markets(market_index);
        canonical_market_index = canonical_indices(market_index);
        data_name = sprintf(config.data_pattern, char(market), trial);
        data_file = fullfile(data_dir, data_name);
        require_file(data_file, 'processed data');
        data = load(data_file, 'data_train', 'data_test');
        validate_matrix(data.data_train, config, 'data_train');
        validate_matrix(data.data_test, config, 'data_test');

        train_full = data.data_train;
        test_full = data.data_test;
        train_short = train_full(:, 1:config.feature_end);
        test_short = test_full(:, 1:config.feature_end);
        n_short = size(train_short(:, config.feature_start:end), 2);
        n_full = size(train_full(:, config.feature_start:end), 2);
        init_short = ones(n_short, 1) / 10;
        init_full = ones(n_full, 1) / 10;
        task_seed = 42 + 1000 * trial + canonical_market_index;

        fprintf('\n[%s %s trial %d] fitting SCM\n', config.name, market, trial);
        rng(task_seed + 10, 'twister');
        [scm, ~, ~, ~, ~, ~, ~] = SCM_MNL_function( ...
            init_short, config.feature_start, train_short, test_short, options, false);

        fprintf('[%s %s trial %d] fitting SeqINV\n', config.name, market, trial);
        rng(task_seed + 20, 'twister');
        [seqinv, ~, ~, ~, ~, ~, ~] = Seq_INV_function( ...
            init_short, 10, config.feature_start, train_short, test_short, options, false);

        fprintf('[%s %s trial %d] fitting NSMNL\n', config.name, market, trial);
        rng(task_seed + 30, 'twister');
        [nsmnl, nsmnl_lambda, ~, ~, ~, ~, ~, ~] = NS_MNL_function( ...
            init_short, 10, config.feature_start, train_short, test_short, options, false);

        fprintf('[%s %s trial %d] fitting MNL\n', config.name, market, trial);
        rng(task_seed + 40, 'twister');
        [mnl, ~, ~, ~, ~, ~, ~] = Seq_MNL_function( ...
            init_full, 1000, config.feature_start, train_full, test_full, options, false);

        fprintf('[%s %s trial %d] fitting CMNL\n', config.name, market, trial);
        rng(task_seed + 50, 'twister');
        [cmnl, ~, ~, ~, ~, ~, ~] = Click_MNL_function( ...
            init_full, 100, config.feature_start, train_full, test_full, options, false);

        fprintf('[%s %s trial %d] fitting MMNL\n', config.name, market, trial);
        mmnl_seed = paper_mmnl_seed(config.name,market,trial);
        rng(mmnl_seed, 'twister');
        n_component = 3;
        mmnl_alpha0 = ones(1, n_component) / n_component;
        mmnl_beta0 = rand(n_full, n_component);
        [mmnl_alpha, mmnl_beta] = ...
            Mix_MNL_EM_function(mmnl_alpha0, mmnl_beta0, ...
            config.feature_start, train_full, test_full, options, false, ...
            mmnl_em_options);
        fit = struct();
        fit.schema_version = schema.version;
        fit.artifact_type = schema.artifact_type;
        fit.dataset = char(config.name);
        fit.market = char(market);
        fit.trial = trial;
        fit.seed = task_seed;
        fit.rng = 'twister';
        fit.mmnl_seed = mmnl_seed;
        fit.mmnl_seed_rule = 'legacy_paper_mmnl_v1';
        fit.feature_start = config.feature_start;
        fit.feature_end = config.feature_end;
        fit.data_schema_version = config.data_schema_version;
        fit.matrix_columns = size(train_full, 2);
        fit.page_size = 10;
        fit.mmnl_components = n_component;
        fit.mmnl_estimator_version = schema.mmnl_estimator_version;
        fit.scm_parameter_order = schema.scm_parameter_order;
        fit.data_file = fullfile('data', 'processed', char(config.name), data_name);
        fit.train_rows = size(train_full,1);
        fit.test_rows = size(test_full,1);
        fit.train_sessions = numel(unique(train_full(:,1)));
        fit.test_sessions = numel(unique(test_full(:,1)));
        fit.models = struct( ...
            'scm', scm, ...
            'mnl', mnl, ...
            'mmnl_alpha', mmnl_alpha, ...
            'mmnl_beta', mmnl_beta, ...
            'cmnl', cmnl, ...
            'seqinv', seqinv, ...
            'nsmnl', nsmnl, ...
            'nsmnl_lambda', nsmnl_lambda);
        if ~isequal(fieldnames(fit.models),schema.model_fields(:))
            error('fit_all_models:InvalidModelSchema', ...
                'The unified model parameter fields are not canonical.');
        end
        fit = orderfields(fit,schema.trivago_fields);
        output_name = sprintf('fit_%s_%d.mat', char(market), trial);
        output_file = fullfile(fit_dir, output_name);
        temporary_file = string(tempname(fit_dir)) + ".mat";
        save(char(temporary_file), 'fit', '-v7');
        [ok, message] = movefile(char(temporary_file), output_file, 'f');
        if ~ok, error('fit_all_models:MoveFailed', '%s', message); end
        fprintf('saved %s\n', output_file);
    end
end
end


function require_file(path, label)
if ~isfile(path)
    error('fit_all_models:MissingFile', 'Missing %s: %s', label, path);
end
end


function trials = validate_trials(trials)
if ~isnumeric(trials) || ~isvector(trials) || isempty(trials) ...
        || ~isreal(trials) || any(~isfinite(trials)) ...
        || any(trials ~= fix(trials)) || any(trials < 1 | trials > 10) ...
        || numel(unique(trials)) ~= numel(trials)
    error('fit_all_models:InvalidTrials', ...
        'Trials must be unique integers from 1 through 10.');
end
trials = double(trials(:).');
end


function preflight_data(data_dir, config, markets, trials)
for trial = trials
    for market = markets
        data_name = sprintf(config.data_pattern, char(market), trial);
        require_file(fullfile(data_dir, data_name), 'processed data');
    end
end
end


function validate_matrix(data, config, label)
if ~isnumeric(data) || size(data, 2) < config.feature_end
    error('fit_all_models:InvalidData', ...
        '%s must be numeric with at least %d columns.', label, config.feature_end);
end
if ~isempty(config.matrix_columns) && size(data, 2) ~= config.matrix_columns
    error('fit_all_models:InvalidData', ...
        '%s must have exactly %d columns under the %s matrix contract.', ...
        label, config.matrix_columns, config.name);
end
if any(~isfinite(data), 'all')
    error('fit_all_models:InvalidData', '%s contains NaN or Inf.', label);
end
ids = data(:, 1);
if any(diff(ids) < 0)
    error('fit_all_models:InvalidData', '%s sessions are not sorted.', label);
end
end
