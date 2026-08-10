function [results, means] = evaluate_all_models(dataset_name, markets, trials)
%EVALUATE_ALL_MODELS Evaluate all six paper models from versioned fresh fits.

here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(here)));
addpath(fullfile(root,'common','matlab','models'),'-begin');
addpath(fullfile(root,'common','matlab','workflow'),'-begin');
addpath(fullfile(root,'common','matlab','publishing'),'-begin');

if nargin < 1 || isempty(dataset_name), dataset_name = "trivago"; end
config = dataset_config(dataset_name);
if config.name ~= "trivago"
    error('evaluate_all_models:UnsupportedDataset', ...
        ['This evaluator supports only Trivago. Use the Expedia ' ...
         'expedia-local-predictive workflow.']);
end
if nargin < 2 || isempty(markets), markets = config.markets; end
if nargin < 3 || isempty(trials), trials = 1:10; end
markets = string(markets);
if ~isvector(markets) || any(ismissing(markets))
    error('evaluate_all_models:InvalidMarkets', ...
        'Markets must be a nonmissing vector of configured market names.');
end
markets = markets(:).';
canonical_indices = canonical_market_indices(markets, config);
trials = validate_trials(trials);

data_dir = fullfile(root, 'data', 'processed', char(config.name));
fit_dir = fullfile(root, 'opt_data', char(config.name));
output_dir = fullfile(fileparts(here), 'results');
processed_manifest_file = fullfile(data_dir,'manifest.json');
require_file(processed_manifest_file);
preflight_files(data_dir, fit_dir, config, markets, trials);

model_names = ["CMNL", "MNL", "MMNL", "SeqINV", "NSMNL", "SCM"];
n_row = numel(markets) * numel(trials) * numel(model_names);
market_col = strings(n_row, 1);
trial_col = zeros(n_row, 1);
model_col = strings(n_row, 1);
metric = nan(n_row, 7); % NLL, RMSE, weighted squared, MAPE, Hit1:3
row = 0;
task_metadata = repmat(struct('market','','trial',0,'data_file','', ...
    'fit_file','', ...
    'model_seeds',zeros(1,numel(model_names))), ...
    numel(markets)*numel(trials),1);
task_row = 0;

for trial = trials
    for market_index = 1:numel(markets)
        market = markets(market_index);
        canonical_market_index = canonical_indices(market_index);
        data_name = sprintf(config.data_pattern, char(market), trial);
        data_file = fullfile(data_dir, data_name);
        fit_file = fullfile(fit_dir, sprintf('fit_%s_%d.mat', char(market), trial));
        require_file(data_file);
        require_file(fit_file);
        D = load(data_file, 'data_test');
        F = load(fit_file, 'fit');
        fit_models = F.fit.models;
        test_full = D.data_test;
        validate_matrix(test_full, config);
        test_short = test_full(:, 1:config.feature_end);
        task_row = task_row+1;
        task_metadata(task_row).market = char(market);
        task_metadata(task_row).trial = trial;
        task_metadata(task_row).data_file = fullfile('data','processed', ...
            char(config.name),data_name);
        task_metadata(task_row).fit_file = fullfile('opt_data', ...
            char(config.name),sprintf('fit_%s_%d.mat',char(market),trial));

        for model_index = 1:numel(model_names)
            model = model_names(model_index);
            seed = 20260808 + 10000 * trial + 100 * canonical_market_index + model_index;
            task_metadata(task_row).model_seeds(model_index) = seed;
            rng(seed, 'twister');
            values = evaluate_one(model,fit_models,test_full,test_short,config);
            row = row + 1;
            market_col(row) = market;
            trial_col(row) = trial;
            model_col(row) = model;
            metric(row, :) = values;
        end
    end
end

results = table(market_col, trial_col, model_col, metric(:, 1), metric(:, 2), ...
    metric(:, 3), metric(:, 4), metric(:, 5), metric(:, 6), metric(:, 7), ...
    'VariableNames', {'Market', 'Trial', 'Model', 'NLL', 'RMSE', ...
    'WeightedSquaredError', 'MAPE', 'Hit1', 'Hit2', 'Hit3'});
means = aggregate_predictive_results(results, markets, model_names);

metadata = struct();
metadata.processed_manifest_file = fullfile( ...
    'data','processed',char(config.name),'manifest.json');
metadata.rng = 'twister';
metadata.master_seed = 20260808;
metadata.seed_formula = ...
    '20260808 + 10000*trial + 100*canonical_market_index + model_index';
metadata.loop_order = 'trial_market_model';
metadata.scm_parameter_order = 'beta_alpha';
metadata.tasks = task_metadata;
prepublish_validator = @() validate_evaluation_inputs( ...
    root,data_dir,fit_dir,config,metadata);
write_trivago_evaluation_bundle(results,means,output_dir,markets,trials, ...
    config.markets,model_names,metadata,prepublish_validator);
disp(means(:, {'Market', 'Model', 'RMSE', 'Hit1', 'Hit2', 'Hit3'}));
end


function trials = validate_trials(trials)
if ~isnumeric(trials) || ~isvector(trials) || isempty(trials) ...
        || ~isreal(trials) || any(~isfinite(trials)) ...
        || any(trials ~= fix(trials)) || any(trials < 1 | trials > 10) ...
        || numel(unique(trials)) ~= numel(trials)
    error('evaluate_all_models:InvalidTrials', ...
        'Trials must be unique integers from 1 through 10.');
end
trials = double(trials(:).');
end


function validate_matrix(data, config)
if ~isnumeric(data) || size(data, 2) < config.feature_end
    error('evaluate_all_models:InvalidData', ...
        'data_test must be numeric with at least %d columns.', config.feature_end);
end
if ~isempty(config.matrix_columns) && size(data, 2) ~= config.matrix_columns
    error('evaluate_all_models:InvalidData', ...
        'data_test must have exactly %d columns under the %s matrix contract.', ...
        config.matrix_columns, config.name);
end
if any(~isfinite(data), 'all')
    error('evaluate_all_models:InvalidData', 'data_test contains NaN or Inf.');
end
end


function values = evaluate_one(model, fit, test_full, test_short, config)
switch model
    case "SCM"
        [~, ~, hit, rmse, weighted, mape] = SCM_MNL_function( ...
            fit.scm, config.feature_start, [], test_short, [], true);
        nll = NaN;
    case "MNL"
        [~, nll, hit, rmse, weighted, mape] = Seq_MNL_function( ...
            fit.mnl, 1000, config.feature_start, [], test_full, [], true);
    case "MMNL"
        [~, ~, nll, hit, rmse, weighted, mape] = Mix_MNL_EM_function( ...
            fit.mmnl_alpha, fit.mmnl_beta, config.feature_start, ...
            [], test_full, [], true);
    case "CMNL"
        [~, ~, hit, rmse, weighted, mape] = Click_MNL_function( ...
            fit.cmnl, 100, config.feature_start, [], test_full, [], true);
        % For long assortments, the maintained CMNL predictor estimates a
        % positive choice probability with a finite Monte Carlo average.  A
        % finite draw can be exactly zero even when the model probability is
        % positive, so -log(p_hat) has a sampling-induced point mass at Inf.
        % Do not report that unstable quantity as a canonical test NLL.
        nll = NaN;
    case "SeqINV"
        [~, nll, hit, rmse, weighted, mape] = Seq_INV_function( ...
            fit.seqinv, 10, config.feature_start, [], test_short, [], true);
    case "NSMNL"
        [~, ~, nll, hit, rmse, weighted, mape] = NS_MNL_function( ...
            fit.nsmnl, 10, config.feature_start, [], test_short, [], true, ...
            fit.nsmnl_lambda);
end
values = [nll, rmse, weighted, mape, hit(:).'];
undefined_nll = ismember(model,["CMNL","SCM"]);
if undefined_nll
    if ~(isnumeric(nll) && isreal(nll) && isscalar(nll) && isnan(nll))
        error('evaluate_all_models:InvalidNLLContract', ...
            'Canonical %s test NLL must be undefined (NaN).',model);
    end
elseif ~(isnumeric(nll) && isreal(nll) && isscalar(nll) && isfinite(nll))
    error('evaluate_all_models:InvalidNLLContract', ...
        'Canonical %s test NLL must be finite.',model);
end
if any(~isfinite(values(2:end)))
    error('evaluate_all_models:NonFiniteMetric', ...
        'Non-finite predictive metric for %s.', model);
end
end


function require_file(path)
if ~isfile(path)
    error('evaluate_all_models:MissingFile', 'Missing required file: %s', path);
end
end


function preflight_files(data_dir, fit_dir, config, markets, trials)
for trial = trials
    for market = markets
        data_name = sprintf(config.data_pattern, char(market), trial);
        require_file(fullfile(data_dir, data_name));
        require_file(fullfile(fit_dir, sprintf('fit_%s_%d.mat', char(market), trial)));
    end
end
end


function validate_evaluation_inputs(root,data_dir,fit_dir,config,metadata)
manifest_file = fullfile(root,metadata.processed_manifest_file);
if ~isfile(manifest_file)
    error('evaluate_all_models:InputChanged', ...
        'The processed-data manifest is missing.');
end

expected_data_names = strings(numel(metadata.tasks),1);
expected_fit_names = strings(numel(metadata.tasks),1);
for index = 1:numel(metadata.tasks)
    task = metadata.tasks(index);
    data_file = fullfile(root,task.data_file);
    fit_file = fullfile(root,task.fit_file);
    if ~isfile(data_file) || ~isfile(fit_file)
        error('evaluate_all_models:InputChanged', ...
            'A required Trivago data or fit file is missing.');
    end
    [~,data_name,data_extension] = fileparts(data_file);
    [~,fit_name,fit_extension] = fileparts(fit_file);
    expected_data_names(index) = string(data_name) + string(data_extension);
    expected_fit_names(index) = string(fit_name) + string(fit_extension);
end

observed_data = dir(fullfile(data_dir,'10k_sample_*_p2000.mat'));
observed_fit = dir(fullfile(fit_dir,'fit_*.mat'));
if ~isequal(sort(string({observed_data.name})),sort(expected_data_names.')) ...
        || ~isequal(sort(string({observed_fit.name})),sort(expected_fit_names.')) ...
        || numel(metadata.tasks) ~= numel(config.markets)*10
    error('evaluate_all_models:InputChanged', ...
        'The Trivago data or fit directory is not the exact 50-task grid.');
end
end
