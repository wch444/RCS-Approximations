function results = evaluate_expedia_archived_models(markets, trials)
%EVALUATE_EXPEDIA_ARCHIVED_MODELS Evaluate five freshly fitted Expedia models.
%   This evaluator computes CMNL, MNL, SeqINV, NSMNL, and SCM metrics from
%   the 50 unified fit files against the 50 corrected Expedia matrices. Fixed
%   filenames define the canonical 5-site by 10-trial grid.  SCM parameters
%   use [beta; alpha].  CMNL and SCM
%   NLL are recorded as NaN because their maintained evaluators do not expose
%   a likelihood comparable to the other three models; all four paper metrics
%   (RMSE and Hit1:3), plus the auxiliary error metrics, must remain finite.
%
%   Only an exact 5-market x 10-trial x 5-model run atomically publishes
%   experiments/predictive_expedia/results/predictive_expedia_legacy_trials.csv.
%   Subset calls return a table without modifying that canonical output.

here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(here)));
addpath(fullfile(root,'common','matlab','models'),'-begin');
addpath(fullfile(root,'common','matlab','workflow'),'-begin');
addpath(fullfile(root,'common','matlab','publishing'),'-begin');
config = dataset_config('expedia');
if nargin < 1 || isempty(markets), markets = config.markets; end
if nargin < 2 || isempty(trials), trials = 1:10; end
markets = validate_markets(markets, config);
trials = validate_trials(trials);
data_dir = fullfile(root, 'data', 'processed', 'expedia');
fit_dir = fullfile(root,'opt_data','expedia');
output_dir = fullfile(fileparts(here), 'results');
[~,~,expected_filenames] = expected_grid(config);
require_exact_file_grid(data_dir,expected_filenames, ...
    '^10k_sample_\d+_\d+\.mat$',false, ...
    'evaluate_expedia_archived_models:ArchiveMismatch');
expected_fit_names = replace(expected_filenames,'10k_sample_','fit_');
require_exact_file_grid(fit_dir,expected_fit_names, ...
    '^fit_\d+_\d+\.mat$',false, ...
    'evaluate_expedia_archived_models:LegacyFitMismatch');

model_names = ["CMNL", "MNL", "SeqINV", "NSMNL", "SCM"];
canonical_model_names = ["CMNL", "MNL", "MMNL", "SeqINV", "NSMNL", "SCM"];
ordered_markets = config.markets(ismember(config.markets, markets));
ordered_trials = sort(trials);
n_row = numel(ordered_markets) * numel(ordered_trials) * numel(model_names);
market_col = strings(n_row,1);
trial_col = zeros(n_row,1);
model_col = strings(n_row,1);
metric = nan(n_row,7);
row = 0;

for market = ordered_markets
    canonical_market_index = find(config.markets == market,1);
    for trial = ordered_trials
        data_name = sprintf(config.data_pattern,char(market),trial);
        data_file = fullfile(data_dir,data_name);
        fit_name = sprintf('fit_%s_%d.mat',char(market),trial);
        fit_file = fullfile(fit_dir,fit_name);
        require_file(data_file, ...
            'evaluate_expedia_archived_models:ArchiveMismatch');
        require_file(fit_file, ...
            'evaluate_expedia_archived_models:LegacyFitMismatch');

        data_payload = load(data_file,'data_test');
        if ~isfield(data_payload,'data_test')
            error('evaluate_expedia_archived_models:InvalidData', ...
                'Archived MAT must contain data_test: %s.',data_name);
        end
        test_full = validate_test_matrix(data_payload.data_test,config);
        test_short = test_full(:,1:config.feature_end);

        fit_payload = load(fit_file,'fit');
        if ~isfield(fit_payload,'fit') || ~isstruct(fit_payload.fit) ...
                || ~isscalar(fit_payload.fit) ...
                || ~isfield(fit_payload.fit,'models')
            error('evaluate_expedia_archived_models:InvalidLegacyFit', ...
                'Unified MAT must contain one scalar fit.models: %s.',fit_name);
        end
        parameters = fit_payload.fit.models;

        for model_index = 1:numel(model_names)
            model = model_names(model_index);
            canonical_model_index = find(canonical_model_names == model,1);
            % Match evaluate_all_models exactly so a model/market/trial task
            % has the same Monte Carlo stream whether it is evaluated from a
            % fresh six-model artifact or this five-model evaluation block.
            seed = 20260808 + 10000*trial ...
                + 100*canonical_market_index + canonical_model_index;
            rng(seed,'twister');
            values = evaluate_one( ...
                model,parameters,test_full,test_short,config);

            row = row + 1;
            market_col(row) = market;
            trial_col(row) = trial;
            model_col(row) = model;
            metric(row,:) = values;
        end
    end
end

results = table(market_col,trial_col,model_col,metric(:,1),metric(:,2), ...
    metric(:,3),metric(:,4),metric(:,5),metric(:,6),metric(:,7), ...
    'VariableNames',{'Market','Trial','Model','NLL','RMSE', ...
    'WeightedSquaredError','MAPE','Hit1','Hit2','Hit3'});
validate_results(results,ordered_markets,ordered_trials,model_names);

prepublish_validator = @() validate_prepublish_inputs( ...
    data_dir,fit_dir,config);
write_legacy_evaluation_bundle(results,output_dir,markets,trials, ...
    config.markets,prepublish_validator);
end


function values = evaluate_one(model,parameters,test_full,test_short,config)
switch model
    case "CMNL"
        [~,~,hit,rmse,weighted,mape] = Click_MNL_function( ...
            parameters.cmnl,100,config.feature_start,[],test_full,[],true);
        nll = NaN;
    case "MNL"
        [~,nll,hit,rmse,weighted,mape] = Seq_MNL_function( ...
            parameters.mnl,1000,config.feature_start,[],test_full,[],true);
    case "SeqINV"
        [~,nll,hit,rmse,weighted,mape] = Seq_INV_function( ...
            parameters.seqinv,10,config.feature_start,[],test_short,[],true);
    case "NSMNL"
        [~,~,nll,hit,rmse,weighted,mape] = NS_MNL_function( ...
            parameters.nsmnl,10,config.feature_start,[],test_short,[],true, ...
            parameters.nsmnl_lambda);
    case "SCM"
        [~,~,hit,rmse,weighted,mape] = SCM_MNL_function( ...
            parameters.scm,config.feature_start,[],test_short,[],true);
        nll = NaN;
end
values = [nll,rmse,weighted,mape,hit(:).'];
paper_and_auxiliary = values(2:end);
valid_nll = ((model == "CMNL" || model == "SCM") && isnan(nll)) ...
    || (~(model == "CMNL" || model == "SCM") ...
        && isfinite(nll) && nll >= 0);
valid_metrics = numel(values) == 7 ...
    && all(isfinite(paper_and_auxiliary)) ...
    && all(paper_and_auxiliary(1:3) >= 0) ...
    && all(paper_and_auxiliary(4:6) >= 0) ...
    && all(paper_and_auxiliary(4:6) <= 1) ...
    && all(diff(paper_and_auxiliary(4:6)) >= -1e-12);
if ~valid_nll || ~valid_metrics
    error('evaluate_expedia_archived_models:NonFiniteMetric', ...
        'Expedia evaluator returned invalid metrics for %s.',model);
end
end


function data = validate_test_matrix(data,config)
if ~isnumeric(data) || ~ismatrix(data) || isempty(data) || ~isreal(data) ...
        || any(~isfinite(data),'all') ...
        || size(data,2) < config.feature_end
    error('evaluate_expedia_archived_models:InvalidData', ...
        'data_test must be a finite nonempty matrix of sufficient width.');
end
if ~isempty(config.matrix_columns) && size(data,2) ~= config.matrix_columns
    error('evaluate_expedia_archived_models:InvalidData', ...
        'data_test does not match the configured matrix width.');
end
session_ids = data(:,1);
if any(session_ids ~= fix(session_ids)) || any(diff(session_ids) < 0)
    error('evaluate_expedia_archived_models:InvalidData', ...
        'data_test session IDs must be sorted finite integers.');
end
starts = [1; find(session_ids(2:end) ~= session_ids(1:end-1))+1];
ends = [starts(2:end)-1; size(data,1)];
if any(~ismember(data(:,5),[0,1])) ...
        || any(splitapply(@sum,data(:,5),findgroups(session_ids)) ~= 1) ...
        || any(data(ends,2) ~= 0) || any(data(ends,4) ~= 100) ...
        || any(data(ends,7) ~= 0)
    error('evaluate_expedia_archived_models:InvalidData', ...
        'data_test violates the session, choice, or trailing-outside contract.');
end
end


function validate_results(results,markets,trials,models)
expected_rows = numel(markets)*numel(trials)*numel(models);
keys = strcat(results.Market,"|",string(results.Trial),"|",results.Model);
expected = strings(expected_rows,1);
row = 0;
for market = markets
    for trial = trials
        for model = models
            row = row + 1;
            expected(row) = strcat(market,"|",string(trial),"|",model);
        end
    end
end
metrics = results{:,{'RMSE','WeightedSquaredError','MAPE', ...
    'Hit1','Hit2','Hit3'}};
undefined_nll = ismember(results.Model,["CMNL","SCM"]);
if height(results) ~= expected_rows ...
        || numel(unique(keys)) ~= expected_rows ...
        || ~isequal(keys,expected) ...
        || any(~isfinite(metrics),'all') ...
        || any(~isnan(results.NLL(undefined_nll))) ...
        || any(~isfinite(results.NLL(~undefined_nll)))
    error('evaluate_expedia_archived_models:IncompleteOutput', ...
        'Expedia result table failed its complete-grid contract.');
end
end


function validate_prepublish_inputs(data_dir,fit_dir,config)
[~,~,expected_filenames] = expected_grid(config);
require_exact_file_grid(data_dir,expected_filenames, ...
    '^10k_sample_\d+_\d+\.mat$',false, ...
    'evaluate_expedia_archived_models:ArchiveMismatch');
expected_fits = replace(expected_filenames,'10k_sample_','fit_');
require_exact_file_grid(fit_dir,expected_fits, ...
    '^fit_\d+_\d+\.mat$',false, ...
    'evaluate_expedia_archived_models:LegacyFitMismatch');
end


function [markets,trials,filenames] = expected_grid(config)
markets = strings(50,1);
trials = zeros(50,1);
filenames = strings(50,1);
row = 0;
for market = config.markets
    for trial = 1:10
        row = row + 1;
        markets(row) = market;
        trials(row) = trial;
        filenames(row) = sprintf(config.data_pattern,char(market),trial);
    end
end
end


function require_exact_file_grid( ...
        folder,expected_names,pattern,allow_unmatched,error_id)
entries = dir(fullfile(folder,'*.mat'));
names = string({entries.name}).';
matches = ~cellfun(@isempty,regexp(cellstr(names),pattern,'once'));
observed = sort(names(matches));
valid_unmatched = allow_unmatched || all(matches);
if ~valid_unmatched || numel(observed) ~= 50 ...
        || ~isequal(observed,sort(expected_names))
    error(error_id,'Archived directory does not contain the exact 50-file grid.');
end
end


function require_file(path,error_id)
if ~isfile(path)
    error(error_id,'Missing required file: %s.',path);
end
end


function markets = validate_markets(markets,config)
markets = string(markets);
if ~isvector(markets) || isempty(markets) || any(ismissing(markets)) ...
        || any(strlength(markets) == 0)
    error('evaluate_expedia_archived_models:InvalidMarkets', ...
        'Markets must be a nonmissing vector.');
end
markets = markets(:).';
try
    canonical_market_indices(markets,config);
catch exception
    if strcmp(exception.identifier,'canonical_market_indices:DuplicateMarket')
        error('evaluate_expedia_archived_models:DuplicateMarkets','%s', ...
            exception.message);
    end
    error('evaluate_expedia_archived_models:InvalidMarkets','%s', ...
        exception.message);
end
end


function trials = validate_trials(trials)
if ~isnumeric(trials) || ~isvector(trials) || isempty(trials) ...
        || ~isreal(trials) || any(~isfinite(trials)) ...
        || any(trials ~= fix(trials)) || any(trials < 1 | trials > 10) ...
        || numel(unique(trials)) ~= numel(trials)
    error('evaluate_expedia_archived_models:InvalidTrials', ...
        'Trials must be unique integers from 1 through 10.');
end
trials = double(trials(:).');
end
