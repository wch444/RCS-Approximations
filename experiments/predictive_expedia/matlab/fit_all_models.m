function output_files = fit_all_models(dataset_name, markets, trials, varargin)
%FIT_ALL_MODELS Fit all six Expedia models into an isolated staging folder.
%   FIT_ALL_MODELS('expedia',MARKETS,TRIALS,'DataDir',DATA_DIR,
%   'FitDir',FIT_DIR) fits SCM, MNL, MMNL, CMNL, SeqINV, and NSMNL for the
%   requested market/trial cross product.  DATA_DIR and FIT_DIR are required
%   explicitly; FIT_DIR must be outside the canonical opt_data/expedia tree.
%
%   Name-value options ShardIndex and ShardCount select a deterministic
%   subset of the canonical 50 tasks.  Resume=true validates and skips an
%   already completed task instead of overwriting it.  Each new MAT is saved
%   atomically only after all six estimates satisfy the unified-fit contract.

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
markets = validate_markets(markets,config);
trials = validate_trials(trials);
options = parse_options(varargin,root);
tasks = select_tasks( ...
    markets,trials,config,options.ShardIndex,options.ShardCount);

canonical_fit_dir = fullfile(root,'opt_data','expedia');
canonical_data_dir = fullfile(root,'data','processed','expedia');
protected_paths = {canonical_fit_dir,canonical_data_dir};
for protected_index = 1:numel(protected_paths)
    protected = protected_paths{protected_index};
    if same_path(options.FitDir,protected) ...
            || path_is_within(options.FitDir,protected) ...
            || same_path(options.DataDir,protected) ...
            || path_is_within(options.DataDir,protected)
        error('fit_all_models:UnsafeStagingPath', ...
            ['Fresh Expedia data and fits must remain outside the canonical ' ...
             'data/processed/expedia and opt_data/expedia trees.']);
    end
end
if same_path(options.DataDir,options.FitDir) ...
        || path_is_within(options.DataDir,options.FitDir) ...
        || path_is_within(options.FitDir,options.DataDir)
    error('fit_all_models:UnsafePathLayout', ...
        'DataDir and FitDir must be separate, non-nested directories.');
end
if is_symbolic_link(options.DataDir) || is_symbolic_link(options.FitDir)
    error('fit_all_models:SymbolicLinkDirectory', ...
        'DataDir and FitDir themselves may not be symbolic links.');
end
if ~isfolder(options.DataDir)
    error('fit_all_models:MissingDataDirectory', ...
        'The explicit Expedia data directory does not exist: %s', ...
        options.DataDir);
end
if ~isfolder(options.FitDir)
    [ok,message] = mkdir(options.FitDir);
    if ~ok
        error('fit_all_models:CreateFitDirectoryFailed','%s',message);
    end
end

preflight_inputs(tasks,options.DataDir,config);
preflight_outputs(tasks,options.FitDir,options.Resume);

optimizer_options = optimoptions('fminunc', ...
    'Display','off', ...
    'MaxIterations',100, ...
    'MaxFunctionEvaluations',5e4, ...
    'UseParallel',false, ...
    'OptimalityTolerance',1e-4);
em_options = struct('MaxIterations',100,'RelativeTolerance',1e-7);
n_component = 3;
output_files = strings(numel(tasks),1);

for task_index = 1:numel(tasks)
    market = tasks(task_index).market;
    trial = tasks(task_index).trial;
    data_name = sprintf(config.data_pattern,char(market),trial);
    data_file = fullfile(options.DataDir,data_name);
    output_name = sprintf('fit_%s_%d.mat',char(market),trial);
    output_file = fullfile(options.FitDir,output_name);
    output_files(task_index) = string(output_file);

    payload = load(data_file,'data_train','data_test');
    if ~all(isfield(payload,{'data_train','data_test'}))
        error('fit_all_models:InvalidData', ...
            'MAT must contain data_train and data_test: %s',data_file);
    end
    train_full = validate_matrix( ...
        payload.data_train,config,'data_train',6000,market);
    test_full = validate_matrix( ...
        payload.data_test,config,'data_test',4000,market);
    validate_matrix_pair(train_full,test_full);

    if isfile(output_file)
        validate_completed_fit( ...
            output_file,data_file,market,trial,train_full,test_full);
        fprintf('[expedia %s trial %d] already complete; skipped\n', ...
            market,trial);
        continue
    end
    claim_file = acquire_task_claim(output_file);
    claim_cleanup = onCleanup(@() release_task_claim(claim_file));
    if isfile(output_file) || isfolder(output_file) ...
            || is_symbolic_link(output_file)
        error('fit_all_models:ConcurrentOutput', ...
            'A staged fit appeared while claiming this task: %s',output_file);
    end

    train_short = train_full(:,1:config.feature_end);
    test_short = test_full(:,1:config.feature_end);
    n_short = size(train_short(:,config.feature_start:end),2);
    n_full = size(train_full(:,config.feature_start:end),2);
    init_short = ones(n_short,1)/10;
    init_full = ones(n_full,1)/10;
    task_seed = paper_mmnl_seed(config.name,market,trial);

    fprintf('\n[expedia %s trial %d] fitting SCM\n',market,trial);
    rng(task_seed+10,'twister');
    [scm,~,~,~,~,~,~] = SCM_MNL_function( ...
        init_short,config.feature_start,train_short,test_short, ...
        optimizer_options,false);

    fprintf('[expedia %s trial %d] fitting SeqINV\n',market,trial);
    rng(task_seed+20,'twister');
    [seqinv,~,~,~,~,~,~] = Seq_INV_function( ...
        init_short,10,config.feature_start,train_short,test_short, ...
        optimizer_options,false);

    fprintf('[expedia %s trial %d] fitting NSMNL\n',market,trial);
    rng(task_seed+30,'twister');
    [nsmnl,nsmnl_lambda,~,~,~,~,~,~] = NS_MNL_function( ...
        init_short,10,config.feature_start,train_short,test_short, ...
        optimizer_options,false);

    fprintf('[expedia %s trial %d] fitting MNL\n',market,trial);
    rng(task_seed+40,'twister');
    [mnl,~,~,~,~,~,~] = Seq_MNL_function( ...
        init_full,1000,config.feature_start,train_full,test_full, ...
        optimizer_options,false);

    fprintf('[expedia %s trial %d] fitting CMNL\n',market,trial);
    rng(task_seed+50,'twister');
    [cmnl,~,~,~,~,~,~] = Click_MNL_function( ...
        init_full,100,config.feature_start,train_full,test_full, ...
        optimizer_options,false);

    % Keep the identified Expedia MMNL initialization and estimator path
    % identical to fit_mmnl_only.  In particular, reset the paper seed before
    % drawing beta0_full so prior model fits cannot change the MMNL stream.
    fprintf('[expedia %s trial %d] fitting MMNL\n',market,trial);
    mmnl_seed = paper_mmnl_seed(config.name,market,trial);
    rng(mmnl_seed,'twister');
    mmnl_alpha0 = ones(1,n_component)/n_component;
    mmnl_beta0_full = rand(n_full,n_component);
    [train_reduced,test_reduced,mmnl_beta0_reduced,parameterization] = ...
        expedia_mmnl_identifiable_parameterization( ...
            train_full,test_full,mmnl_beta0_full,config.feature_start);
    [mmnl_alpha,mmnl_beta_reduced] = ...
        Mix_MNL_EM_function( ...
            mmnl_alpha0,mmnl_beta0_reduced,config.feature_start, ...
            train_reduced,test_reduced,optimizer_options,false,em_options);
    validate_mmnl_parameters( ...
        mmnl_alpha,mmnl_beta_reduced, ...
        parameterization.reduced_feature_dimension,n_component);
    mmnl_beta = zeros(n_full,n_component);
    mmnl_beta(parameterization.reduced_feature_indices,:) = ...
        mmnl_beta_reduced;
    validate_mmnl_parameters( ...
        mmnl_alpha,mmnl_beta,n_full,n_component);

    models = struct( ...
        'scm',scm, ...
        'mnl',mnl, ...
        'mmnl_alpha',mmnl_alpha, ...
        'mmnl_beta',mmnl_beta, ...
        'cmnl',cmnl, ...
        'seqinv',seqinv, ...
        'nsmnl',nsmnl, ...
        'nsmnl_lambda',nsmnl_lambda);
    models = orderfields(models,schema.model_fields);

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
    fit.matrix_columns = size(train_full,2);
    fit.page_size = 10;
    fit.train_rows = size(train_full,1);
    fit.test_rows = size(test_full,1);
    fit.train_sessions = numel(unique(train_full(:,1)));
    fit.test_sessions = numel(unique(test_full(:,1)));
    fit.mmnl_components = n_component;
    fit.mmnl_estimator_version = schema.mmnl_estimator_version;
    fit.scm_parameter_order = schema.scm_parameter_order;
    fit.data_file = fullfile( ...
        'data','processed','expedia',data_name);
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
    fit.mmnl_beta0_full = mmnl_beta0_full;
    fit.mmnl_beta0_reduced = mmnl_beta0_reduced;
    fit.mmnl_beta_reduced = mmnl_beta_reduced;
    fit.models = models;
    fit = orderfields(fit,schema.expedia_fields);
    validate_unified_fit( ...
        fit,'expedia',market,trial,train_full,test_full);
    atomic_save(output_file,fit);
    clear claim_cleanup
    fprintf('saved %s\n',output_file);
end
end


function require_expedia(dataset_name)
valid_text = (ischar(dataset_name) && isrow(dataset_name)) ...
    || (isstring(dataset_name) && isscalar(dataset_name) ...
        && ~ismissing(dataset_name));
if ~valid_text || ~strcmp(char(dataset_name),'expedia')
    error('fit_all_models:UnsupportedDataset', ...
        'This complete fitting workflow supports only Expedia.');
end
end


function options = parse_options(arguments,root)
parser = inputParser;
parser.FunctionName = 'fit_all_models';
addParameter(parser,'DataDir','',@is_text_scalar);
addParameter(parser,'FitDir','',@is_text_scalar);
addParameter(parser,'ShardIndex',1,@is_positive_integer_scalar);
addParameter(parser,'ShardCount',1,@is_positive_integer_scalar);
addParameter(parser,'Resume',false,@is_logical_scalar);
parse(parser,arguments{:});
options = parser.Results;
if strlength(string(options.DataDir)) == 0 ...
        || strlength(string(options.FitDir)) == 0
    error('fit_all_models:ExplicitPathsRequired', ...
        ['DataDir and FitDir are required.  Use separate staging paths ' ...
         'for corrected matrices and fresh fitted artifacts.']);
end
options.DataDir = resolve_path(root,options.DataDir);
options.FitDir = resolve_path(root,options.FitDir);
options.ShardIndex = double(options.ShardIndex);
options.ShardCount = double(options.ShardCount);
options.Resume = logical(options.Resume);
if options.ShardIndex > options.ShardCount
    error('fit_all_models:InvalidShard', ...
        'ShardIndex must not exceed ShardCount.');
end
end


function value = resolve_path(root,value)
value = char(string(value));
is_absolute = startsWith(value,filesep) ...
    || ~isempty(regexp(value,'^[A-Za-z]:[\\/]','once'));
if ~is_absolute
    value = fullfile(root,value);
end
end


function valid = is_text_scalar(value)
valid = (ischar(value) && isrow(value)) ...
    || (isstring(value) && isscalar(value) && ~ismissing(value));
end


function valid = is_positive_integer_scalar(value)
valid = isnumeric(value) && isreal(value) && isscalar(value) ...
    && isfinite(value) && value >= 1 && value == fix(value);
end


function valid = is_logical_scalar(value)
valid = islogical(value) && isscalar(value);
end


function markets = validate_markets(markets,config)
markets = string(markets);
if ~isvector(markets) || isempty(markets) || any(ismissing(markets)) ...
        || any(strlength(markets) == 0)
    error('fit_all_models:InvalidMarkets', ...
        'Markets must be a nonempty, nonmissing vector.');
end
markets = markets(:).';
try
    canonical_market_indices(markets,config);
catch exception
    error('fit_all_models:InvalidMarkets','%s',exception.message);
end
markets = config.markets(ismember(config.markets,markets));
end


function trials = validate_trials(trials)
if ~isnumeric(trials) || ~isvector(trials) || isempty(trials) ...
        || ~isreal(trials) || any(~isfinite(trials)) ...
        || any(trials ~= fix(trials)) || any(trials < 1 | trials > 10) ...
        || numel(unique(trials)) ~= numel(trials)
    error('fit_all_models:InvalidTrials', ...
        'Trials must be unique integers from 1 through 10.');
end
trials = sort(double(trials(:).'));
end


function tasks = select_tasks(markets,trials,config,shard_index,shard_count)
tasks = struct('market',{},'trial',{},'canonical_index',{});
for trial = trials
    for canonical_index = 1:numel(config.markets)
        market = config.markets(canonical_index);
        if ~ismember(market,markets)
            continue
        end
        global_index = (trial-1)*numel(config.markets)+canonical_index;
        if mod(global_index-1,shard_count) ~= shard_index-1
            continue
        end
        tasks(end+1) = struct( ...
            'market',market, ...
            'trial',trial, ...
            'canonical_index',global_index); %#ok<AGROW>
    end
end
if isempty(tasks)
    error('fit_all_models:EmptyShard', ...
        'The requested market/trial subset contains no task in this shard.');
end
end


function preflight_inputs(tasks,data_dir,config)
[task_records,manifest_time] = ...
    validate_preprocess_manifest(tasks,data_dir,config);
for index = 1:numel(tasks)
    data_name = sprintf( ...
        config.data_pattern,char(tasks(index).market),tasks(index).trial);
    data_file = fullfile(data_dir,data_name);
    if ~isfile(data_file)
        error('fit_all_models:MissingData', ...
            'Missing staged Expedia MAT: %s',data_file);
    end
    if is_symbolic_link(data_file)
        error('fit_all_models:SymbolicLinkInput', ...
            'Staged Expedia MAT files must be ordinary files: %s',data_file);
    end
    data_info = dir(data_file);
    if isempty(data_info) || data_info.datenum > manifest_time
        error('fit_all_models:StalePreprocessManifest', ...
            'A staged MAT is newer than its preprocess manifest: %s',data_file);
    end
    payload = load(data_file,'data_train','data_test');
    if ~all(isfield(payload,{'data_train','data_test'}))
        error('fit_all_models:InvalidData', ...
            'MAT must contain data_train and data_test: %s',data_file);
    end
    record = task_records{index};
    if size(payload.data_train,1) ~= record.train_rows ...
            || size(payload.data_test,1) ~= record.test_rows ...
            || size(payload.data_train,2) ~= record.columns ...
            || size(payload.data_test,2) ~= record.columns
        error('fit_all_models:InvalidPreprocessManifest', ...
            'Manifest dimensions do not match staged MAT: %s',data_file);
    end
    market = tasks(index).market;
    train_data = validate_matrix( ...
        payload.data_train,config,'data_train',6000,market);
    test_data = validate_matrix( ...
        payload.data_test,config,'data_test',4000,market);
    validate_matrix_pair(train_data,test_data);
end
end


function [task_records,manifest_time] = ...
        validate_preprocess_manifest(tasks,data_dir,config)
manifest_file = fullfile(data_dir,'expedia_preprocess_manifest.json');
if ~isfile(manifest_file) || is_symbolic_link(manifest_file)
    error('fit_all_models:MissingPreprocessManifest', ...
        'Corrected staged data require an ordinary preprocess manifest: %s', ...
        manifest_file);
end
manifest_info = dir(manifest_file);
manifest_time = manifest_info.datenum;
try
    manifest = jsondecode(fileread(manifest_file));
catch exception
    error('fit_all_models:InvalidPreprocessManifest', ...
        'Cannot read the corrected preprocess manifest: %s',exception.message);
end
required = {'query_sample_size','train_fraction','max_query_clicks', ...
    'split_seed','standardization','files'};
if ~isstruct(manifest) || ~isscalar(manifest) ...
        || ~all(isfield(manifest,required)) ...
        || manifest.query_sample_size ~= 10000 ...
        || abs(manifest.train_fraction-0.6) > eps ...
        || manifest.max_query_clicks ~= 4 ...
        || ~strcmp(manifest.split_seed,'trial') ...
        || ~strcmp(manifest.standardization, ...
            'separate train/test sample SD (ddof=1)') ...
        || ~isstruct(manifest.files)
    error('fit_all_models:InvalidPreprocessManifest', ...
        'The staged-data manifest does not match the corrected protocol.');
end
records = manifest.files;
task_records = cell(numel(tasks),1);
record_fields = {'site','trial','file','train_queries','test_queries', ...
    'train_rows','test_rows','columns'};
if ~all(isfield(records,record_fields))
    error('fit_all_models:InvalidPreprocessManifest', ...
        'The staged-data manifest file records are incomplete.');
end
for index = 1:numel(tasks)
    market = tasks(index).market;
    trial = tasks(index).trial;
    site = str2double(market);
    matches = [records.site] == site & [records.trial] == trial;
    expected_name = sprintf(config.data_pattern,char(market),trial);
    if sum(matches) ~= 1
        error('fit_all_models:InvalidPreprocessManifest', ...
            'Manifest must contain exactly one record for %s trial %d.', ...
            market,trial);
    end
    record = records(matches);
    if ~strcmp(record.file,expected_name) ...
            || record.train_queries ~= 6000 ...
            || record.test_queries ~= 4000 ...
            || record.train_rows <= record.train_queries ...
            || record.test_rows <= record.test_queries ...
            || record.columns < config.feature_end
        error('fit_all_models:InvalidPreprocessManifest', ...
            'Manifest record is invalid for %s trial %d.',market,trial);
    end
    task_records{index} = record;
end
end


function preflight_outputs(tasks,fit_dir,resume)
for index = 1:numel(tasks)
    output_name = sprintf( ...
        'fit_%s_%d.mat',char(tasks(index).market),tasks(index).trial);
    output_file = fullfile(fit_dir,output_name);
    if is_symbolic_link(output_file)
        error('fit_all_models:SymbolicLinkOutput', ...
            'Staged fit paths may not be symbolic links: %s',output_file);
    end
    if isfolder(output_file)
        error('fit_all_models:NonRegularOutput', ...
            'A staged fit target may not be a directory: %s',output_file);
    end
    claim_file = [output_file '.lock'];
    if isfile(claim_file) || isfolder(claim_file) ...
            || is_symbolic_link(claim_file)
        error('fit_all_models:TaskClaimed', ...
            ['A requested task already has a claim file.  Confirm no fitting ' ...
             'process owns it before removing it: %s'],claim_file);
    end
    if isfile(output_file) && ~resume
        error('fit_all_models:OutputExists', ...
            ['A requested staging output already exists.  Use Resume=true ' ...
             'to validate and skip completed tasks: %s'], ...
            output_file);
    end
end
end


function data = validate_matrix( ...
        data,config,label,expected_sessions,market)
expected_columns = 28;
if market == "24"
    expected_columns = 27;
end
if ~isnumeric(data) || ~isreal(data) || ~ismatrix(data) || isempty(data) ...
        || size(data,2) < config.feature_end ...
        || size(data,2) ~= expected_columns ...
        || any(~isfinite(data),'all')
    error('fit_all_models:InvalidData', ...
        '%s must be a finite nonempty real Expedia matrix.',label);
end
ids = data(:,1);
if any(ids ~= fix(ids)) || any(diff(ids) < 0) ...
        || numel(unique(ids)) ~= expected_sessions
    error('fit_all_models:InvalidData', ...
        '%s must contain exactly %d sorted integer query IDs.', ...
        label,expected_sessions);
end
groups = findgroups(ids);
starts = [1;find(ids(2:end) ~= ids(1:end-1))+1];
ends = [starts(2:end)-1;size(data,1)];
if any(~ismember(data(:,5),[0,1])) ...
        || any(splitapply(@sum,data(:,5),groups) ~= 1) ...
        || any(data(ends,2) ~= 0) ...
        || any(data(ends,4) ~= 100) ...
        || any(data(ends,6) ~= 0) ...
        || any(data(ends,7) ~= 0) ...
        || any(data(ends,8) ~= 0) ...
        || any(data(ends,3) ~= 0)
    error('fit_all_models:InvalidData', ...
        '%s violates the choice or trailing-outside-option contract.',label);
end
inside = true(size(data,1),1);
inside(ends) = false;
expected_site = str2double(market);
if any(data(inside,8) ~= 1) || any(data(inside,3) ~= expected_site)
    error('fit_all_models:InvalidData', ...
        '%s rows do not match Expedia site %s.',label,market);
end
if size(data,2) >= 21
    position = data(:,21:end);
    if any(~ismember(position,[0,1]),'all') ...
            || any(sum(position,2) ~= data(:,8))
        error('fit_all_models:InvalidData', ...
            '%s has invalid position indicators.',label);
    end
end
end


function validate_matrix_pair(train_data,test_data)
if size(train_data,2) ~= size(test_data,2)
    error('fit_all_models:InconsistentData', ...
        'Training and test matrices must have the same width.');
end
if ~isempty(intersect(unique(train_data(:,1)),unique(test_data(:,1))))
    error('fit_all_models:InconsistentData', ...
        'Training and test query IDs must be disjoint.');
end
end


function validate_mmnl_parameters(alpha,beta,n_feature,n_component)
valid = isnumeric(alpha) && isreal(alpha) ...
    && isequal(size(alpha),[1,n_component]) ...
    && all(isfinite(alpha),'all') && all(alpha >= 0) ...
    && abs(sum(alpha)-1) <= 1e-10 ...
    && isnumeric(beta) && isreal(beta) ...
    && isequal(size(beta),[n_feature,n_component]) ...
    && all(isfinite(beta),'all');
if ~valid
    error('fit_all_models:InvalidMMNLEstimate', ...
        'MMNL parameters are malformed, nonfinite, or outside the simplex.');
end
end


function validate_completed_fit( ...
        output_file,data_file,market,trial,train_data,test_data)
variables = whos('-file',output_file);
if numel(variables) ~= 1 || ~strcmp(variables.name,'fit') ...
        || ~strcmp(variables.class,'struct')
    error('fit_all_models:InvalidExistingFit', ...
        'Existing staging MAT must contain only the fit struct: %s', ...
        output_file);
end
data_info = dir(data_file);
fit_info = dir(output_file);
if isempty(data_info) || isempty(fit_info) ...
        || fit_info.datenum <= data_info.datenum
    error('fit_all_models:StaleExistingFit', ...
        ['Resume requires each staged fit to be newer than its current ' ...
         'staged input MAT: %s'],output_file);
end
payload = load(output_file,'fit');
if ~isfield(payload,'fit') || ~isstruct(payload.fit) ...
        || ~isscalar(payload.fit)
    error('fit_all_models:InvalidExistingFit', ...
        'Existing staging MAT must contain one scalar fit: %s',output_file);
end
try
    validate_unified_fit( ...
        payload.fit,'expedia',market,trial,train_data,test_data);
catch exception
    error('fit_all_models:InvalidExistingFit', ...
        'Existing staging fit failed validation (%s): %s', ...
        output_file,exception.message);
end
end


function atomic_save(output_file,fit)
folder = fileparts(output_file);
temporary_file = string(tempname(folder))+".mat";
cleanup = onCleanup(@() delete_if_present(temporary_file));
save(char(temporary_file),'fit','-v7');
if isfile(output_file) || isfolder(output_file) ...
        || is_symbolic_link(output_file)
    error('fit_all_models:ConcurrentOutput', ...
        'Refusing to overwrite a concurrently created fit: %s',output_file);
end
[ok,message] = movefile(char(temporary_file),output_file);
if ~ok
    error('fit_all_models:MoveFailed','%s',message);
end
clear cleanup
end


function claim_file = acquire_task_claim(output_file)
claim_file = [output_file '.lock'];
claim = java.io.File(claim_file);
if ~claim.createNewFile()
    error('fit_all_models:TaskClaimed', ...
        'Another fitting process already claimed this task: %s',claim_file);
end
end


function release_task_claim(claim_file)
if isfile(claim_file)
    delete(claim_file);
end
end


function delete_if_present(path)
if isfile(path)
    delete(path);
end
end


function equal = same_path(left,right)
equal = strcmp(canonical_path(left),canonical_path(right));
end


function within = path_is_within(child,parent)
child = canonical_path(child);
parent = canonical_path(parent);
if ispc
    child = lower(child);
    parent = lower(parent);
end
within = startsWith(child,[parent filesep]);
end


function path = canonical_path(path)
file = java.io.File(char(path));
path = char(file.getCanonicalPath());
end


function symbolic = is_symbolic_link(path)
empty_paths = javaArray('java.lang.String',0);
java_path = javaMethod('get','java.nio.file.Paths',char(path),empty_paths);
symbolic = javaMethod('isSymbolicLink','java.nio.file.Files',java_path);
end
