function wrote = write_trivago_evaluation_bundle( ...
        results,means,output_dir,requested_markets,requested_trials, ...
        canonical_markets,model_names,metadata,prepublish_validator)
%WRITE_TRIVAGO_EVALUATION_BUNDLE Publish canonical Trivago evaluation CSVs.
%   Only the exact ordered 5-by-10-by-6 grid is publishable.  The trials and
%   means tables are serialized before replacement and prior files are
%   restored if either replacement or post-publication verification fails.

validate_contract(results,means,output_dir,requested_markets, ...
    requested_trials,canonical_markets,model_names,metadata, ...
    prepublish_validator);
if ~is_complete_axis(requested_markets,canonical_markets) ...
        || ~is_complete_axis(requested_trials,1:10)
    warning('evaluate_all_models:SubsetNotWritten', ...
        ['Subset or reordered results were returned in memory and did not ' ...
         'replace the canonical Trivago evaluation outputs.']);
    wrote = false;
    return
end
validate_full_bundle(results,means,canonical_markets,model_names,metadata);

output_dir = char(string(output_dir));
if ~isfolder(output_dir)
    [ok,message] = mkdir(output_dir);
    if ~ok
        error('write_trivago_evaluation_bundle:CreateFailed','%s',message);
    end
end
if is_symbolic_link(output_dir)
    error('write_trivago_evaluation_bundle:NonRegularTarget', ...
        'The Trivago evaluation output directory cannot be a symbolic link.');
end
lock_cleanup = acquire_exclusive_lock(output_dir); %#ok<NASGU>
output_files = {fullfile(output_dir,'predictive_trivago_trials.csv'), ...
    fullfile(output_dir,'predictive_trivago_means.csv')};
wrote = write_publication_tables({results,means},output_files, ...
    {requested_markets,requested_trials,model_names}, ...
    {canonical_markets,1:10,model_names},[300,30], ...
    'evaluate_all_models:SubsetNotWritten', ...
    'A subset cannot replace the canonical Trivago evaluation outputs.', ...
    prepublish_validator);
end


function validate_full_bundle(results,means,canonical_markets,models,metadata)
result_columns = {'Market','Trial','Model','NLL','RMSE', ...
    'WeightedSquaredError','MAPE','Hit1','Hit2','Hit3'};
mean_columns = {'Market','Model','NLL','RMSE','WeightedSquaredError', ...
    'MAPE','Hit1','Hit2','Hit3'};
canonical_markets = string(canonical_markets(:));
models = string(models(:));
n_market = numel(canonical_markets);
n_model = numel(models);
expected_market = strings(n_market*10*n_model,1);
expected_trial = zeros(n_market*10*n_model,1);
expected_model = strings(n_market*10*n_model,1);
row = 0;
for trial = 1:10
    for market = canonical_markets.'
        for model = models.'
            row = row+1;
            expected_market(row) = market;
            expected_trial(row) = trial;
            expected_model(row) = model;
        end
    end
end
expected_mean_market = repelem(canonical_markets,n_model);
expected_mean_model = repmat(models,n_market,1);
valid_grid = height(results) == 300 && height(means) == 30 ...
    && isequal(results.Properties.VariableNames,result_columns) ...
    && isequal(means.Properties.VariableNames,mean_columns) ...
    && isequal(string(results.Market),expected_market) ...
    && isnumeric(results.Trial) && isreal(results.Trial) ...
    && isequal(double(results.Trial),expected_trial) ...
    && isequal(string(results.Model),expected_model) ...
    && isequal(string(means.Market),expected_mean_market) ...
    && isequal(string(means.Model),expected_mean_model);
metric = results{:,4:10};
undefined_nll = ismember(string(results.Model),["CMNL","SCM"]);
valid_metric = isnumeric(metric) && isreal(metric) ...
    && all(isnan(metric(undefined_nll,1))) ...
    && all(isfinite(metric(~undefined_nll,1))) ...
    && all(metric(~undefined_nll,1) >= 0) ...
    && all(isfinite(metric(:,2:end)),'all') ...
    && all(metric(:,2:4) >= 0,'all') ...
    && all(metric(:,5:7) >= 0 & metric(:,5:7) <= 1,'all') ...
    && all(diff(metric(:,5:7),1,2) >= -1e-12,'all');
valid_metadata = validate_metadata(metadata,canonical_markets,models);
if ~valid_grid || ~valid_metric || ~valid_metadata
    error('write_trivago_evaluation_bundle:IncompleteGrid', ...
        'The canonical Trivago evaluation grid or metadata is invalid.');
end

mean_metric = means{:,3:9};
valid_means = isnumeric(mean_metric) && isreal(mean_metric);
for market_index = 1:n_market
    for model_index = 1:n_model
        use = string(results.Market) == canonical_markets(market_index) ...
            & string(results.Model) == models(model_index);
        mean_row = (market_index-1)*n_model + model_index;
        expected = [mean_nll(results.NLL(use),models(model_index)), ...
            mean(results{use,5:10},1)];
        observed = mean_metric(mean_row,:);
        finite = isfinite(expected);
        tolerance = 1e-12*(1+abs(expected(finite)));
        valid_means = valid_means && sum(use) == 10 ...
            && all(isnan(observed(~finite))) ...
            && all(abs(observed(finite)-expected(finite)) <= tolerance);
    end
end
if ~valid_means
    error('write_trivago_evaluation_bundle:InvalidMeans', ...
        'Trivago means do not equal the canonical trial aggregation.');
end
end


function valid = validate_metadata(metadata,canonical_markets,models)
required = {'processed_manifest_file','rng','master_seed','seed_formula', ...
    'loop_order','scm_parameter_order','tasks'};
valid = isstruct(metadata) && isscalar(metadata) ...
    && all(isfield(metadata,required)) ...
    && strcmp(metadata.processed_manifest_file, ...
        fullfile('data','processed','trivago','manifest.json')) ...
    && strcmp(metadata.rng,'twister') && metadata.master_seed == 20260808 ...
    && strcmp(metadata.seed_formula, ...
        '20260808 + 10000*trial + 100*canonical_market_index + model_index') ...
    && strcmp(metadata.loop_order,'trial_market_model') ...
    && strcmp(metadata.scm_parameter_order,'beta_alpha') ...
    && isstruct(metadata.tasks) && numel(metadata.tasks) == 50;
if ~valid, return; end
task_fields = {'market','trial','data_file','fit_file','model_seeds'};
task_index = 0;
for trial = 1:10
    for market_index = 1:numel(canonical_markets)
        task_index = task_index+1;
        task = metadata.tasks(task_index);
        market = canonical_markets(market_index);
        expected_data = fullfile('data','processed','trivago', ...
            sprintf('10k_sample_%s_%d_p2000.mat',char(market),trial));
        expected_fit = fullfile('opt_data','trivago', ...
            sprintf('fit_%s_%d.mat',char(market),trial));
        expected_seeds = 20260808 + 10000*trial + 100*market_index ...
            + (1:numel(models));
        if ~all(isfield(task,task_fields)) ...
                || ~strcmp(task.market,market) || task.trial ~= trial ...
                || ~strcmp(task.data_file,expected_data) ...
                || ~strcmp(task.fit_file,expected_fit) ...
                || ~isnumeric(task.model_seeds) ...
                || ~isequal(double(task.model_seeds(:).'),expected_seeds)
            valid = false;
            return
        end
    end
end
end


function value = mean_nll(values,model)
if ismember(model,["CMNL","SCM"])
    value = NaN;
else
    value = mean(values);
end
end


function validate_contract(results,means,output_dir,markets,trials, ...
        canonical_markets,models,metadata,prepublish_validator)
valid = istable(results) && istable(means) && is_text_scalar(output_dir) ...
    && isvector(markets) && isvector(trials) && isvector(canonical_markets) ...
    && isvector(models) && isstruct(metadata) && isscalar(metadata) ...
    && isa(prepublish_validator,'function_handle');
if ~valid
    error('write_trivago_evaluation_bundle:InvalidContract', ...
        'Trivago evaluation publication arguments are malformed.');
end
end


function complete = is_complete_axis(requested,canonical)
if ischar(requested), requested = string(requested); end
if ischar(canonical), canonical = string(canonical); end
requested = requested(:);
canonical = canonical(:);
complete = numel(requested) == numel(canonical) ...
    && numel(unique(requested,'stable')) == numel(requested) ...
    && numel(unique(canonical,'stable')) == numel(canonical) ...
    && isequal(requested,canonical);
end


function cleanup = acquire_exclusive_lock(output_dir)
lock_file = fullfile(output_dir,'.predictive_trivago_evaluation.lock');
lock = javaObject('java.io.File',lock_file);
try
    acquired = lock.createNewFile();
catch exception
    error('write_trivago_evaluation_bundle:LockFailed', ...
        'Cannot create the exclusive Trivago evaluation lock: %s', ...
        exception.message);
end
if ~acquired
    error('write_trivago_evaluation_bundle:Locked', ...
        'Another Trivago evaluation publication holds the exclusive lock.');
end
identifier = javaMethod('randomUUID','java.util.UUID');
token = char(identifier.toString());
try
    write_lock_token(lock_file,token);
    if ~strcmp(read_lock_token(lock_file),token)
        error('write_trivago_evaluation_bundle:LockFailed', ...
            'The exclusive Trivago evaluation lock failed verification.');
    end
catch exception
    if isfile(lock_file), delete(lock_file); end
    error('write_trivago_evaluation_bundle:LockFailed', ...
        'Cannot initialize the Trivago evaluation lock: %s',exception.message);
end
cleanup = onCleanup(@() release_lock(lock_file,token));
end


function write_lock_token(lock_file,token)
file = fopen(lock_file,'wb');
if file < 0
    error('write_trivago_evaluation_bundle:LockFailed', ...
        'Cannot open the exclusive Trivago evaluation lock.');
end
cleanup = onCleanup(@() fclose(file));
bytes = uint8(token);
if fwrite(file,bytes,'uint8') ~= numel(bytes)
    error('write_trivago_evaluation_bundle:LockFailed', ...
        'Cannot completely initialize the Trivago evaluation lock.');
end
clear cleanup
end


function release_lock(lock_file,token)
if strcmp(read_lock_token(lock_file),token), delete(lock_file); end
end


function observed = read_lock_token(lock_file)
observed = '';
if ~isfile(lock_file), return; end
file = fopen(lock_file,'rb');
if file < 0, return; end
cleanup = onCleanup(@() fclose(file));
observed = char(fread(file,Inf,'*uint8').');
clear cleanup
end


function symbolic = is_symbolic_link(path)
entry = javaObject('java.io.File',char(string(path)));
symbolic = javaMethod('isSymbolicLink','java.nio.file.Files',entry.toPath());
end


function valid = is_text_scalar(value)
valid = (ischar(value) && isrow(value)) ...
    || (isstring(value) && isscalar(value) && ~ismissing(value));
end
