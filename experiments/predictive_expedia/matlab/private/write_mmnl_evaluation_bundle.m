function wrote = write_mmnl_evaluation_bundle( ...
        results,means,output_dir,requested_markets,requested_trials, ...
        canonical_markets,prepublish_validator)
%WRITE_MMNL_EVALUATION_BUNDLE Publish canonical Expedia MMNL CSVs.
%   Only the exact ordered 5-market by 10-trial grid can replace the trials
%   and means files.  Publication remains locked, verified, and reversible.

validate_contract(results,means,output_dir,requested_markets, ...
    requested_trials,canonical_markets,prepublish_validator);
if ~is_complete_axis(requested_markets,canonical_markets) ...
        || ~is_complete_axis(requested_trials,1:10)
    warning('evaluate_mmnl_only:SubsetNotWritten', ...
        ['Subset or reordered results were returned in memory and did not ' ...
         'replace the canonical MMNL publication outputs.']);
    wrote = false;
    return
end
validate_full_bundle(results,means,canonical_markets);

output_dir = char(string(output_dir));
if ~isfolder(output_dir)
    [ok,message] = mkdir(output_dir);
    if ~ok
        error('write_mmnl_evaluation_bundle:CreateFailed','%s',message);
    end
end
if is_symbolic_link(output_dir)
    error('write_mmnl_evaluation_bundle:NonRegularTarget', ...
        'The MMNL evaluation output directory cannot be a symbolic link.');
end
lock_cleanup = acquire_exclusive_lock(output_dir); %#ok<NASGU>
output_files = {fullfile(output_dir,'predictive_expedia_mmnl_trials.csv'), ...
    fullfile(output_dir,'predictive_expedia_mmnl_means.csv')};
wrote = write_publication_tables({results,means},output_files, ...
    {requested_markets,requested_trials}, ...
    {canonical_markets,1:10},[50,5], ...
    'evaluate_mmnl_only:SubsetNotWritten', ...
    'A subset cannot replace the canonical Expedia MMNL outputs.', ...
    prepublish_validator);
end


function validate_full_bundle(results,means,canonical_markets)
result_columns = {'Market','Trial','Model','NLL','RMSE', ...
    'WeightedSquaredError','MAPE','Hit1','Hit2','Hit3'};
mean_columns = {'Market','Model','NLL','RMSE','WeightedSquaredError', ...
    'MAPE','Hit1','Hit2','Hit3','NTasks'};
canonical_markets = string(canonical_markets(:));
expected_market = repmat(canonical_markets,10,1);
expected_trial = kron((1:10).',ones(numel(canonical_markets),1));
metric = results{:,4:10};
valid_grid = height(results) == 50 && height(means) == 5 ...
    && isequal(results.Properties.VariableNames,result_columns) ...
    && isequal(means.Properties.VariableNames,mean_columns) ...
    && isequal(string(results.Market),expected_market) ...
    && isnumeric(results.Trial) && isreal(results.Trial) ...
    && isequal(double(results.Trial),expected_trial) ...
    && all(string(results.Model) == "MMNL") ...
    && isequal(string(means.Market),canonical_markets) ...
    && all(string(means.Model) == "MMNL");
valid_metric = isnumeric(metric) && isreal(metric) ...
    && all(isfinite(metric),'all') && all(metric(:,1:4) >= 0,'all') ...
    && all(metric(:,5:7) >= 0 & metric(:,5:7) <= 1,'all') ...
    && all(diff(metric(:,5:7),1,2) >= -1e-12,'all');
if ~valid_grid || ~valid_metric
    error('write_mmnl_evaluation_bundle:IncompleteGrid', ...
        'The canonical MMNL trial schema or task grid is invalid.');
end

mean_metric = means{:,3:9};
valid_means = isnumeric(mean_metric) && isreal(mean_metric) ...
    && all(isfinite(mean_metric),'all') ...
    && isnumeric(means.NTasks) && all(means.NTasks == 10);
for market_index = 1:numel(canonical_markets)
    use = string(results.Market) == canonical_markets(market_index);
    expected_metric = mean(results{use,4:10},1);
    tolerance = 1e-12*(1+abs(expected_metric));
    valid_means = valid_means && sum(use) == 10 ...
        && all(abs(mean_metric(market_index,:)-expected_metric) <= tolerance);
end
if ~valid_means
    error('write_mmnl_evaluation_bundle:InvalidMeans', ...
        'MMNL means do not equal the canonical trial aggregation.');
end
end


function validate_contract(results,means,output_dir,markets,trials, ...
        canonical_markets,prepublish_validator)
valid = istable(results) && istable(means) ...
    && is_text_scalar(output_dir) && isvector(markets) && isvector(trials) ...
    && isvector(canonical_markets) ...
    && isa(prepublish_validator,'function_handle');
if ~valid
    error('write_mmnl_evaluation_bundle:InvalidContract', ...
        'MMNL evaluation publication arguments are malformed.');
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
lock_file = fullfile(output_dir,'.predictive_expedia_mmnl_evaluation.lock');
lock = javaObject('java.io.File',lock_file);
try
    acquired = lock.createNewFile();
catch exception
    error('write_mmnl_evaluation_bundle:LockFailed', ...
        'Cannot create the exclusive MMNL evaluation lock: %s', ...
        exception.message);
end
if ~acquired
    error('write_mmnl_evaluation_bundle:Locked', ...
        'Another MMNL evaluation publication holds the exclusive lock.');
end
identifier = javaMethod('randomUUID','java.util.UUID');
token = char(identifier.toString());
try
    write_lock_token(lock_file,token);
    if ~strcmp(read_lock_token(lock_file),token)
        error('write_mmnl_evaluation_bundle:LockFailed', ...
            'The exclusive MMNL evaluation lock failed verification.');
    end
catch exception
    if isfile(lock_file), delete(lock_file); end
    error('write_mmnl_evaluation_bundle:LockFailed', ...
        'Cannot initialize the exclusive MMNL evaluation lock: %s', ...
        exception.message);
end
cleanup = onCleanup(@() release_lock(lock_file,token));
end


function write_lock_token(lock_file,token)
file = fopen(lock_file,'wb');
if file < 0
    error('write_mmnl_evaluation_bundle:LockFailed', ...
        'Cannot open the exclusive MMNL evaluation lock.');
end
cleanup = onCleanup(@() fclose(file));
bytes = uint8(token);
if fwrite(file,bytes,'uint8') ~= numel(bytes)
    error('write_mmnl_evaluation_bundle:LockFailed', ...
        'Cannot completely initialize the exclusive MMNL evaluation lock.');
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
