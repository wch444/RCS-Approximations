function wrote = write_legacy_evaluation_bundle( ...
        results,output_dir,requested_markets,requested_trials, ...
        canonical_markets,prepublish_validator)
%WRITE_LEGACY_EVALUATION_BUNDLE Publish the five-model Expedia CSV block.
%   Only a complete 5-market by 10-trial by 5-model table can replace the
%   publication file.  Serialization, backup, replacement, verification,
%   and rollback are handled as one locked publication operation.

validate_contract(results,output_dir,requested_markets,requested_trials, ...
    canonical_markets,prepublish_validator);
full_grid = is_complete_axis(requested_markets,canonical_markets) ...
    && is_complete_axis(requested_trials,1:10);
output_file = fullfile(char(string(output_dir)), ...
    'predictive_expedia_legacy_trials.csv');
if ~full_grid
    wrote = write_publication_tables({results},{output_file}, ...
        {requested_markets,requested_trials},{canonical_markets,1:10},250, ...
        'evaluate_expedia_archived_models:SubsetNotWritten', ...
        ['Subset results were returned in memory and did not replace the ' ...
         'complete Expedia five-model evaluation output.'], ...
        prepublish_validator);
    return
end

validate_full_bundle(results,canonical_markets);
output_dir = char(string(output_dir));
if ~isfolder(output_dir)
    [ok,message] = mkdir(output_dir);
    if ~ok
        error('write_legacy_evaluation_bundle:CreateFailed','%s',message);
    end
end
if is_symbolic_link(output_dir)
    error('write_legacy_evaluation_bundle:NonRegularTarget', ...
        'The legacy evaluation output directory cannot be a symbolic link.');
end
lock_cleanup = acquire_exclusive_lock(output_dir); %#ok<NASGU>
wrote = write_publication_tables({results},{output_file}, ...
    {requested_markets,requested_trials},{canonical_markets,1:10},250, ...
    'evaluate_expedia_archived_models:SubsetNotWritten', ...
    'A subset cannot replace the complete Expedia five-model output.', ...
    prepublish_validator);
end


function validate_full_bundle(results,canonical_markets)
columns = {'Market','Trial','Model','NLL','RMSE', ...
    'WeightedSquaredError','MAPE','Hit1','Hit2','Hit3'};
models = ["CMNL","MNL","SeqINV","NSMNL","SCM"];
canonical_markets = string(canonical_markets(:));
if height(results) ~= 250 ...
        || ~isequal(results.Properties.VariableNames,columns) ...
        || ~isequal(canonical_markets,["5";"14";"15";"24";"32"])
    error('write_legacy_evaluation_bundle:IncompleteGrid', ...
        'The five-model trial table schema is invalid.');
end
expected_market = repelem(canonical_markets,10*numel(models));
expected_trial = repmat(repelem((1:10).',numel(models)), ...
    numel(canonical_markets),1);
expected_model = repmat(models.',10*numel(canonical_markets),1);
metric = results{:,{'RMSE','WeightedSquaredError','MAPE', ...
    'Hit1','Hit2','Hit3'}};
undefined_nll = ismember(string(results.Model),["CMNL","SCM"]);
nll = results.NLL;
valid_grid = isequal(string(results.Market),expected_market) ...
    && isnumeric(results.Trial) && isreal(results.Trial) ...
    && isequal(double(results.Trial),expected_trial) ...
    && isequal(string(results.Model),expected_model);
valid_metrics = isnumeric(metric) && isreal(metric) ...
    && all(isfinite(metric),'all') && all(metric(:,1:3) >= 0,'all') ...
    && all(metric(:,4:6) >= 0 & metric(:,4:6) <= 1,'all') ...
    && all(diff(metric(:,4:6),1,2) >= -1e-12,'all') ...
    && isnumeric(nll) && isreal(nll) ...
    && all(isnan(nll(undefined_nll))) ...
    && all(isfinite(nll(~undefined_nll))) ...
    && all(nll(~undefined_nll) >= 0);
if ~valid_grid || ~valid_metrics
    error('write_legacy_evaluation_bundle:IncompleteGrid', ...
        'The five-model trial task grid or metrics are invalid.');
end
end


function validate_contract(results,output_dir,markets,trials, ...
        canonical_markets,prepublish_validator)
valid = istable(results) && is_text_scalar(output_dir) ...
    && isvector(markets) && isvector(trials) ...
    && isvector(canonical_markets) ...
    && isa(prepublish_validator,'function_handle');
if ~valid
    error('write_legacy_evaluation_bundle:InvalidContract', ...
        'Legacy evaluation publication arguments are malformed.');
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
    && all(ismember(canonical,requested));
end


function cleanup = acquire_exclusive_lock(output_dir)
lock_file = fullfile(output_dir, ...
    '.predictive_expedia_legacy_evaluation.lock');
lock = javaObject('java.io.File',lock_file);
try
    acquired = lock.createNewFile();
catch exception
    error('write_legacy_evaluation_bundle:LockFailed', ...
        'Cannot create the exclusive legacy evaluation lock: %s', ...
        exception.message);
end
if ~acquired
    error('write_legacy_evaluation_bundle:Locked', ...
        'Another legacy evaluation publication holds the exclusive lock.');
end
identifier = javaMethod('randomUUID','java.util.UUID');
token = char(identifier.toString());
try
    write_lock_token(lock_file,token);
    if ~strcmp(read_lock_token(lock_file),token)
        error('write_legacy_evaluation_bundle:LockFailed', ...
            'The exclusive legacy evaluation lock failed verification.');
    end
catch exception
    if isfile(lock_file), delete(lock_file); end
    error('write_legacy_evaluation_bundle:LockFailed', ...
        'Cannot initialize the exclusive legacy evaluation lock: %s', ...
        exception.message);
end
cleanup = onCleanup(@() release_lock(lock_file,token));
end


function write_lock_token(lock_file,token)
file = fopen(lock_file,'wb');
if file < 0
    error('write_legacy_evaluation_bundle:LockFailed', ...
        'Cannot open the exclusive legacy evaluation lock.');
end
cleanup = onCleanup(@() fclose(file));
bytes = uint8(token);
if fwrite(file,bytes,'uint8') ~= numel(bytes)
    error('write_legacy_evaluation_bundle:LockFailed', ...
        'Cannot completely initialize the exclusive legacy evaluation lock.');
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
