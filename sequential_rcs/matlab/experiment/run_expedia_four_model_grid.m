function [tasks,artifact_files] = run_expedia_four_model_grid(varargin)
%RUN_EXPEDIA_FOUR_MODEL_GRID Deterministic and guarded four-model runner.
%   The default is DryRun=true. Any fitting requires DryRun=false. Selecting
%   the complete 5x10x4 grid additionally requires
%   RequireFullRunConfirmation=true, which prevents an accidental 200-fit
%   launch. A single site/trial/model smoke is supported through
%   MaxTrainSessions and MaxTestSessions and is saved only under smoke_fits/.

here = fileparts(mfilename('fullpath'));
matlab_root = fileparts(here);
models_dir = fullfile(matlab_root,'models');
addpath(here,'-begin');
addpath(models_dir,'-begin');
config = expedia_experiment_config();

parser = inputParser;
parser.FunctionName = 'run_expedia_four_model_grid';
addParameter(parser,'DataDir','',@is_text_scalar);
addParameter(parser,'OutputDir','',@is_text_scalar);
addParameter(parser,'Sites',config.sites,@isnumeric);
addParameter(parser,'Trials',config.trials,@isnumeric);
addParameter(parser,'Models',config.modelIds,@is_text_collection);
addParameter(parser,'DryRun',true,@is_logical_scalar);
addParameter(parser,'RequireFullRunConfirmation',false,@is_logical_scalar);
addParameter(parser,'Resume',false,@is_logical_scalar);
addParameter(parser,'ShardIndex',1,@valid_positive_integer);
addParameter(parser,'ShardCount',1,@valid_positive_integer);
addParameter(parser,'MaxTrainSessions',[],@valid_optional_positive_integer);
addParameter(parser,'MaxTestSessions',[],@valid_optional_positive_integer);
addParameter(parser,'Verbose',true,@is_logical_scalar);
parse(parser,varargin{:});
options = parser.Results;
options.DryRun = logical(options.DryRun);
options.RequireFullRunConfirmation = logical(options.RequireFullRunConfirmation);
options.Resume = logical(options.Resume);
options.Verbose = logical(options.Verbose);
if options.ShardIndex > options.ShardCount
    error('ExpediaRunner:InvalidShard','ShardIndex must not exceed ShardCount.');
end

data_dir = char(string(options.DataDir));
if isempty(data_dir) || ~isfolder(data_dir)
    error('ExpediaRunner:MissingDataDir','DataDir must name the generated data directory.');
end
requested_tasks = build_expedia_task_grid( ...
    options.Sites,options.Trials,options.Models);
is_complete_grid = height(requested_tasks) == 200 ...
    && isempty(setxor(unique(requested_tasks.site)',config.sites)) ...
    && isempty(setxor(unique(requested_tasks.trial)',config.trials)) ...
    && isempty(setxor(unique(requested_tasks.model_id)',config.modelIds));
if ~options.DryRun && is_complete_grid && ~options.RequireFullRunConfirmation
    error('ExpediaRunner:FullRunNotConfirmed', ...
        ['The complete 200-fit grid is blocked. Reissue with ' ...
         'RequireFullRunConfirmation=true only after explicit approval.']);
end

validate_expedia_manifest(data_dir,unique(requested_tasks.site,'stable'), ...
    unique(requested_tasks.trial,'stable'));
shard_mask = mod((1:height(requested_tasks))'-1,options.ShardCount) ...
    == options.ShardIndex-1;
tasks = requested_tasks(shard_mask,:);
artifact_files = strings(height(tasks),1);

if options.DryRun
    if options.Verbose
        fprintf(['Dry run: %d requested tasks; shard %d/%d contains %d. ' ...
            'No model fitting or output writes performed.\n'], ...
            height(requested_tasks),options.ShardIndex,options.ShardCount,height(tasks));
        disp(tasks);
    end
    return
end

output_dir = char(string(options.OutputDir));
if isempty(output_dir)
    error('ExpediaRunner:MissingOutputDir','OutputDir is required when DryRun=false.');
end
if same_or_nested_path(data_dir,output_dir)
    error('ExpediaRunner:UnsafePathLayout', ...
        'DataDir and OutputDir must be separate, non-nested directories.');
end
if ~isfolder(output_dir)
    [ok,message] = mkdir(output_dir);
    if ~ok, error('ExpediaRunner:CreateOutputFailed','%s',message); end
end
is_smoke_requested = ~isempty(options.MaxTrainSessions) ...
    || ~isempty(options.MaxTestSessions);
if is_smoke_requested
    fits_dir = fullfile(output_dir,'smoke_fits');
else
    fits_dir = fullfile(output_dir,'fits');
end
if ~isfolder(fits_dir)
    [ok,message] = mkdir(fits_dir);
    if ~ok, error('ExpediaRunner:CreateOutputFailed','%s',message); end
end

pair_keys = unique(tasks(:,{'site','trial'}),'rows','stable');
for pair_index = 1:height(pair_keys)
    site = pair_keys.site(pair_index);
    trial = pair_keys.trial(pair_index);
    data_name = sprintf(config.dataPattern,site,trial);
    data_file = fullfile(data_dir,data_name);
    payload = load(data_file,'data_train','data_test');
    if ~all(isfield(payload,{'data_train','data_test'}))
        error('ExpediaRunner:InvalidMAT', ...
            'MAT must contain data_train and data_test: %s',data_file);
    end
    expected_data_hash = manifest_file_hash(data_dir,site,trial);
    actual_data_hash = sha256_file(data_file);
    if ~strcmp(actual_data_hash,expected_data_hash)
        error('ExpediaRunner:DataHashMismatch', ...
            'Live MAT hash differs from manifest for site %d trial %d.',site,trial);
    end
    split_summary = validate_expedia_data_pair( ...
        payload.data_train,payload.data_test,site);
    train_data = take_first_sessions(payload.data_train,options.MaxTrainSessions);
    test_data = take_first_sessions(payload.data_test,options.MaxTestSessions);
    is_smoke = is_smoke_requested;
    if is_smoke
        validate_expedia_matrix(train_data,'Label','smoke data_train');
        validate_expedia_matrix(test_data,'Label','smoke data_test');
    end
    pair_rows = find(tasks.site == site & tasks.trial == trial);
    for pair_row = pair_rows(:)'
        model_id = tasks.model_id(pair_row);
        seed = tasks.seed(pair_row);
        suffix = '';
        if is_smoke
            suffix = sprintf('_tr%d_te%d', ...
                size(unique(train_data(:,1)),1),size(unique(test_data(:,1)),1));
        end
        artifact_file = fullfile(fits_dir,sprintf( ...
            'fit_%d_%d_%s%s.mat',site,trial,char(model_id),suffix));
        artifact_files(pair_row) = string(artifact_file);
        if isfile(artifact_file)
            if ~options.Resume
                error('ExpediaRunner:ExistingOutput', ...
                    'Refusing to overwrite existing fit: %s',artifact_file);
            end
            loaded = load(artifact_file,'artifact');
            if ~isfield(loaded,'artifact')
                error('ExpediaRunner:InvalidExistingOutput', ...
                    'Existing MAT has no artifact variable: %s',artifact_file);
            end
            validate_expedia_fit_artifact(loaded.artifact, ...
                'ExpectedSite',site,'ExpectedTrial',trial, ...
                'ExpectedModel',model_id,'AllowSmoke',is_smoke);
            if ~strcmp(loaded.artifact.data_file_sha256,actual_data_hash) ...
                    || ~strcmp(loaded.artifact.data_file,canonical_path(data_file))
                error('ExpediaRunner:ExistingDataMismatch', ...
                    'Existing artifact does not match the live validated MAT.');
            end
            if loaded.artifact.train_sessions ~= numel(unique(train_data(:,1))) ...
                    || loaded.artifact.test_sessions ~= numel(unique(test_data(:,1)))
                error('ExpediaRunner:WrongSmokeSize', ...
                    'Existing artifact does not match the requested smoke sizes.');
            end
            if options.Verbose
                fprintf('[site %d trial %d %s] valid existing fit skipped\n', ...
                    site,trial,model_id);
            end
            continue
        end

        model_opts = frozen_model_options(config,seed,options.Verbose);
        rng(seed,'twister');
        if options.Verbose
            fprintf('[site %d trial %d %s] fitting (%d train, %d test queries)\n', ...
                site,trial,model_id,numel(unique(train_data(:,1))), ...
                numel(unique(test_data(:,1))));
        end
        fit_start = tic;
        [model,fit_diagnostics] = fit_one_model(model_id,train_data,model_opts);
        train_metrics = evaluate_expedia_model(model,train_data,model_opts, ...
            'ValidateData',false);
        test_metrics = evaluate_expedia_model(model,test_data,model_opts, ...
            'ValidateData',false);
        runtime_seconds = toc(fit_start);
        diagnostics = extract_expedia_fit_diagnostics( ...
            model_id,model,fit_diagnostics);

        artifact = struct();
        artifact.schema_version = config.schemaVersion;
        artifact.protocol = config.protocol;
        artifact.site = site;
        artifact.trial = trial;
        artifact.model_id = char(model_id);
        artifact.seed = seed;
        artifact.data_file = canonical_path(data_file);
        artifact.data_file_sha256 = actual_data_hash;
        artifact.feature_columns = config.featureCols;
        artifact.stage_size = config.stageSize;
        artifact.model = model;
        artifact.fit_diagnostics = fit_diagnostics;
        artifact.diagnostics = diagnostics;
        artifact.train_metrics = train_metrics;
        artifact.test_metrics = test_metrics;
        artifact.runtime_seconds = runtime_seconds;
        artifact.train_sessions = train_metrics.numSessions;
        artifact.test_sessions = test_metrics.numSessions;
        artifact.is_smoke = is_smoke;
        artifact.split_summary = split_summary;
        artifact.created_utc = char(datetime('now','TimeZone','UTC', ...
            'Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
        validate_expedia_fit_artifact(artifact,'ExpectedSite',site, ...
            'ExpectedTrial',trial,'ExpectedModel',model_id,'AllowSmoke',is_smoke);
        atomic_save(artifact_file,artifact);
        if options.Verbose
            fprintf('[site %d trial %d %s] saved %s\n', ...
                site,trial,model_id,artifact_file);
        end
    end
end
end


function opts = frozen_model_options(config,seed,verbose)
opts = struct();
opts.featureCols = config.featureCols;
opts.stageSize = config.stageSize;
opts.queryCol = config.queryCol;
opts.itemCol = config.itemCol;
opts.positionCol = config.positionCol;
opts.choiceCol = config.choiceCol;
opts.clickCol = config.clickCol;
opts.noPurchasePosition = config.outsidePosition;
opts.directMaxIter = config.directMaxIter;
opts.innerMaxIter = config.innerMaxIter;
opts.optimizerOptimalityTolerance = config.optimizerOptimalityTolerance;
opts.imrcsOuterMaxIter = config.imrcsOuterMaxIter;
opts.emRelativeTolerance = config.emRelativeTolerance;
opts.monotonicityTolerance = config.monotonicityTolerance;
opts.probabilityTolerance = config.probabilityTolerance;
opts.rngSeed = seed;
opts.verbose = verbose;
end


function [model,fit_diagnostics] = fit_one_model(model_id,train_data,opts)
switch string(model_id)
    case "sqmnl"
        [model,fit_diagnostics] = fit_sqmnl_model(train_data,opts);
    case "nsmnl"
        [model,fit_diagnostics] = fit_nsmnl_model(train_data,opts);
    case "iml"
        [model,fit_diagnostics] = fit_iml_model(train_data,opts);
    case "imrcs"
        [model,fit_diagnostics] = fit_imrcs_model(train_data,opts);
    otherwise
        error('ExpediaRunner:UnknownModel','Unknown model id %s.',model_id);
end
if ~isstruct(model) || ~isfield(model,'id') || string(model.id) ~= string(model_id)
    error('ExpediaRunner:ModelContract', ...
        'Fit function returned the wrong or missing model.id for %s.',model_id);
end
end


function data = take_first_sessions(data,max_sessions)
if isempty(max_sessions), return; end
starts = [1; find(diff(data(:,1)) ~= 0)+1];
ends = [starts(2:end)-1; size(data,1)];
keep = min(max_sessions,numel(starts));
data = data(1:ends(keep),:);
end


function hash = manifest_file_hash(data_dir,site,trial)
config = expedia_experiment_config();
manifest = jsondecode(fileread(fullfile(data_dir,config.manifestName)));
matches = arrayfun(@(x) double(x.site) == site ...
    && double(x.trial) == trial,manifest.files);
if sum(matches) ~= 1 || ~isfield(manifest.files(matches),'file_sha256')
    error('ExpediaRunner:MissingHash', ...
        'Manifest has no unique file hash for site %d trial %d.',site,trial);
end
hash = char(manifest.files(matches).file_sha256);
end


function atomic_save(output_file,artifact)
output_dir = fileparts(output_file);
temporary_file = [tempname(output_dir) '.mat'];
cleanup = onCleanup(@() delete_if_present(temporary_file));
save(temporary_file,'artifact','-v7.3');
if isfile(output_file)
    error('ExpediaRunner:ConcurrentOutput', ...
        'Output appeared before atomic publish: %s',output_file);
end
[ok,message] = movefile(temporary_file,output_file);
if ~ok, error('ExpediaRunner:AtomicSaveFailed','%s',message); end
clear cleanup
end


function delete_if_present(path)
if isfile(path), delete(path); end
end


function path = canonical_path(path)
path = char(java.io.File(path).getCanonicalPath());
end


function nested = same_or_nested_path(left,right)
left = canonical_path(left);
right = canonical_path(right);
nested = strcmp(left,right) ...
    || startsWith([left filesep],[right filesep]) ...
    || startsWith([right filesep],[left filesep]);
end


function valid = is_text_scalar(value)
valid = (ischar(value) && isrow(value)) ...
    || (isstring(value) && isscalar(value) && ~ismissing(value));
end


function valid = is_text_collection(value)
valid = ischar(value) || isstring(value);
end


function valid = is_logical_scalar(value)
valid = (islogical(value) || isnumeric(value)) && isscalar(value) ...
    && ismember(value,[0 1]);
end


function valid = valid_positive_integer(value)
valid = isnumeric(value) && isreal(value) && isscalar(value) ...
    && isfinite(value) && value >= 1 && value == fix(value);
end


function valid = valid_optional_positive_integer(value)
valid = isempty(value) || valid_positive_integer(value);
end
