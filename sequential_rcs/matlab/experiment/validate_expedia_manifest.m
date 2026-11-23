function summary = validate_expedia_manifest(data_dir,sites,trials)
%VALIDATE_EXPEDIA_MANIFEST Verify the frozen no-filter preprocessing contract.

config = expedia_experiment_config();
if nargin < 2 || isempty(sites), sites = config.sites; end
if nargin < 3 || isempty(trials), trials = config.trials; end
data_dir = char(string(data_dir));
manifest_file = fullfile(data_dir,config.manifestName);
if ~isfile(manifest_file)
    error('ExpediaManifest:Missing','Missing preprocessing manifest: %s',manifest_file);
end
manifest = jsondecode(fileread(manifest_file));

require_equal(manifest,'protocol',config.protocol);
require_equal(manifest,'input_sha256',config.rawInputSha256);
require_equal(manifest,'matrix_column_count',config.expectedColumns);
require_equal(manifest,'query_sample_size',10000);
require_equal(manifest,'train_queries',config.expectedTrainSessions);
require_equal(manifest,'test_queries',config.expectedTestSessions);
require_equal(manifest,'split_seed','trial');
require_equal(manifest,'standardization','separate train/test sample SD (ddof=1)');
require_equal(manifest,'click_count_filter_enabled',false);
require_equal(manifest,'position_encoding_enabled',false);
if ~isfield(manifest,'click_count_filter') || ~isempty(manifest.click_count_filter)
    error('ExpediaManifest:ClickFilterEnabled', ...
        'click_count_filter must be JSON null/empty.');
end
if ~isfield(manifest,'position_encoding') || ~isempty(manifest.position_encoding)
    error('ExpediaManifest:PositionEncodingEnabled', ...
        'position_encoding must be JSON null/empty.');
end
if ~isfield(manifest,'model_feature_column_range_1_based') ...
        || ~isequal(double(manifest.model_feature_column_range_1_based(:)'),[8 20])
    error('ExpediaManifest:WrongFeatures', ...
        'Model feature range must be exactly [8,20].');
end
if ~isfield(manifest,'matrix_columns') ...
        || ~isequal(string(manifest.matrix_columns(:)'),config.matrixColumns)
    error('ExpediaManifest:WrongColumns', ...
        'matrix_columns does not match the frozen 20-column schema.');
end
if ~isfield(manifest,'model_feature_columns') ...
        || ~isequal(string(manifest.model_feature_columns(:)'),config.modelFeatureColumns)
    error('ExpediaManifest:WrongFeatures', ...
        'model_feature_columns does not match columns 8:20.');
end
if ~isfield(manifest,'files') || numel(manifest.files) ~= 50
    error('ExpediaManifest:WrongFileCount', ...
        'The manifest must contain exactly 50 file records.');
end

selected_count = 0;
for site = sites(:)'
    for trial = trials(:)'
        matches = arrayfun(@(x) double(x.site) == site ...
            && double(x.trial) == trial,manifest.files);
        if sum(matches) ~= 1
            error('ExpediaManifest:MissingFileRecord', ...
                'Expected exactly one manifest record for site %d trial %d.',site,trial);
        end
        record = manifest.files(matches);
        expected_name = sprintf(config.dataPattern,site,trial);
        if ~strcmp(char(string(record.file)),expected_name) ...
                || double(record.columns) ~= config.expectedColumns ...
                || double(record.train_queries) ~= config.expectedTrainSessions ...
                || double(record.test_queries) ~= config.expectedTestSessions
            error('ExpediaManifest:InvalidFileRecord', ...
                'Invalid manifest record for site %d trial %d.',site,trial);
        end
        if ~isfile(fullfile(data_dir,expected_name))
            error('ExpediaManifest:MissingDataFile', ...
                'Manifest data file is missing: %s',fullfile(data_dir,expected_name));
        end
        selected_count = selected_count+1;
    end
end

summary = struct();
summary.file = manifest_file;
summary.protocol = char(manifest.protocol);
summary.standardization = char(manifest.standardization);
summary.selectedFiles = selected_count;
summary.totalFiles = numel(manifest.files);
summary.inputSha256 = char(manifest.input_sha256);
end


function require_equal(value_struct,field_name,expected)
if ~isfield(value_struct,field_name)
    error('ExpediaManifest:MissingField','Missing manifest field %s.',field_name);
end
actual = value_struct.(field_name);
if ischar(expected) || isstring(expected)
    valid = strcmp(char(string(actual)),char(string(expected)));
else
    valid = isequal(actual,expected) || (isnumeric(actual) && isequal(double(actual),double(expected)));
end
if ~valid
    error('ExpediaManifest:WrongField','Manifest field %s has an unexpected value.',field_name);
end
end
