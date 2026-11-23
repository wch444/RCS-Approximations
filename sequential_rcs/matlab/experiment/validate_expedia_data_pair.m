function summary = validate_expedia_data_pair(train_data,test_data,site,varargin)
%VALIDATE_EXPEDIA_DATA_PAIR Validate schema, sizes, site, and split disjointness.

config = expedia_experiment_config();
parser = inputParser;
parser.FunctionName = 'validate_expedia_data_pair';
addParameter(parser,'ExpectedTrainSessions',config.expectedTrainSessions,@valid_count);
addParameter(parser,'ExpectedTestSessions',config.expectedTestSessions,@valid_count);
parse(parser,varargin{:});
options = parser.Results;

train_summary = validate_expedia_matrix(train_data, ...
    'ExpectedSessions',options.ExpectedTrainSessions, ...
    'ExpectedSite',site,'Label','data_train');
test_summary = validate_expedia_matrix(test_data, ...
    'ExpectedSessions',options.ExpectedTestSessions, ...
    'ExpectedSite',site,'Label','data_test');
overlap = intersect(train_summary.sessionIds,test_summary.sessionIds);
if ~isempty(overlap)
    error('ExpediaData:SplitOverlap', ...
        'Training and test splits share %d query ids (first overlap %.0f).', ...
        numel(overlap),overlap(1));
end

summary = struct();
summary.train = rmfield(train_summary,'sessionIds');
summary.test = rmfield(test_summary,'sessionIds');
summary.disjoint = true;
end


function valid = valid_count(value)
valid = isnumeric(value) && isreal(value) && isscalar(value) ...
    && isfinite(value) && value >= 1 && value == fix(value);
end
