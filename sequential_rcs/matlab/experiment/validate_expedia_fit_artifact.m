function validate_expedia_fit_artifact(artifact,varargin)
%VALIDATE_EXPEDIA_FIT_ARTIFACT Reject partial or schema-incompatible fits.

config = expedia_experiment_config();
parser = inputParser;
parser.FunctionName = 'validate_expedia_fit_artifact';
addParameter(parser,'ExpectedSite',[],@valid_optional_integer);
addParameter(parser,'ExpectedTrial',[],@valid_optional_integer);
addParameter(parser,'ExpectedModel','',@is_text_scalar);
addParameter(parser,'AllowSmoke',false,@is_logical_scalar);
parse(parser,varargin{:});
options = parser.Results;

required = {'schema_version','protocol','site','trial','model_id','seed', ...
    'data_file','data_file_sha256','feature_columns','stage_size','model', ...
    'fit_diagnostics','diagnostics','train_metrics','test_metrics', ...
    'runtime_seconds','train_sessions','test_sessions','is_smoke','created_utc'};
if ~isstruct(artifact) || ~isscalar(artifact) || ~all(isfield(artifact,required))
    error('ExpediaArtifact:Invalid','Fit artifact is missing required fields.');
end
if ~strcmp(artifact.schema_version,config.schemaVersion) ...
        || ~strcmp(artifact.protocol,config.protocol)
    error('ExpediaArtifact:WrongSchema','Fit artifact schema/protocol mismatch.');
end
model_id = string(artifact.model_id);
if ~ismember(model_id,config.modelIds)
    error('ExpediaArtifact:WrongModel','Unknown model id %s.',model_id);
end
if ~isfield(artifact.model,'id') || string(artifact.model.id) ~= model_id
    error('ExpediaArtifact:WrongModel','Saved model.id does not match artifact model_id.');
end
model_index = find(config.modelIds == model_id,1);
expected_seed = config.seedBase+1000*artifact.site+10*artifact.trial+model_index;
if artifact.seed ~= expected_seed
    error('ExpediaArtifact:WrongSeed','Artifact seed does not match the frozen key rule.');
end
if isempty(regexp(char(string(artifact.data_file_sha256)), ...
        '^[0-9a-f]{64}$','once'))
    error('ExpediaArtifact:InvalidHash','Artifact data SHA-256 is invalid.');
end
if ~isempty(options.ExpectedSite) && artifact.site ~= options.ExpectedSite
    error('ExpediaArtifact:WrongKey','Artifact site does not match its requested key.');
end
if ~isempty(options.ExpectedTrial) && artifact.trial ~= options.ExpectedTrial
    error('ExpediaArtifact:WrongKey','Artifact trial does not match its requested key.');
end
if strlength(string(options.ExpectedModel)) > 0 ...
        && model_id ~= string(options.ExpectedModel)
    error('ExpediaArtifact:WrongKey','Artifact model does not match its requested key.');
end
if ~isequal(double(artifact.feature_columns(:)'),config.featureCols) ...
        || artifact.stage_size ~= config.stageSize
    error('ExpediaArtifact:WrongFeatures','Artifact feature/stage protocol mismatch.');
end
if artifact.is_smoke && ~options.AllowSmoke
    error('ExpediaArtifact:SmokeNotCanonical', ...
        'Smoke-test artifacts cannot enter canonical results.');
end
if ~artifact.is_smoke && (artifact.train_sessions ~= config.expectedTrainSessions ...
        || artifact.test_sessions ~= config.expectedTestSessions)
    error('ExpediaArtifact:WrongSessionCount','Full fit has incomplete train/test sessions.');
end
if artifact.train_metrics.numSessions ~= artifact.train_sessions ...
        || artifact.test_metrics.numSessions ~= artifact.test_sessions
    error('ExpediaArtifact:WrongSessionCount', ...
        'Artifact session counts disagree with evaluator metrics.');
end
validate_metrics(artifact.train_metrics,'train');
validate_metrics(artifact.test_metrics,'test');
if ~isfinite(artifact.runtime_seconds) || artifact.runtime_seconds < 0
    error('ExpediaArtifact:InvalidRuntime','Artifact runtime must be finite and nonnegative.');
end
if model_id == "imrcs" && artifact.diagnostics.maxIterations ~= config.imrcsOuterMaxIter
    error('ExpediaArtifact:WrongIMRCSMaxIter','IM-RCS max iteration cap must be 100.');
end
end


function validate_metrics(metrics,label)
config = expedia_experiment_config();
required = {'negLogLik','negLogLikPerQuery','hit','numSessions', ...
    'numBookedSessions','numOutsideSessions','maxProbabilitySumError', ...
    'minProbability','maxProbability','minObservedProbability'};
if ~isstruct(metrics) || ~all(isfield(metrics,required))
    error('ExpediaArtifact:InvalidMetrics','Missing %s metric fields.',label);
end
if ~isfinite(metrics.negLogLik) || ~isfinite(metrics.negLogLikPerQuery) ...
        || numel(metrics.hit) ~= 3 ...
        || any(~isfinite(metrics.hit)) ...
        || any(metrics.hit < 0) || any(metrics.hit > 1) ...
        || metrics.maxProbabilitySumError > config.probabilityTolerance ...
        || metrics.minProbability < 0 || metrics.maxProbability > 1 ...
        || metrics.minObservedProbability <= 0
    error('ExpediaArtifact:InvalidMetrics','Invalid %s predictive metrics.',label);
end
end


function valid = valid_optional_integer(value)
valid = isempty(value) || (isnumeric(value) && isreal(value) && isscalar(value) ...
    && isfinite(value) && value == fix(value));
end


function valid = is_text_scalar(value)
valid = (ischar(value) && isrow(value)) ...
    || (isstring(value) && isscalar(value) && ~ismissing(value));
end


function valid = is_logical_scalar(value)
valid = (islogical(value) || isnumeric(value)) && isscalar(value) ...
    && ismember(value,[0 1]);
end
