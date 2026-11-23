function metrics = evaluate_expedia_model(model,data,model_opts,varargin)
%EVALUATE_EXPEDIA_MODEL Predict each session, then use the common evaluator.

config = expedia_experiment_config();
parser = inputParser;
parser.FunctionName = 'evaluate_expedia_model';
addParameter(parser,'ProbabilityTolerance',config.probabilityTolerance,@valid_tolerance);
addParameter(parser,'ValidateData',true,@is_logical_scalar);
parse(parser,varargin{:});
options = parser.Results;
if options.ValidateData
    validate_expedia_matrix(data,'Label','model evaluation data');
end

query_ids = data(:,config.queryCol);
starts = [1; find(diff(query_ids) ~= 0)+1];
ends = [starts(2:end)-1; size(data,1)];
session_probabilities = cell(numel(starts),1);
for t = 1:numel(starts)
    query_rows = data(starts(t):ends(t),:);
    probability = predict_sequential_model(model,query_rows,model_opts);
    session_probabilities{t} = probability(:);
end
metrics = evaluate_expedia_predictions(data,session_probabilities, ...
    'ProbabilityTolerance',options.ProbabilityTolerance,'ValidateData',false);
end


function valid = valid_tolerance(value)
valid = isnumeric(value) && isreal(value) && isscalar(value) ...
    && isfinite(value) && value > 0;
end


function valid = is_logical_scalar(value)
valid = (islogical(value) || isnumeric(value)) && isscalar(value) ...
    && ismember(value,[0 1]);
end
