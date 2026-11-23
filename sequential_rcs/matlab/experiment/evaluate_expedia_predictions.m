function metrics = evaluate_expedia_predictions(data,session_probabilities,varargin)
%EVALUATE_EXPEDIA_PREDICTIONS Apply one common predictive scoring contract.
%   NLL uses every query and the observed product-or-outside probability.
%   Hit@1/2/3 uses booked queries only, excludes outside, and resolves equal
%   probabilities by displayed position, product id, then canonical row.
%   Probabilities are audited but never clipped or renormalized.

config = expedia_experiment_config();
parser = inputParser;
parser.FunctionName = 'evaluate_expedia_predictions';
addParameter(parser,'ProbabilityTolerance',config.probabilityTolerance,@valid_tolerance);
addParameter(parser,'ValidateData',true,@is_logical_scalar);
parse(parser,varargin{:});
options = parser.Results;
if options.ValidateData
    validate_expedia_matrix(data,'Label','evaluation data');
end
if ~iscell(session_probabilities)
    error('ExpediaEvaluation:InvalidPredictions', ...
        'Predictions must be a cell array with one vector per session.');
end

query_ids = data(:,config.queryCol);
starts = [1; find(diff(query_ids) ~= 0)+1];
ends = [starts(2:end)-1; size(data,1)];
num_sessions = numel(starts);
if numel(session_probabilities) ~= num_sessions
    error('ExpediaEvaluation:WrongSessionCount', ...
        'Received %d probability vectors for %d sessions.', ...
        numel(session_probabilities),num_sessions);
end

neg_log_lik = 0;
hit = zeros(3,1);
num_booked = 0;
max_sum_error = 0;
min_probability = inf;
max_probability = -inf;
min_observed_probability = inf;
for t = 1:num_sessions
    rows = data(starts(t):ends(t),:);
    n_products = size(rows,1)-1;
    probability = session_probabilities{t};
    if ~isnumeric(probability) || ~isreal(probability)
        error('ExpediaEvaluation:InvalidProbability', ...
            'Query %.0f probabilities must be a real numeric vector.',query_ids(starts(t)));
    end
    probability = probability(:);
    if numel(probability) ~= n_products+1
        error('ExpediaEvaluation:WrongProbabilityLength', ...
            'Query %.0f has %d probabilities; expected %d products plus outside.', ...
            query_ids(starts(t)),numel(probability),n_products);
    end
    if any(~isfinite(probability)) || any(probability < 0) || any(probability > 1)
        error('ExpediaEvaluation:InvalidProbability', ...
            'Query %.0f has nonfinite or out-of-range probabilities.',query_ids(starts(t)));
    end
    sum_error = abs(sum(probability)-1);
    if sum_error > options.ProbabilityTolerance
        error('ExpediaEvaluation:ProbabilityMass', ...
            'Query %.0f probabilities sum to %.17g (error %.3g); no renormalization is allowed.', ...
            query_ids(starts(t)),sum(probability),sum_error);
    end
    max_sum_error = max(max_sum_error,sum_error);
    min_probability = min(min_probability,min(probability));
    max_probability = max(max_probability,max(probability));

    chosen_row = find(rows(:,config.choiceCol) == 1);
    if numel(chosen_row) ~= 1
        error('ExpediaEvaluation:InvalidOutcome', ...
            'Query %.0f does not have exactly one observed outcome.',query_ids(starts(t)));
    end
    observed_index = chosen_row;
    observed_probability = probability(observed_index);
    min_observed_probability = min(min_observed_probability,observed_probability);
    neg_log_lik = neg_log_lik-log(observed_probability);

    if chosen_row <= n_products
        num_booked = num_booked+1;
        product_probability = probability(1:n_products);
        tie_table = [-product_probability, ...
            rows(1:n_products,config.positionCol), ...
            rows(1:n_products,config.itemCol), ...
            (1:n_products)'];
        [~,order] = sortrows(tie_table,[1 2 3 4]);
        for cutoff = 1:3
            hit(cutoff) = hit(cutoff)+ismember(chosen_row,order(1:min(cutoff,n_products)));
        end
    end
end

metrics = struct();
metrics.negLogLik = neg_log_lik;
metrics.negLogLikPerQuery = neg_log_lik/num_sessions;
metrics.hit = hit/max(num_booked,1);
if num_booked == 0
    metrics.hit(:) = NaN;
end
metrics.numSessions = num_sessions;
metrics.numBookedSessions = num_booked;
metrics.numOutsideSessions = num_sessions-num_booked;
metrics.maxProbabilitySumError = max_sum_error;
metrics.minProbability = min_probability;
metrics.maxProbability = max_probability;
metrics.minObservedProbability = min_observed_probability;
end


function valid = valid_tolerance(value)
valid = isnumeric(value) && isreal(value) && isscalar(value) ...
    && isfinite(value) && value > 0;
end


function valid = is_logical_scalar(value)
valid = (islogical(value) || isnumeric(value)) && isscalar(value) ...
    && ismember(value,[0 1]);
end
