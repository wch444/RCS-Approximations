function [prob, detail] = predict_sequential_model(model, query_rows, opts)
%PREDICT_SEQUENTIAL_MODEL Common feature-only prediction interface.
%
% The returned vector is aligned to products sorted by
% (displayed position, item id, input row), followed by the outside option.
% Clicks are never read during prediction.  In particular, NS-MNL always
% marginalizes its training-estimated stop-page distribution.

if nargin < 3 || isempty(opts)
    if isfield(model, 'opts')
        opts = model.opts;
    else
        opts = struct();
    end
end
opts = sequential_model_options(opts);
if ~isfield(model, 'id') || ~isfield(model, 'theta')
    error('SequentialModels:BadModel', 'model.id and model.theta are required.');
end

% Prediction deliberately ignores click values.  Replace them before
% parsing so a click-leakage regression test can exercise this guarantee.
query_rows(:, opts.clickCol) = 0;
parsed = prepare_sequential_sessions(query_rows, opts);
if parsed.numSessions ~= 1
    error('SequentialModels:OneSession', ...
        'predict_sequential_model expects exactly one query/session.');
end
s = parsed.sessions(1);
X = parsed.X(s.idx, :);
eta = X * model.theta(:);

switch lower(model.id)
    case 'sqmnl'
        prob = sqmnl_session_probability(eta, s.stageOf, opts.probabilityTolerance);
        rank_pos = [];
    case 'nsmnl'
        require_field(model, 'lambda');
        prob = nsmnl_session_probability(eta, s.stageOf, model.lambda, ...
            opts.probabilityTolerance);
        rank_pos = [];
    case 'iml'
        require_field(model, 'p');
        prob = iml_session_probability(eta, s.stageOf, model.p, ...
            opts.probabilityTolerance);
        rank_pos = [];
    case 'imrcs'
        require_field(model, 'p');
        require_field(model, 'beta');
        require_field(model, 'rankings');
        product_rows = parsed.productData(s.idx, :);
        rank_pos = imrcs_rank_positions(product_rows, model.rankings, opts);
        prob = imrcs_session_probability(eta, s.stageOf, rank_pos, ...
            model.p, model.beta, opts.probabilityTolerance);
    otherwise
        error('SequentialModels:UnknownModel', 'Unknown model id %s.', model.id);
end

detail = struct();
detail.queryId = s.id;
detail.stageOf = s.stageOf;
detail.productRows = parsed.productData(s.idx, :);
detail.originalInputRows = s.canonicalInputRows;
detail.rankPositions = rank_pos;
detail.probabilityAudit = assert_probability_vector(prob, ...
    opts.probabilityTolerance, upper(model.id));
end

function require_field(model, name)
if ~isfield(model, name)
    error('SequentialModels:BadModel', 'Model %s is missing field %s.', model.id, name);
end
end
