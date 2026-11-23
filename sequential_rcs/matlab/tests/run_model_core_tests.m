function run_model_core_tests()
%RUN_MODEL_CORE_TESTS Exact probability-oracle and leakage regression tests.
this_file = mfilename('fullpath');
model_dir = fullfile(fileparts(fileparts(this_file)), 'models');
addpath(model_dir);
cleanup = onCleanup(@() rmpath(model_dir)); %#ok<NASGU>

tolerance = 5e-13;
eta = log([1; 2; 3]);
stage_of = [1; 1; 2];

sq_expected = [1/4; 2/4; 3/16; 1/16];
assert_close(sqmnl_session_probability(eta, stage_of), sq_expected, ...
    tolerance, 'SQ-MNL two-stage oracle');

stop_lambda = [1/4; 3/4];
ns_expected = [1/16 + 3/28; 1/8 + 3/14; 9/28; 1/16 + 3/28];
assert_close(nsmnl_session_probability(eta, stage_of, stop_lambda), ...
    ns_expected, tolerance, 'NS-MNL marginal-stop oracle');

patience = [1/4; 3/4];
iml_expected = [1/4; 1/2; 9/112; 1/16 + 3/28];
assert_close(iml_session_probability(eta, stage_of, patience), ...
    iml_expected, tolerance, 'IML cumulative-denominator oracle');

q = [0.2; 0.5; 0.8];
imrcs_eta = log(q ./ (1 - q));
rank_pos = [1 2; 2 1; 1 1];
ranking_weight = [0.3; 0.7];
imrcs_expected = [0.13; 0.47; 0.24; 0.16];
assert_close(imrcs_session_probability(imrcs_eta, stage_of, rank_pos, ...
    patience, ranking_weight), imrcs_expected, tolerance, ...
    'IM-RCS ranking-mixture oracle');

% Each model reduces to ordinary MNL when a session has one stage and the
% latent stopping/patience distribution is degenerate at that stage.
mnl_expected = [1; 2; 1] ./ 4;
one_stage = [1; 1];
assert_close(sqmnl_session_probability(log([1; 2]), one_stage), ...
    mnl_expected, tolerance, 'SQ-MNL one-stage reduction');
assert_close(nsmnl_session_probability(log([1; 2]), one_stage, 1), ...
    mnl_expected, tolerance, 'NS-MNL degenerate-stop reduction');
assert_close(iml_session_probability(log([1; 2]), one_stage, 1), ...
    mnl_expected, tolerance, 'IML one-stage reduction');

% Prediction must be invariant to test clicks for every model, especially
% NS-MNL.  It also verifies canonical (position,item,row) ordering.
opts = struct('stageSize', 2, 'imrcsOuterMaxIter', 100);
query = tiny_query();
theta = zeros(13, 1);
theta(2) = 1;
ns_model = struct('id', 'nsmnl', 'theta', theta, ...
    'lambda', stop_lambda, 'opts', sequential_model_options(opts));
query_click_variant = query;
query_click_variant(1:3, 6) = [1; 0; 1];
[prob_a, detail] = predict_sequential_model(ns_model, query, opts);
prob_b = predict_sequential_model(ns_model, query_click_variant, opts);
assert_close(prob_a, prob_b, 0, 'NS-MNL test-click invariance');
if ~isequal(detail.productRows(:, 2), [101; 102; 103])
    error('ModelCoreTests:CanonicalOrder', 'Unexpected canonical product ordering.');
end

% The feature schema must reject any position encoding or trailing feature.
bad_opts = opts;
bad_opts.featureCols = 8:21;
did_reject = false;
try
    sequential_model_options(bad_opts);
catch exception
    did_reject = strcmp(exception.identifier, 'SequentialModels:FeatureSchema');
end
if ~did_reject
    error('ModelCoreTests:FeatureSchema', 'Noncanonical feature columns were not rejected.');
end

fprintf('All sequential model core tests passed.\n');
end

function data = tiny_query()
data = zeros(4, 20);
data(:, 1) = 7;
data(1:3, 2) = [102; 101; 103];
data(1:3, 4) = [2; 1; 3];
data(4, 2) = 0;
data(4, 4) = 100;
data(4, 5) = 1;
data(1:3, 8) = 1;
data(1:3, 9) = log([2; 1; 3]);
end

function assert_close(actual, expected, tolerance, label)
actual = actual(:);
expected = expected(:);
if numel(actual) ~= numel(expected) || any(abs(actual - expected) > tolerance)
    error('ModelCoreTests:Mismatch', ...
        '%s failed. max absolute error %.17g.', label, ...
        max(abs(actual - expected)));
end
end
