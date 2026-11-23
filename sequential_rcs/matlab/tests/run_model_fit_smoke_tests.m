function run_model_fit_smoke_tests()
%RUN_MODEL_FIT_SMOKE_TESTS Tiny end-to-end fit/predict coverage for all models.
this_file = mfilename('fullpath');
model_dir = fullfile(fileparts(fileparts(this_file)), 'models');
addpath(model_dir);
cleanup = onCleanup(@() rmpath(model_dir)); %#ok<NASGU>

train_data = tiny_training_data();
opts = struct('stageSize', 2, 'imrcsOuterMaxIter', 100, 'directMaxIter', 30, ...
    'innerMaxIter', 30, 'numStarts', 1, 'rngSeed', 19, ...
    'optimizerOptimalityTolerance', 1e-5, ...
    'emRelativeTolerance', 1e-5, 'gradientCheck', true, 'verbose', false);

fitters = {@fit_sqmnl_model, @fit_nsmnl_model, ...
    @fit_iml_model, @fit_imrcs_model};
expected_ids = {'sqmnl', 'nsmnl', 'iml', 'imrcs'};
for j = 1:numel(fitters)
    [model, diag] = fitters{j}(train_data, opts);
    if ~strcmp(model.id, expected_ids{j}) || ~isfinite(diag.trainNegLogLik)
        error('ModelFitSmoke:Fit', '%s returned invalid fit metadata.', expected_ids{j});
    end
    if ~isstruct(diag.gradientCheck) || ~diag.gradientCheck.passed
        error('ModelFitSmoke:GradientCheck', ...
            '%s did not pass its finite-difference gradient check.', expected_ids{j});
    end
    ids = unique(train_data(:, 1));
    predicted_nll = 0;
    for t = 1:numel(ids)
        query = train_data(train_data(:, 1) == ids(t), :);
        [prob, detail] = predict_sequential_model(model, query, opts);
        assert_probability_vector(prob, 1e-8, [expected_ids{j} ' smoke']);
        chosen_product = find(detail.productRows(:, 5) == 1, 1, 'first');
        if isempty(chosen_product)
            observed_probability = prob(end);
        else
            observed_probability = prob(chosen_product);
        end
        predicted_nll = predicted_nll - log(observed_probability);
    end
    if abs(predicted_nll - diag.trainNegLogLik) > 1e-7
        error('ModelFitSmoke:LikelihoodMismatch', ...
            '%s fit and prediction NLL differ by %.17g.', ...
            expected_ids{j}, predicted_nll - diag.trainNegLogLik);
    end
end

% Stop-page counts are an explicit training-only diagnostic.  The fixture
% has three stage-1 proxies and five stage-2 proxies.
[ns_model, ~] = fit_nsmnl_model(train_data, opts);
if ~isequal(ns_model.stopCounts(:), [3; 5])
    error('ModelFitSmoke:NSStopCounts', ...
        'Unexpected NS-MNL training stop counts: [%s].', ...
        sprintf('%g ', ns_model.stopCounts));
end

fprintf('All sequential model tiny-fit smoke tests passed.\n');
end

function data = tiny_training_data()
T = 8;
rows_per_session = 5;
data = zeros(T * rows_per_session, 20);
chosen = [1; 2; 3; 4; 0; 1; 3; 0];
last_click = [1; 2; 3; 4; 0; 1; 3; 0];

for t = 1:T
    idx = (t - 1) * rows_per_session + (1:rows_per_session);
    product_idx = idx(1:4);
    outside_idx = idx(5);
    data(idx, 1) = t;
    data(product_idx, 2) = 100 * t + (1:4)';
    data(product_idx, 4) = (1:4)';
    data(outside_idx, 2) = 0;
    data(outside_idx, 4) = 100;
    if chosen(t) == 0
        data(outside_idx, 5) = 1;
    else
        data(product_idx(chosen(t)), 5) = 1;
    end
    if last_click(t) > 0
        data(product_idx(last_click(t)), 6) = 1;
    end

    data(product_idx, 8) = 1;
    data(product_idx, 9) = [-1.0; -0.2; 0.4; 1.1] + 0.03 * t;
    data(product_idx, 10) = [2; 3; 4; 5];
    data(product_idx, 11) = [0.1; 0.8; -0.4; 0.3];
    data(product_idx, 12) = [1.5; 0.5; 1.0; 2.0];
    data(product_idx, 13:20) = repmat((1:8) .* 0.01 * t, 4, 1);
end
end
