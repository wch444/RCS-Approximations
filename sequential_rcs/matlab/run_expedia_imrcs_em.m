% Run the Impatient Mixed-RCS EM estimator on an Expedia sample.
%
% Edit the block below for full runs. Keeping sampleTrainSessions nonempty is
% useful for a quick smoke test before launching the full data job.

clear;
clc;

scriptDir = fileparts(mfilename('fullpath'));
basePath = string(fileparts(fileparts(scriptDir)));
addpath(fullfile(basePath, "sequential_rcs", "matlab"));

site = 5;
trial = 1;
sampleTrainSessions = [];
sampleTestSessions = [];

dataFile = fullfile(basePath, "data.nosync", "10k_sample_" + site + "_" + trial + ".mat");
outFile = fullfile(basePath, "opt_data", "imrcs_em_" + site + "_" + trial + ".mat");
recordFile = fullfile(basePath, "record", "imrcs_em_expedia.txt");

D = load(dataFile);
trainData = D.data_train;
testData = D.data_test;

if ~isempty(sampleTrainSessions)
    trainData = imrcs_take_first_sessions(trainData, 1, sampleTrainSessions);
end
if ~isempty(sampleTestSessions)
    testData = imrcs_take_first_sessions(testData, 1, sampleTestSessions);
end

opts = struct();
opts.featureStart = 8;
opts.stageSize = 10;
opts.maxIter = 30;
opts.tol = 1e-4;
opts.rngSeed = 42 + 1000 * trial + site;
opts.verbose = true;
opts.optimOptions = optimoptions('fminunc', ...
    'Display', 'off', ...
    'Algorithm', 'quasi-newton', ...
    'SpecifyObjectiveGradient', true, ...
    'MaxIterations', 100, ...
    'MaxFunctionEvaluations', 5e4, ...
    'OptimalityTolerance', 1e-5);

tic;
[model, metrics] = IMRCS_EM_function(trainData, testData, opts);
elapsed = toc;

result = struct();
result.model = model;
result.metrics = metrics;
result.site = site;
result.trial = trial;
result.dataFile = dataFile;
result.elapsed = elapsed;
result.sampleTrainSessions = sampleTrainSessions;
result.sampleTestSessions = sampleTestSessions;

save(outFile, 'result');

fileID = fopen(recordFile, 'a+');
fprintf(fileID, ...
    'IMRCS_EM %d %d trainNLL %.6f testNLL %.6f testRMSE %.6f testWSQ %.6f hit1 %.6f hit2 %.6f hit3 %.6f time %.3f\n', ...
    site, trial, metrics.train.negLogLik, metrics.test.negLogLik, ...
    metrics.test.rmse, metrics.test.weightedSq, metrics.test.hit(1), ...
    metrics.test.hit(2), metrics.test.hit(3), elapsed);
fclose(fileID);

fprintf('Saved %s\n', outFile);
fprintf('Train negLL %.6f, test negLL %.6f, test hit %.4f %.4f %.4f\n', ...
    metrics.train.negLogLik, metrics.test.negLogLik, metrics.test.hit(1), ...
    metrics.test.hit(2), metrics.test.hit(3));

function out = imrcs_take_first_sessions(data, queryCol, nSessions)
sessionStart = [1; find(diff(data(:, queryCol)) ~= 0) + 1];
sessionEnd = [sessionStart(2:end) - 1; size(data, 1)];
keep = 1:min(nSessions, numel(sessionStart));
lastRow = sessionEnd(keep(end));
out = data(1:lastRow, :);
end
