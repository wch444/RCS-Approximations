function tests = test_experiment_layer
%TEST_EXPERIMENT_LAYER Unit tests for scheduling, schema, and scoring only.

tests = functiontests(localfunctions);
end


function setupOnce(test_case)
tests_dir = fileparts(mfilename('fullpath'));
matlab_dir = fileparts(tests_dir);
addpath(fullfile(matlab_dir,'experiment'),'-begin');
test_case.TestData.realDataDir = fullfile(fileparts(fileparts(matlab_dir)), ...
    'data.nosync','expedia_no_click_filter_no_position_20260816');
end


function testDeterministicFullTaskGrid(test_case)
tasks_a = build_expedia_task_grid();
tasks_b = build_expedia_task_grid();
verifyEqual(test_case,tasks_a,tasks_b);
verifyEqual(test_case,height(tasks_a),200);
verifyEqual(test_case,tasks_a(1,{'site','trial','model_id'}), ...
    table(5,1,"sqmnl",'VariableNames',{'site','trial','model_id'}));
verifyEqual(test_case,tasks_a(end,{'site','trial','model_id'}), ...
    table(32,10,"imrcs",'VariableNames',{'site','trial','model_id'}));
verifyEqual(test_case,numel(unique(tasks_a.seed)),200);
end


function testValidatorRetainsMoreThanFourClicks(test_case)
data = synthetic_data(true);
summary = validate_expedia_matrix(data,'ExpectedSessions',2,'ExpectedSite',5);
verifyEqual(test_case,summary.queriesClickGreaterThan4,1);
verifyEqual(test_case,summary.columns,20);
verifyEqual(test_case,summary.bookedSessions,1);
end


function testCommonNLLAndDeterministicHits(test_case)
data = synthetic_data(false);
probabilities = {[0.3;0.3;0.1;0.3]; [0.2;0.1;0.7]};
metrics = evaluate_expedia_predictions(data,probabilities);
verifyEqual(test_case,metrics.negLogLik,-log(0.3)-log(0.7),'AbsTol',1e-12);
verifyEqual(test_case,metrics.negLogLikPerQuery, ...
    (-log(0.3)-log(0.7))/2,'AbsTol',1e-12);
% Products one and two tie. Displayed position puts product one first, so the
% chosen second product misses @1 but appears at @2 and @3.
verifyEqual(test_case,metrics.hit,[0;1;1]);
verifyEqual(test_case,metrics.numBookedSessions,1);
verifyEqual(test_case,metrics.numOutsideSessions,1);
end


function testNoSilentProbabilityRenormalization(test_case)
data = synthetic_data(false);
bad = {[0.3;0.3;0.1;0.2]; [0.2;0.1;0.7]};
verifyError(test_case,@() evaluate_expedia_predictions(data,bad), ...
    'ExpediaEvaluation:ProbabilityMass');
end


function testIMRCSDiagnostics(test_case)
model = struct('history',struct('logLik',[-10;-9;-8.9999995]), ...
    'opts',struct('imrcsOuterMaxIter',100));
fitdiag = struct('converged',true,'maxIterations',100);
diagnostics = extract_expedia_fit_diagnostics('imrcs',model,fitdiag);
verifyEqual(test_case,diagnostics.iterations,2);
verifyTrue(test_case,diagnostics.converged);
verifyEqual(test_case,diagnostics.monotonicityViolations,0);
verifyEqual(test_case,diagnostics.status,'converged');

model.history.logLik = [-10;-9;-9.01];
diagnostics = extract_expedia_fit_diagnostics('imrcs',model,fitdiag);
verifyFalse(test_case,diagnostics.converged);
verifyEqual(test_case,diagnostics.monotonicityViolations,1);
verifyEqual(test_case,diagnostics.status,'monotonicity_violation');
end


function testRealManifestAndSingleTaskDryRun(test_case)
data_dir = canonical_if_present(test_case.TestData.realDataDir);
assumeTrue(test_case,isfolder(data_dir), ...
    'Generated no-click-filter data is not present in this checkout.');
manifest = validate_expedia_manifest(data_dir,5,1);
verifyEqual(test_case,manifest.selectedFiles,1);
verifyEqual(test_case,sha256_file(fullfile(data_dir, ...
    'expedia_preprocess_manifest.json')), ...
    '419b9646ee56b446149f8fce645cdd6b310e30f5199cf96a8cf5c7b82d44d09f');
[tasks,files] = run_expedia_four_model_grid( ...
    'DataDir',data_dir,'Sites',5,'Trials',1,'Models','sqmnl', ...
    'DryRun',true,'Verbose',false);
verifyEqual(test_case,height(tasks),1);
verifyEqual(test_case,tasks.model_id,"sqmnl");
verifyEqual(test_case,files,"");
end


function testRealSiteFiveTrialOneMatrices(test_case)
data_dir = canonical_if_present(test_case.TestData.realDataDir);
assumeTrue(test_case,isfolder(data_dir), ...
    'Generated no-click-filter data is not present in this checkout.');
payload = load(fullfile(data_dir,'10k_sample_5_1.mat'), ...
    'data_train','data_test');
summary = validate_expedia_data_pair(payload.data_train,payload.data_test,5);
verifyEqual(test_case,summary.train.sessions,6000);
verifyEqual(test_case,summary.test.sessions,4000);
verifyGreaterThan(test_case,summary.train.queriesClickGreaterThan4,0);
verifyGreaterThan(test_case,summary.test.queriesClickGreaterThan4,0);
end


function testFullRunRequiresExplicitConfirmation(test_case)
data_dir = canonical_if_present(test_case.TestData.realDataDir);
assumeTrue(test_case,isfolder(data_dir), ...
    'Generated no-click-filter data is not present in this checkout.');
verifyError(test_case,@() run_expedia_four_model_grid( ...
    'DataDir',data_dir,'OutputDir',tempdir,'DryRun',false, ...
    'RequireFullRunConfirmation',false,'Verbose',false), ...
    'ExpediaRunner:FullRunNotConfirmed');
verifyError(test_case,@() run_expedia_four_model_grid( ...
    'DataDir',data_dir,'OutputDir',tempdir,'Sites',[32 24 15 14 5], ...
    'Trials',10:-1:1,'Models',["imrcs" "iml" "nsmnl" "sqmnl"], ...
    'DryRun',false,'RequireFullRunConfirmation',false,'Verbose',false), ...
    'ExpediaRunner:FullRunNotConfirmed');
end


function testCanonicalCompilerKeepsAllNonconvergedRows(test_case)
config = expedia_experiment_config();
temporary_root = tempname;
fits_dir = fullfile(temporary_root,'fits');
mkdir(fits_dir);
cleanup = onCleanup(@() remove_temp_tree(temporary_root)); %#ok<NASGU>
tasks = build_expedia_task_grid();
for index = 1:height(tasks)
    artifact = mock_artifact(tasks.site(index),tasks.trial(index), ...
        tasks.model_id(index),tasks.seed(index),config);
    output_file = fullfile(fits_dir,sprintf('fit_%d_%d_%s.mat', ...
        artifact.site,artifact.trial,artifact.model_id));
    save(output_file,'artifact');
end
results = compile_expedia_results(temporary_root);
verifyEqual(test_case,height(results),200);
verifyEqual(test_case,height(unique(results(:,{'site','trial','model_id'}),'rows')),200);
verifyEqual(test_case,sum(results.convergence_status == "hit_cap"),50);
verifyTrue(test_case,isfile(fullfile(temporary_root, ...
    'expedia_four_model_results.csv')));
end


function data = synthetic_data(click_gt_four)
if click_gt_four, n_first = 6; else, n_first = 3; end
data = zeros(n_first+4,20);
% Query 101: products and one outside row, with product 2 booked.
for i = 1:n_first
    data(i,1:8) = [101,1000+i,5,i-1,i==2,click_gt_four,100+i,1];
end
data(n_first+1,1:7) = [101,0,0,100,0,0,0];
% Query 202: two products and an outside choice.
data(n_first+2,1:8) = [202,2001,5,0,0,0,90,1];
data(n_first+3,1:8) = [202,2002,5,1,0,0,95,1];
data(n_first+4,1:7) = [202,0,0,100,1,0,0];
end


function artifact = mock_artifact(site,trial,model_id,seed,config)
metric = struct('negLogLik',100,'negLogLikPerQuery',0.025, ...
    'hit',[0.1;0.2;0.3],'numSessions',4000,'numBookedSessions',2800, ...
    'numOutsideSessions',1200,'maxProbabilitySumError',1e-15, ...
    'minProbability',1e-6,'maxProbability',0.5, ...
    'minObservedProbability',1e-4);
train_metric = metric;
train_metric.numSessions = 6000;
train_metric.numBookedSessions = 4200;
train_metric.numOutsideSessions = 1800;
diagnostics = struct('status','converged','converged',true, ...
    'iterations',10,'maxIterations',500,'optimizerExitFlag',1, ...
    'hitIterationCap',false,'finalLogLikGain',NaN, ...
    'finalRelativeLogLikGain',NaN,'monotonicityViolations',0, ...
    'worstLogLikDecrease',0,'innerOptimizerFailures',0);
if model_id == "imrcs"
    diagnostics.status = 'hit_cap';
    diagnostics.converged = false;
    diagnostics.iterations = 100;
    diagnostics.maxIterations = 100;
    diagnostics.optimizerExitFlag = NaN;
    diagnostics.hitIterationCap = true;
    diagnostics.finalLogLikGain = 0.01;
    diagnostics.finalRelativeLogLikGain = 1e-5;
end
artifact = struct();
artifact.schema_version = config.schemaVersion;
artifact.protocol = config.protocol;
artifact.site = site;
artifact.trial = trial;
artifact.model_id = char(model_id);
artifact.seed = seed;
artifact.data_file = '/tmp/mock-expedia.mat';
artifact.data_file_sha256 = repmat('0',1,64);
artifact.feature_columns = config.featureCols;
artifact.stage_size = config.stageSize;
artifact.model = struct('id',char(model_id),'theta',zeros(13,1));
artifact.fit_diagnostics = struct();
artifact.diagnostics = diagnostics;
artifact.train_metrics = train_metric;
artifact.test_metrics = metric;
artifact.runtime_seconds = 1;
artifact.train_sessions = 6000;
artifact.test_sessions = 4000;
artifact.is_smoke = false;
artifact.created_utc = '2026-08-16T00:00:00Z';
end


function remove_temp_tree(path)
if isfolder(path), rmdir(path,'s'); end
end


function path = canonical_if_present(path)
path = char(java.io.File(path).getCanonicalPath());
end
