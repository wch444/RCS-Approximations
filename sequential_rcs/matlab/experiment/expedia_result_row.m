function row = expedia_result_row(artifact)
%EXPEDIA_RESULT_ROW Flatten one validated fit into the canonical CSV schema.

validate_expedia_fit_artifact(artifact);
config = expedia_experiment_config();
train = artifact.train_metrics;
test = artifact.test_metrics;
diag = artifact.diagnostics;
row = struct();
row.schema_version = string(config.resultSchemaVersion);
row.site = artifact.site;
row.trial = artifact.trial;
row.model_id = string(artifact.model_id);
row.seed = artifact.seed;
row.data_file = string(artifact.data_file);
row.data_file_sha256 = string(artifact.data_file_sha256);
row.feature_columns = "8:20";
row.stage_size = artifact.stage_size;
row.train_sessions = artifact.train_sessions;
row.test_sessions = artifact.test_sessions;
row.train_nll = train.negLogLik;
row.train_nll_per_query = train.negLogLikPerQuery;
row.test_nll = test.negLogLik;
row.test_nll_per_query = test.negLogLikPerQuery;
row.test_hit1 = test.hit(1);
row.test_hit2 = test.hit(2);
row.test_hit3 = test.hit(3);
row.test_booked_queries = test.numBookedSessions;
row.test_outside_queries = test.numOutsideSessions;
row.max_probability_sum_error = max( ...
    train.maxProbabilitySumError,test.maxProbabilitySumError);
row.min_observed_probability = min( ...
    train.minObservedProbability,test.minObservedProbability);
row.runtime_seconds = artifact.runtime_seconds;
row.convergence_status = string(diag.status);
row.converged = logical(diag.converged);
row.iterations = diag.iterations;
row.max_iterations = diag.maxIterations;
row.hit_iteration_cap = logical(diag.hitIterationCap);
row.final_loglik_gain = diag.finalLogLikGain;
row.final_relative_loglik_gain = diag.finalRelativeLogLikGain;
row.monotonicity_violations = diag.monotonicityViolations;
row.worst_loglik_decrease = diag.worstLogLikDecrease;
row.inner_optimizer_failures = diag.innerOptimizerFailures;
row.optimizer_exit_flag = diag.optimizerExitFlag;
row.created_utc = string(artifact.created_utc);
end
