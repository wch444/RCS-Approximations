function results = compile_expedia_results(output_dir,varargin)
%COMPILE_EXPEDIA_RESULTS Build the canonical, exactly 200-row results CSV.
%   Only full non-smoke artifacts with the frozen keys are accepted. The CSV
%   is replaced atomically after all 200 artifacts pass validation.

parser = inputParser;
parser.FunctionName = 'compile_expedia_results';
addParameter(parser,'OutputFile','expedia_four_model_results.csv',@is_text_scalar);
parse(parser,varargin{:});
options = parser.Results;
output_dir = char(string(output_dir));
fits_dir = fullfile(output_dir,'fits');
if ~isfolder(fits_dir)
    error('ExpediaResults:MissingFits','Missing fit directory: %s',fits_dir);
end

tasks = build_expedia_task_grid();
rows = repmat(empty_row(),height(tasks),1);
for index = 1:height(tasks)
    artifact_file = fullfile(fits_dir,sprintf('fit_%d_%d_%s.mat', ...
        tasks.site(index),tasks.trial(index),char(tasks.model_id(index))));
    if ~isfile(artifact_file)
        error('ExpediaResults:IncompleteGrid', ...
            'Missing canonical artifact %d/%d: %s',index,height(tasks),artifact_file);
    end
    loaded = load(artifact_file,'artifact');
    if ~isfield(loaded,'artifact')
        error('ExpediaResults:InvalidArtifact','No artifact variable in %s.',artifact_file);
    end
    validate_expedia_fit_artifact(loaded.artifact, ...
        'ExpectedSite',tasks.site(index),'ExpectedTrial',tasks.trial(index), ...
        'ExpectedModel',tasks.model_id(index));
    rows(index) = expedia_result_row(loaded.artifact);
end
results = struct2table(rows,'AsArray',true);
if height(results) ~= 200 ...
        || height(unique(results(:,{'site','trial','model_id'}),'rows')) ~= 200
    error('ExpediaResults:InvalidKeys', ...
        'Canonical results must have exactly 200 unique site/trial/model rows.');
end
if ~isequal(string(results.model_id),tasks.model_id) ...
        || ~isequal(results.site,tasks.site) || ~isequal(results.trial,tasks.trial)
    error('ExpediaResults:WrongOrdering', ...
        'Canonical result ordering must match site, trial, then frozen model order.');
end

output_file = fullfile(output_dir,char(string(options.OutputFile)));
temporary_file = [tempname(output_dir) '.csv'];
cleanup = onCleanup(@() delete_if_present(temporary_file));
writetable(results,temporary_file);
[ok,message] = movefile(temporary_file,output_file,'f');
if ~ok, error('ExpediaResults:AtomicWriteFailed','%s',message); end
clear cleanup
fprintf('Wrote %d canonical rows to %s\n',height(results),output_file);
end


function row = empty_row()
row = struct( ...
    'schema_version',"",'site',NaN,'trial',NaN,'model_id',"",'seed',NaN, ...
    'data_file',"",'data_file_sha256',"",'feature_columns',"", ...
    'stage_size',NaN,'train_sessions',NaN,'test_sessions',NaN, ...
    'train_nll',NaN,'train_nll_per_query',NaN, ...
    'test_nll',NaN,'test_nll_per_query',NaN, ...
    'test_hit1',NaN,'test_hit2',NaN,'test_hit3',NaN, ...
    'test_booked_queries',NaN,'test_outside_queries',NaN, ...
    'max_probability_sum_error',NaN,'min_observed_probability',NaN, ...
    'runtime_seconds',NaN,'convergence_status',"",'converged',false, ...
    'iterations',NaN,'max_iterations',NaN,'hit_iteration_cap',false, ...
    'final_loglik_gain',NaN,'final_relative_loglik_gain',NaN, ...
    'monotonicity_violations',NaN,'worst_loglik_decrease',NaN, ...
    'inner_optimizer_failures',NaN,'optimizer_exit_flag',NaN, ...
    'created_utc',"");
end


function delete_if_present(path)
if isfile(path), delete(path); end
end


function valid = is_text_scalar(value)
valid = (ischar(value) && isrow(value)) ...
    || (isstring(value) && isscalar(value) && ~ismissing(value));
end
