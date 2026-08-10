function [trials, summary, manifest, run_dir] = run_synthetic_experiments( ...
        n_rep, master_seed, use_parallel, n_list, b_list, output_dir)
%RUN_SYNTHETIC_EXPERIMENTS Deterministic CMNL/SCM synthetic experiment.
%   This producer never publishes canonical paper files directly.  It writes
%   one complete MATLAB generation (trial CSV, summary CSV, and producer
%   manifest) to an empty staging directory.  The Python reporting stage
%   revalidates the 18,000 trial rows, recomputes the summary, and atomically
%   publishes the complete paper-output generation.
%
%   A call without OUTPUT_DIR creates a unique run directory under
%   experiments/synthetic/results/synthetic_runs.  Consequently, subset and
%   smoke calls cannot overwrite the canonical production generation.

if nargin < 1 || isempty(n_rep), n_rep = 1000; end
if nargin < 2 || isempty(master_seed), master_seed = 20260808; end
if nargin < 3 || isempty(use_parallel), use_parallel = true; end
if nargin < 4 || isempty(n_list), n_list = [10, 15, 20]; end
if nargin < 5 || isempty(b_list), b_list = [3, 5, 8]; end
if nargin < 6, output_dir = []; end

[n_rep, master_seed, use_parallel, n_list, b_list] = validate_design( ...
    n_rep, master_seed, use_parallel, n_list, b_list);

here = fileparts(mfilename('fullpath'));
experiment_root = fileparts(here);
% This producer has no shared-code dependency; MATLAB resolves its two
% experiment-specific helpers from the adjacent private directory.
% Reject the canonical publication directory before inspecting its existing
% contents.  This keeps the safety contract stable after a successful run,
% when the directory is necessarily nonempty.
if nargin >= 6 && ~isempty(output_dir) ...
        && (ischar(output_dir) ...
        || (isstring(output_dir) && isscalar(output_dir))) ...
        && same_directory(char(string(output_dir)), ...
        fullfile(experiment_root, 'results'))
    error('run_synthetic_experiments:DirectProductionWrite', ...
        ['MATLAB synthetic runs must write to an isolated staging ' ...
        'directory; only the validated Python transaction may publish.']);
end
run_dir = resolve_run_directory(experiment_root, output_dir, nargin >= 6);
if same_directory(run_dir, fullfile(experiment_root, 'results'))
    error('run_synthetic_experiments:DirectProductionWrite', ...
        ['MATLAB synthetic runs must write to an isolated staging ' ...
        'directory; only the validated Python transaction may publish.']);
end

models = ["CMNL", "SCM"];
assortment_cache = build_assortment_cache(n_list, b_list);

n_total = numel(models) * numel(n_list) * numel(b_list) * n_rep;
model_col = strings(n_total, 1);
n_col = zeros(n_total, 1);
b_col = zeros(n_total, 1);
rep_col = zeros(n_total, 1);
seed_col = zeros(n_total, 1);
heuristic_col = zeros(n_total, 1);
optimum_col = zeros(n_total, 1);
ratio_col = zeros(n_total, 1);
row = 0;

for model_index = 1:numel(models)
    model = models(model_index);
    for n_index = 1:numel(n_list)
        n = n_list(n_index);
        for b_index = 1:numel(b_list)
            b = b_list(b_index);
            assortments = assortment_cache{n_index, b_index};
            block_seed = zeros(n_rep, 1);
            for replication = 1:n_rep
                block_seed(replication) = task_seed( ...
                    master_seed, model_index, n, b, replication);
            end

            heuristic = zeros(n_rep, 1);
            optimum = zeros(n_rep, 1);
            if use_parallel
                parfor replication = 1:n_rep
                    [heuristic(replication), optimum(replication)] = ...
                        one_instance(model_index, n, b, ...
                        block_seed(replication), assortments);
                end
            else
                for replication = 1:n_rep
                    [heuristic(replication), optimum(replication)] = ...
                        one_instance(model_index, n, b, ...
                        block_seed(replication), assortments);
                end
            end

            ratio = heuristic ./ optimum;
            index = row + (1:n_rep);
            model_col(index) = model;
            n_col(index) = n;
            b_col(index) = b;
            rep_col(index) = (1:n_rep)';
            seed_col(index) = block_seed;
            heuristic_col(index) = heuristic;
            optimum_col(index) = optimum;
            ratio_col(index) = ratio;
            row = row + n_rep;

            fprintf('%s n=%d b=%d: mean %.8f min %.8f sd %.8f\n', ...
                model, n, b, mean(ratio), min(ratio), std(ratio));
        end
    end
end

trials = table(model_col, n_col, b_col, rep_col, seed_col, ...
    heuristic_col, optimum_col, ratio_col, ...
    'VariableNames', {'Model', 'N', 'B', 'Replication', 'Seed', ...
    'HeuristicRevenue', 'OptimalRevenue', 'Ratio'});
summary = aggregate_summary(trials, models, n_list, b_list);
validate_generation(trials, summary, models, n_list, b_list, ...
    n_rep, master_seed);

trial_name = 'synthetic_trial_level.csv';
summary_name = 'synthetic_summary.csv';
manifest_name = 'synthetic_producer_manifest.json';
trial_file = fullfile(run_dir, trial_name);
summary_file = fullfile(run_dir, summary_name);
manifest_file = fullfile(run_dir, manifest_name);

write_trial_csv(trial_file, trials);
write_summary_csv(summary_file, summary);

manifest = struct();
manifest.schema_version = 1;
manifest.workflow = 'canonical_synthetic_cmnl_scm';
manifest.artifact_type = 'matlab_staging_generation';
manifest.models = cellstr(models);
manifest.n_list = n_list;
manifest.b_list = b_list;
manifest.n_rep = n_rep;
manifest.master_seed = master_seed;
manifest.use_parallel = use_parallel;
manifest.rng = 'mt19937ar';
manifest.seed_formula = [ ...
    'master_seed + model_index*100000000 + N*1000000 + ' ...
    'B*10000 + replication'];
manifest.ratio_tolerance = 1e-10;
manifest.numeric_format = '%.17g';
manifest.trial_rows = height(trials);
manifest.summary_rows = height(summary);
manifest.matlab_version = version;
manifest.matlab_release = version('-release');
manifest.trial_file = trial_name;
manifest.summary_file = summary_name;
write_json(manifest_file, manifest);

fprintf('synthetic staging generation -> %s\n', run_dir);
end


function [n_rep, master_seed, use_parallel, n_list, b_list] = ...
        validate_design(n_rep, master_seed, use_parallel, n_list, b_list)
if ~isnumeric(n_rep) || ~isscalar(n_rep) || ~isfinite(n_rep) ...
        || n_rep ~= fix(n_rep) || n_rep < 1 || n_rep > 9999
    error('run_synthetic_experiments:InvalidReplicationCount', ...
        'n_rep must be an integer in [1, 9999].');
end
if ~isnumeric(master_seed) || ~isscalar(master_seed) ...
        || ~isfinite(master_seed) || master_seed ~= fix(master_seed) ...
        || master_seed < 0
    error('run_synthetic_experiments:InvalidMasterSeed', ...
        'master_seed must be a nonnegative integer.');
end
valid_parallel = (islogical(use_parallel) && isscalar(use_parallel)) ...
    || (isnumeric(use_parallel) && isscalar(use_parallel) ...
    && isfinite(use_parallel) && any(use_parallel == [0, 1]));
if ~valid_parallel
    error('run_synthetic_experiments:InvalidParallelFlag', ...
        'use_parallel must be a logical scalar.');
end

n_list = validate_axis(n_list, 'n_list');
b_list = validate_axis(b_list, 'b_list');
if max(n_list) > 20 || max(b_list) > 20 || max(b_list) > min(n_list)
    error('run_synthetic_experiments:InvalidGrid', ...
        ['Require unique integer axes no larger than 20 and every ' ...
        'crossed cardinality B to satisfy B <= N.  The exact restricted ' ...
        'subset cache is deliberately bounded at the production maximum.']);
end

n_rep = double(n_rep);
master_seed = double(master_seed);
use_parallel = logical(use_parallel);
largest_seed = uint64(master_seed) + uint64(2) * uint64(100000000) ...
    + uint64(max(n_list)) * uint64(1000000) ...
    + uint64(max(b_list)) * uint64(10000) + uint64(n_rep);
if largest_seed > uint64(intmax('uint32'))
    error('run_synthetic_experiments:SeedOverflow', ...
        'The task-key seed exceeds the uint32 RandStream seed range.');
end
end


function axis = validate_axis(axis, name)
if ~isnumeric(axis) || ~isvector(axis) || isempty(axis) ...
        || any(~isfinite(axis)) || any(axis ~= fix(axis)) || any(axis < 1)
    error('run_synthetic_experiments:InvalidGrid', ...
        '%s must be a nonempty positive-integer vector.', name);
end
axis = double(axis(:)');
if numel(unique(axis, 'stable')) ~= numel(axis)
    error('run_synthetic_experiments:InvalidGrid', ...
        '%s cannot contain duplicate values.', name);
end
end


function run_dir = resolve_run_directory(experiment_root, output_dir, supplied)
if ~supplied || isempty(output_dir)
    parent = fullfile(experiment_root, 'results', 'synthetic_runs');
    if ~exist(parent, 'dir'), mkdir(parent); end
    run_dir = tempname(parent);
    mkdir(run_dir);
else
    if ~(ischar(output_dir) || (isstring(output_dir) && isscalar(output_dir)))
        error('run_synthetic_experiments:InvalidOutputDirectory', ...
            'output_dir must be a text scalar.');
    end
    run_dir = char(string(output_dir));
    if ~exist(run_dir, 'dir'), mkdir(run_dir); end
end

reserved = {'synthetic_trial_level.csv', 'synthetic_summary.csv', ...
    'synthetic_producer_manifest.json'};
for index = 1:numel(reserved)
    if isfile(fullfile(run_dir, reserved{index}))
        error('run_synthetic_experiments:OutputNotEmpty', ...
            'Refusing to overwrite an existing staging artifact: %s', ...
            fullfile(run_dir, reserved{index}));
    end
end
end


function same = same_directory(first, second)
first_file = java.io.File(first);
second_file = java.io.File(second);
same = strcmp(char(first_file.getCanonicalPath()), ...
    char(second_file.getCanonicalPath()));
end


function seed = task_seed(master_seed, model_index, n, b, replication)
value = uint64(master_seed) + uint64(model_index) * uint64(100000000) ...
    + uint64(n) * uint64(1000000) + uint64(b) * uint64(10000) ...
    + uint64(replication);
if value > uint64(intmax('uint32'))
    error('run_synthetic_experiments:SeedOverflow', ...
        'A task-key seed exceeds the uint32 RandStream seed range.');
end
seed = double(value);
end


function cache = build_assortment_cache(n_list, b_list)
cache = cell(numel(n_list), numel(b_list));
for n_index = 1:numel(n_list)
    for b_index = 1:numel(b_list)
        cache{n_index, b_index} = synthetic_offer_cache( ...
            n_list(n_index), b_list(b_index));
    end
end
end


function [heuristic_revenue, optimal_revenue] = one_instance( ...
        model_index, n, b, seed, cache)
stream = RandStream('mt19937ar', 'Seed', seed);
attention = rand(stream, 1, n);
revenue = 10 * rand(stream, 1, n);
attraction = 10 * rand(stream, 1, n);
if model_index == 2
    % The manuscript draws the fixed SCM evaluation order uniformly.  A
    % common permutation of the iid product triples expresses that order in
    % positions 1,...,n while preserving the old fixed-order SCM formula.
    permutation = randperm(stream, n);
    attention = attention(permutation);
    revenue = revenue(permutation);
    attraction = attraction(permutation);
end

all_revenues = synthetic_zeta_revenues( ...
    model_index, attention, revenue, attraction, cache);
optimal_revenue = max(all_revenues(2:end));

q = attention .* attraction ./ (1 + attraction);
if model_index == 1
    [revenue_descending, index] = sort(revenue, 'descend');
    [first, ~] = rcs_optimal(revenue_descending, q(index), b);
    candidates = flip(index(first == 1));
    [second, ~] = rcs_optimal( ...
        revenue(candidates), q(candidates), numel(candidates));
    offer = false(1, n);
    offer(candidates(second == 1)) = true;
else
    [offer, ~] = rcs_optimal(revenue, q, b);
end
mask = uint32(double(offer) * double(cache.powers)');
offer_row = cache.mask_to_row(double(mask) + 1);
if offer_row == 0
    error('run_synthetic_experiments:HeuristicOutsideGrid', ...
        'The heuristic returned an offer outside the cached feasible grid.');
end
heuristic_revenue = all_revenues(offer_row);
end


function [offer, optimum] = rcs_optimal(revenue, q, b)
n = numel(revenue);
revenue = flip(revenue);
q = flip(q);
dp = zeros(n, b);
dp(1, :) = revenue(1) * q(1);
for item = 2:n
    for cardinality = 1:b
        if cardinality == 1
            dp(item, cardinality) = max( ...
                revenue(item) * q(item), dp(item-1, cardinality));
        else
            dp(item, cardinality) = max( ...
                (1-q(item)) * dp(item-1, cardinality-1) ...
                + q(item) * revenue(item), dp(item-1, cardinality));
        end
    end
end
offer = false(1, n);
item = n;
cardinality = b;
while item >= 1 && cardinality >= 1
    if item == 1
        offer(item) = true;
    elseif dp(item, cardinality) ~= dp(item-1, cardinality)
        offer(item) = true;
        cardinality = cardinality - 1;
    end
    item = item - 1;
end
offer = flip(offer);
optimum = dp(n, b);
end


function summary = aggregate_summary(trials, models, n_list, b_list)
n_row = numel(models) * numel(n_list) * numel(b_list);
model_col = strings(n_row, 1);
n_col = zeros(n_row, 1);
b_col = zeros(n_row, 1);
mean_col = zeros(n_row, 1);
min_col = zeros(n_row, 1);
sd_col = zeros(n_row, 1);
row = 0;
for model = models
    for n = n_list
        for b = b_list
            row = row + 1;
            use = trials.Model == model & trials.N == n & trials.B == b;
            ratio = trials.Ratio(use);
            model_col(row) = model;
            n_col(row) = n;
            b_col(row) = b;
            mean_col(row) = mean(ratio);
            min_col(row) = min(ratio);
            sd_col(row) = std(ratio);
        end
    end
end
summary = table(model_col, n_col, b_col, mean_col, min_col, sd_col, ...
    'VariableNames', {'Model', 'N', 'B', 'MeanRatio', 'MinRatio', 'StdRatio'});
end


function validate_generation(trials, summary, models, n_list, b_list, ...
        n_rep, master_seed)
expected_trial_rows = numel(models) * numel(n_list) * numel(b_list) * n_rep;
expected_summary_rows = numel(models) * numel(n_list) * numel(b_list);
if height(trials) ~= expected_trial_rows || height(summary) ~= expected_summary_rows
    error('run_synthetic_experiments:IncompleteGrid', ...
        'Synthetic trial or summary grid is incomplete.');
end
if any(~isfinite(trials.HeuristicRevenue)) ...
        || any(~isfinite(trials.OptimalRevenue)) ...
        || any(~isfinite(trials.Ratio)) ...
        || any(trials.HeuristicRevenue < 0) ...
        || any(trials.OptimalRevenue <= 0) ...
        || any(trials.Ratio < -1e-12) || any(trials.Ratio > 1 + 1e-10)
    error('run_synthetic_experiments:InvalidRatio', ...
        'Synthetic revenues and ratios must be finite and feasible.');
end
recomputed_ratio = trials.HeuristicRevenue ./ trials.OptimalRevenue;
if any(abs(recomputed_ratio - trials.Ratio) ...
        > 32 * eps(max(1, abs(recomputed_ratio))))
    error('run_synthetic_experiments:RatioMismatch', ...
        'Stored ratios do not match the underlying revenues.');
end
if numel(unique(trials.Seed)) ~= height(trials)
    error('run_synthetic_experiments:SeedCollision', ...
        'Task-key seeds are not unique on the requested grid.');
end

row = 0;
for model_index = 1:numel(models)
    for n = n_list
        for b = b_list
            for replication = 1:n_rep
                row = row + 1;
                expected_seed = task_seed( ...
                    master_seed, model_index, n, b, replication);
                if trials.Model(row) ~= models(model_index) ...
                        || trials.N(row) ~= n || trials.B(row) ~= b ...
                        || trials.Replication(row) ~= replication ...
                        || trials.Seed(row) ~= expected_seed
                    error('run_synthetic_experiments:TaskOrderMismatch', ...
                        'Synthetic trial rows are not in canonical task order.');
                end
            end
        end
    end
end

recomputed_summary = aggregate_summary(trials, models, n_list, b_list);
if ~isequal(summary(:, 1:3), recomputed_summary(:, 1:3)) ...
        || any(abs(summary{:, 4:6} - recomputed_summary{:, 4:6}) > 0, 'all') ...
        || any(~isfinite(summary{:, 4:6}), 'all')
    error('run_synthetic_experiments:SummaryMismatch', ...
        'Synthetic summary does not exactly aggregate the trial rows.');
end
end


function write_trial_csv(path, trials)
file_id = open_output(path);
cleanup = onCleanup(@() fclose(file_id));
fprintf(file_id, ['Model,N,B,Replication,Seed,HeuristicRevenue,' ...
    'OptimalRevenue,Ratio\n']);
for row = 1:height(trials)
    fprintf(file_id, '%s,%d,%d,%d,%.0f,%.17g,%.17g,%.17g\n', ...
        char(trials.Model(row)), trials.N(row), trials.B(row), ...
        trials.Replication(row), trials.Seed(row), ...
        trials.HeuristicRevenue(row), trials.OptimalRevenue(row), ...
        trials.Ratio(row));
end
end


function write_summary_csv(path, summary)
file_id = open_output(path);
cleanup = onCleanup(@() fclose(file_id));
fprintf(file_id, 'Model,N,B,MeanRatio,MinRatio,StdRatio\n');
for row = 1:height(summary)
    fprintf(file_id, '%s,%d,%d,%.17g,%.17g,%.17g\n', ...
        char(summary.Model(row)), summary.N(row), summary.B(row), ...
        summary.MeanRatio(row), summary.MinRatio(row), summary.StdRatio(row));
end
end


function write_json(path, value)
file_id = open_output(path);
cleanup = onCleanup(@() fclose(file_id));
fprintf(file_id, '%s\n', jsonencode(value));
end


function file_id = open_output(path)
file_id = fopen(path, 'wt');
if file_id < 0
    error('run_synthetic_experiments:OutputOpenFailed', ...
        'Cannot open output file: %s', path);
end
end
