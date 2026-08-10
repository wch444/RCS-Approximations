function run_mixed_order(SMOKE)
%% Canonical mixed-ordering counterfactual study (referee response)
% Full design: five Expedia sites x ten trial-specific data/fit pairs x
% 500 independently sampled test queries.  All fitted SCM parameters use the
% canonical [beta; alpha] order.

if nargin < 1 || isempty(SMOKE)
    SMOKE = false;
end
if ~isscalar(SMOKE) || ~(islogical(SMOKE) || isnumeric(SMOKE))
    error('run_mixed_order:InvalidSmokeFlag', 'SMOKE must be a scalar logical flag.');
end
SMOKE = logical(SMOKE);

here = fileparts(mfilename('fullpath'));
base = fileparts(fileparts(fileparts(here)));
addpath(fullfile(base,'common','matlab','models'),'-begin');
addpath(fullfile(base,'common','matlab','workflow'),'-begin');

feature_start = 8;
feature_end = 20;
sites = [5, 14, 15, 24, 32];
data_dir = fullfile(base,'data','processed','expedia');
fit_dir = fullfile(base,'opt_data','expedia');
require_expedia_grid(data_dir,sites,'data');
require_expedia_grid(fit_dir,sites,'fit');
if SMOKE
    trials = 1;
    queries_per_task = 30;
    output_stem = "mixed_order_results_smoke";
    run_mode = "smoke";
else
    trials = 1:10;
    queries_per_task = 500;
    output_stem = "mixed_order_results";
    run_mode = "full";
end

% Column indices in the truncated Expedia matrix.
POS = 4;
PRICE = 7;
STAR = 10;
LOC = 11;
ord_names = ["Default", "PriceLowHigh", "PriceHighLow", ...
    "Location", "StarRating"];
K = numel(ord_names);

sampling_seed_base = 20260809;
evaluation_seed_base = 120260809;
n_tasks = numel(sites) * numel(trials);
out_parts = cell(n_tasks, 1);
input_files = repmat(struct( ...
    'site', 0, 'trial', 0, 'data_path', '', 'fit_path', '', ...
    'total_queries', 0, ...
    'eligible_queries', 0, 'excluded_ineligible_queries', 0), n_tasks, 1);

t_all = tic;
task_index = 0;
for site_index = 1:numel(sites)
    site = sites(site_index);
    for trial_index = 1:numel(trials)
        trial = trials(trial_index);
        task_index = task_index + 1;

        data_name = sprintf('10k_sample_%d_%d.mat',site,trial);
        fit_name = sprintf('fit_%d_%d.mat',site,trial);
        data_relative = sprintf('data/processed/expedia/%s',data_name);
        fit_relative = sprintf('opt_data/expedia/%s',fit_name);
        data_file = fullfile(base,data_relative);
        fit_file = fullfile(base,fit_relative);
        require_file(data_file, 'trial-specific Expedia data');
        require_file(fit_file, 'trial-specific Expedia fit');

        D = load(data_file, 'data_test');
        P = load(fit_file,'fit');
        if ~isfield(D,'data_test') || ~isfield(P,'fit') ...
                || ~isstruct(P.fit) || ~isscalar(P.fit)
            error('run_mixed_order:InvalidInput', ...
                'Missing data_test or canonical fit.models.scm for site %d trial %d.', ...
                site, trial);
        end

        data_test = validate_data_test(D.data_test,feature_end,site,trial);
        models = P.fit.models;
        scm = models.scm;
        if ~isnumeric(scm) || ~isreal(scm) || any(~isfinite(scm)) ...
                || numel(scm) ~= 2 * (feature_end - feature_start + 1)
            error('run_mixed_order:InvalidSCMDimension', ...
                'Expected 26 SCM coefficients for site %d trial %d.', site, trial);
        end
        scm_beta = scm(1:13);
        scm_alpha = scm(14:26);

        test = data_test(:, 1:feature_end);
        [group, all_uids] = findgroups(test(:, 1));
        queries = splitapply(@(rows) {rows}, test, group);
        total_queries = numel(all_uids);
        valid_outside = cellfun(@has_trailing_outside,queries);
        if any(~valid_outside)
            error('run_mixed_order:InvalidOutsideOption', ...
                'Site %d trial %d has a query without one trailing outside row.', ...
                site,trial);
        end
        product_counts = cellfun(@(rows) size(rows,1)-1,queries);
        eligible = product_counts >= 2;
        eligible_queries = sum(eligible);
        excluded_ineligible_queries = total_queries-eligible_queries;
        all_uids = all_uids(eligible);
        queries = queries(eligible);
        if eligible_queries < queries_per_task
            error('run_mixed_order:InsufficientQueries', ...
                ['Site %d trial %d has %d eligible queries with at least two ' ...
                 'products; %d are required.'], ...
                site,trial,eligible_queries,queries_per_task);
        end

        sampling_seed = task_sampling_seed( ...
            sampling_seed_base, site, trial);
        sampling_stream = RandStream('mt19937ar', 'Seed', sampling_seed);
        selected = randperm(sampling_stream, numel(all_uids), queries_per_task);
        uids = all_uids(selected);
        queries = queries(selected);
        if numel(unique(uids)) ~= queries_per_task
            error('run_mixed_order:DuplicateQueryWithinTask', ...
                'Duplicate query selected for site %d trial %d.', site, trial);
        end

        t_task = tic;
        % Numeric output columns:
        % site, trial, srch_id, n_prod, card_SqL, UB,
        % sampling_seed, evaluation_seed, H(5), G(5), ALL(5).
        task_out = nan(queries_per_task, 23);
        parfor query_index = 1:queries_per_task
            evaluation_seed = task_evaluation_seed( ...
                evaluation_seed_base, site, trial, query_index);
            rng(evaluation_seed, 'twister');

            uid = uids(query_index);
            query = queries{query_index};
            query = query(1:end-1, :); % remove the outside-option row
            n_product = size(query, 1);
            if n_product < 2
                error('run_mixed_order:InvalidQuery', ...
                    'Site %d trial %d query %.0f has fewer than two products.', ...
                    site, trial, uid);
            end

            X = query(:, feature_start:end);
            attraction = exp(X * scm_beta);
            attention = 1 ./ (1 + exp(-(X * scm_alpha)));
            last_choice = attention .* attraction ./ (1 + attraction);
            revenue = query(:, PRICE);

            [~, upper_order] = sort(revenue, 'descend');
            upper_bound = Rc_cascade( ...
                revenue(upper_order), last_choice(upper_order));
            if ~isfinite(upper_bound) || upper_bound <= 0
                error('run_mixed_order:InvalidUpperBound', ...
                    'Invalid upper bound for site %d trial %d query %.0f.', ...
                    site, trial, uid);
            end

            [~, lower_order] = sort(revenue, 'ascend');
            [selected_assortment, ~] = assort_RCS( ...
                revenue(lower_order), last_choice(lower_order), n_product);
            selected_index = lower_order(selected_assortment == 1);
            selected_query = query(selected_index, :);
            selected_revenue = revenue(selected_index);

            ratio_H = zeros(1, K);
            ratio_G = zeros(1, K);
            ratio_ALL = zeros(1, K);
            order_sub = build_orderings( ...
                selected_query, POS, PRICE, STAR, LOC);
            order_full = build_orderings(query, POS, PRICE, STAR, LOC);
            scm_query = [scm_beta; scm_alpha];

            for order_index = 1:K
                probability_H = Seq_Click_single_rev_prob( ...
                    selected_query(order_sub{order_index}, :), ...
                    scm_query, feature_start);
                ratio_H(order_index) = sum( ...
                    selected_revenue(order_sub{order_index}) .* probability_H) ...
                    / upper_bound;

                probability_ALL = Seq_Click_single_rev_prob( ...
                    query(order_full{order_index}, :), scm_query, feature_start);
                ratio_ALL(order_index) = sum( ...
                    revenue(order_full{order_index}) .* probability_ALL) ...
                    / upper_bound;

                ordered_revenue = revenue(order_full{order_index});
                ordered_last_choice = last_choice(order_full{order_index});
                [selected_G, ~] = assort_RCS( ...
                    ordered_revenue, ordered_last_choice, n_product);
                selected_G_index = order_full{order_index}(selected_G == 1);
                probability_G = Seq_Click_single_rev_prob( ...
                    query(selected_G_index, :), scm_query, feature_start);
                ratio_G(order_index) = sum( ...
                    revenue(selected_G_index) .* probability_G) / upper_bound;
            end

            task_out(query_index, :) = [ ...
                site, trial, uid, n_product, numel(selected_index), upper_bound, ...
                sampling_seed, evaluation_seed, ratio_H, ratio_G, ratio_ALL];
        end

        if size(task_out, 1) ~= queries_per_task ...
                || any(~isfinite(task_out), 'all')
            error('run_mixed_order:IncompleteTask', ...
                'Site %d trial %d did not produce a complete finite task.', ...
                site, trial);
        end
        if numel(unique(task_out(:, 3))) ~= queries_per_task
            error('run_mixed_order:DuplicateQueryWithinTask', ...
                'Output contains a duplicate query for site %d trial %d.', ...
                site, trial);
        end

        out_parts{task_index} = task_out;
        input_files(task_index).site = site;
        input_files(task_index).trial = trial;
        input_files(task_index).data_path = data_relative;
        input_files(task_index).fit_path = fit_relative;
        input_files(task_index).total_queries = total_queries;
        input_files(task_index).eligible_queries = eligible_queries;
        input_files(task_index).excluded_ineligible_queries = ...
            excluded_ineligible_queries;
        fprintf('site %2d trial %2d: %4d queries in %6.1fs\n', ...
            site, trial, queries_per_task, toc(t_task));
    end
end

out = vertcat(out_parts{:});
expected_rows = n_tasks * queries_per_task;
if size(out, 1) ~= expected_rows
    error('run_mixed_order:IncompleteRun', ...
        'Expected %d rows but produced %d.', expected_rows, size(out, 1));
end

numeric_names = [ ...
    "site", "trial", "srch_id", "n_prod", "card_SqL", "UB", ...
    "sampling_seed", "evaluation_seed", ...
    "H_" + ord_names, "G_" + ord_names, "ALL_" + ord_names];
numeric_table = array2table(out, 'VariableNames', cellstr(numeric_names));
results = numeric_table;

task_key = compose('%d|%d|%.0f', ...
    results.site, results.trial, results.srch_id);
if numel(unique(task_key)) ~= height(results)
    error('run_mixed_order:DuplicateOutputKey', ...
        'The output key (site, trial, srch_id) is not unique.');
end

provenance = struct();
provenance.schema_version = 3;
provenance.artifact_type = 'mixed_order_counterfactual';
provenance.mode = char(run_mode);
provenance.sites = sites;
provenance.trials = trials;
provenance.queries_per_task = queries_per_task;
provenance.expected_rows = expected_rows;
provenance.actual_rows = height(results);
provenance.feature_start = feature_start;
provenance.feature_end = feature_end;
provenance.orderings = cellstr(ord_names);
provenance.csv_columns = results.Properties.VariableNames;
provenance.sampling_seed_base = sampling_seed_base;
provenance.sampling_seed_rule = ...
    'sampling_seed_base + 100000*site + 1000*trial';
provenance.evaluation_seed_base = evaluation_seed_base;
provenance.evaluation_seed_rule = ...
    'evaluation_seed_base + 100000*site + 1000*trial + query_index';
provenance.metric = ['Mean across queries of the query-level ratio ' ...
    'R(S|pi_k)/R_C(N|q,pi_U); it is not a ratio of pooled revenues.'];
provenance.input_files = input_files;

outdir = fullfile(fileparts(here), 'results');
if ~isfolder(outdir)
    [created,message] = mkdir(outdir);
    if ~created
        error('run_mixed_order:OutputDirectory', ...
            'Could not create output directory %s: %s',outdir,message);
    end
end
publish_mixed_order_pair(outdir, output_stem, results, out, ...
    ord_names, sites, trials, provenance);
fprintf('TOTAL %d tasks, %d queries in %.1fs\n', ...
    n_tasks, expected_rows, toc(t_all));
fprintf('saved atomically -> %s/%s.{csv,mat}\n', outdir, output_stem);
end


function seed = task_sampling_seed(base_seed, site, trial)
seed = base_seed + 100000 * site + 1000 * trial;
end


function seed = task_evaluation_seed(base_seed, site, trial, query_index)
seed = base_seed + 100000 * site + 1000 * trial + query_index;
end


function R = Rc_cascade(revenue, q)
R = 0;
survival = 1;
for product = 1:numel(revenue)
    R = R + survival * q(product) * revenue(product);
    survival = survival * (1 - q(product));
end
end


function orderings = build_orderings(rows, POS, PRICE, STAR, LOC)
[~, default_order] = sort(rows(:, POS), 'ascend');
[~, price_low_high] = sort(rows(:, PRICE), 'ascend');
[~, price_high_low] = sort(rows(:, PRICE), 'descend');
[~, location_order] = sort(rows(:, LOC), 'descend');
[~, star_order] = sort(rows(:, STAR), 'descend');
orderings = {default_order, price_low_high, price_high_low, ...
    location_order, star_order};
end


function [opt_S, opt_rev] = assort_RCS(revenue_input, q_input, capacity)
revenue = flip(revenue_input);
q = flip(q_input);
n_product = length(revenue);
dp = zeros(n_product, capacity);
dp(1, :) = revenue(1) * q(1);
for product = 2:n_product
    for cardinality = 1:capacity
        if cardinality == 1
            dp(product, cardinality) = max( ...
                revenue(product) * q(product), dp(product - 1, cardinality));
        else
            dp(product, cardinality) = max( ...
                (1 - q(product)) * dp(product - 1, cardinality - 1) ...
                + q(product) * revenue(product), ...
                dp(product - 1, cardinality));
        end
    end
end

product = n_product;
cardinality = capacity;
opt_S = zeros(1, n_product);
while product >= 1 && cardinality >= 1
    if product == 1
        opt_S(product) = 1;
    elseif dp(product, cardinality) ~= dp(product - 1, cardinality)
        opt_S(product) = 1;
        cardinality = cardinality - 1;
    end
    product = product - 1;
end
opt_S = flip(opt_S);
opt_rev = dp(n_product, capacity);
end


function valid = has_trailing_outside(rows)
valid = isnumeric(rows) && size(rows,1) >= 2 && size(rows,2) >= 8 ...
    && rows(end,2) == 0 && rows(end,4) == 100 && rows(end,7) == 0 ...
    && sum(rows(:,2) == 0 & rows(:,4) == 100 & rows(:,7) == 0) == 1;
end



function require_file(path, label)
if ~isfile(path)
    error('run_mixed_order:MissingFile', 'Missing %s: %s', label, path);
end
end


function require_expedia_grid(folder,sites,kind)
kind = validatestring(kind,{'data','fit'});
expected = strings(numel(sites)*10,1);
row = 0;
for site = sites
    for trial = 1:10
        row = row+1;
        if strcmp(kind,'fit')
            expected(row) = sprintf('fit_%d_%d.mat',site,trial);
        else
            expected(row) = sprintf('10k_sample_%d_%d.mat',site,trial);
        end
    end
end
entries = dir(fullfile(folder,'*.mat'));
names = string({entries.name}).';
if numel(names) ~= numel(expected) ...
        || ~isequal(sort(names),sort(expected))
    error('run_mixed_order:InvalidInputGrid', ...
        'Expected the fixed 5-site by 10-trial Expedia MAT grid in %s.',folder);
end
end


function data = validate_data_test(data,feature_end,site,trial)
if ~isnumeric(data) || ~isreal(data) || ~ismatrix(data) || isempty(data) ...
        || size(data,2) < feature_end || any(~isfinite(data),'all') ...
        || any(data(:,1) ~= fix(data(:,1))) || any(diff(data(:,1)) < 0)
    error('run_mixed_order:InvalidInput', ...
        'Invalid data_test matrix for site %d trial %d.',site,trial);
end
session_ids = data(:,1);
groups = findgroups(session_ids);
if any(~ismember(data(:,5),[0,1])) ...
        || any(splitapply(@sum,data(:,5),groups) ~= 1)
    error('run_mixed_order:InvalidInput', ...
        'Invalid choice indicators for site %d trial %d.',site,trial);
end
end
