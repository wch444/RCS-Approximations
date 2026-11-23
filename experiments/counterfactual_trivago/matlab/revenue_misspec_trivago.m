function summary = revenue_misspec_trivago(models, plats, trials, max_queries)
%% Robustness of the counterfactual analysis to the ground-truth model -- Trivago.
%
% Evaluate the six maintained Trivago models under assortment misspecification.
% referee concern: Section 6.2 scores every assortment with R(S_t|pi_t) under
% SCM, so the documented gains over the platform might rely on SCM being the
% true model. Here we REPEAT the Section 6.2 counterfactual on the SAME Trivago
% data and FITTED parameters, but take the ground truth to be a different choice
% model X in {MNL, MMNL, CMNL} (the competitors of Table~\ref{Tab:compare1}).
%
% Exactly as in Section 6.2, our heuristic needs only each product's last-choice
% (standalone) purchase probability q_i; under ground truth X this is
%       q_i = P_X( buy i | S = {i} ),
% computed from X's Trivago-fitted parameters. A singleton is displayed at
% relative position zero, so full-model q_i uses position bin 0 regardless of
% i's position in the recorded list. Given (r, q) we form the two
% heuristics of Section 6.2 with the maintained cascade dynamic program:
%   G-heuristic  S^G = argmax_{S subseteq S_t} R_C(S | q, pi_t)     (position order)
%   W-heuristic  S^W = argmax_{S subseteq S_t} R_C(S | q, pi_{t,L}) (ascending revenue)
% We then evaluate the TRUE expected revenue R_X(.) and compare each heuristic
% DIRECTLY to the platform's offer-all assortment S_t (no optimum / upper bound):
%   ratio_G = R_X(S^G) / R_X(S_t),   ratio_W = R_X(S^W) / R_X(S_t).
% ratio > 1 means the heuristic beats current platform practice under ground truth X.
% For either counterfactual subset, pi_t is restricted to the retained hotels:
% their original row order is preserved, the list is compacted to positions
% 0,...,|S|-1, and the five position indicators are rebuilt before R_X is
% evaluated. Consequently, fitted MNL and MMNL utilities are symmetric for a
% fixed feature matrix, but these position-augmented counterfactual
% specifications are not order independent when a hotel changes position.
%
% Trivago specifics:
%   * markets BR/US/JP/UK/DE replace the numeric Expedia site ids;
%   * processed matrices live under data/processed/trivago and fitted
%     parameters under opt_data/trivago;
%   * matrix columns are id, item, market, position, observed click-out (5),
%     interaction indicator (6), displayed price (7), and features (8:end);
%   * truncated models (SCM) use the 11 features cols 8:18 -> beta is 22x1
%     [attraction(1:11); attention(12:22)]; the SCM feature count is inferred as
%     numel(beta)/2 so no dimension is hard-coded;
%   * full models (MNL/CMNL/MMNL) use the 16 features cols 8:end: an
%     intercept, ten covariates, and five position indicators;
%   * every model is loaded from the versioned fresh-fit artifact.
%
% Output: experiments/counterfactual_trivago/results/counterfactual_trivago.csv,
% one aggregate row per model/market/trial. No query or session identifier is
% written. Each row
% records the evaluated and unique-query counts, aggregate revenues, and the
% equally weighted mean session-level revenue ratios.
%
% Usage:
%   revenue_misspec_trivago                                  % all models/markets/trials
%   revenue_misspec_trivago(["MNL","MMNL"])                  % cheap, exact models
%   revenue_misspec_trivago("CMNL",["US","JP"],1:3)          % CMNL, 2 markets, 3 trials
%   revenue_misspec_trivago("SCM",["US","JP"],1:3)

if nargin < 1 || isempty(models), models = ["SCM","MNL","MMNL","CMNL","SeqINV","NSMNL"]; end
if nargin < 2 || isempty(plats),  plats  = ["BR","US","JP","UK","DE"];  end
if nargin < 3 || isempty(trials), trials = 1:10; end
if nargin < 4 || isempty(max_queries), max_queries = Inf; end
canonical_models = ["SCM","MNL","MMNL","CMNL","SeqINV","NSMNL"];
canonical_plats = ["BR","US","JP","UK","DE"];
queries_per_task = 4000;
models = validate_string_axis(models,canonical_models, ...
    'revenue_misspec_trivago:InvalidModels', ...
    'revenue_misspec_trivago:DuplicateModels','models');
plats = validate_string_axis(plats,canonical_plats, ...
    'revenue_misspec_trivago:InvalidMarkets', ...
    'revenue_misspec_trivago:DuplicateMarkets','markets');
trials = validate_trials(trials);
max_queries = validate_max_queries(max_queries);

fea  = 8;     % features = columns 8:end (16 incl. 5 position bins)
v0   = 1;     % outside-option value
here = fileparts(mfilename('fullpath'));
base = fileparts(fileparts(fileparts(here)));       % canonical repository root
addpath(fullfile(base,'common','matlab','models'),'-begin');
addpath(fullfile(base,'common','matlab','workflow'),'-begin');
addpath(fullfile(base,'common','matlab','publishing'),'-begin');
data_dir = fullfile(base,'data','processed','trivago');
fit_dir = fullfile(base,'opt_data','trivago');
output_dir = fullfile(fileparts(here),'results');
validate_input_grid(data_dir,fit_dir,canonical_plats);

n_task = numel(models) * numel(plats) * numel(trials);
model_col = strings(n_task,1); market_col = strings(n_task,1);
trial_col = zeros(n_task,1); n_query_col = zeros(n_task,1);
n_unique_query_col = zeros(n_task,1);
sum_platform = zeros(n_task,1); sum_g = zeros(n_task,1); sum_w = zeros(n_task,1);
mean_g_ratio = zeros(n_task,1); mean_w_ratio = zeros(n_task,1);
task = 0;

for mi = 1:numel(models)
    X = models(mi);
    canonical_model_index = find(canonical_models == X,1);
    for pi = 1:numel(plats)
        p = plats(pi);
        canonical_market_index = find(canonical_plats == p,1);
        for l = trials
            data_file = fullfile(data_dir,sprintf('10k_sample_%s_%d_p2000.mat',p,l));
            fit_file = fullfile(fit_dir,sprintf('fit_%s_%d.mat',p,l));
            if ~isfile(data_file), error('Missing data: %s',data_file); end
            if ~isfile(fit_file), error('Missing fit: %s',fit_file); end
            D  = load(data_file,'data_test');
            if ~isfield(D,'data_test')
                error('revenue_misspec_trivago:InvalidTestData', ...
                    'Processed data must contain data_test: %s.',data_file);
            end
            te = D.data_test;
            th = load_params(X,fit_file,p,l,te); % ground-truth params

            [query_ids,queries] = validate_test_queries(te,queries_per_task,p,l);
            if isfinite(max_queries)
                selected = 1:min(max_queries,numel(queries));
                queries = queries(selected);
                query_ids = query_ids(selected);
            end
            n_unique_queries = numel(unique(query_ids));
            if n_unique_queries ~= numel(query_ids)
                error('revenue_misspec_trivago:DuplicateTestSession', ...
                    'Selected test query IDs are not unique for %s trial %d.',p,l);
            end
            R = nan(numel(queries), 3);
            parfor qi = 1:numel(queries)
                evaluation_seed = counterfactual_evaluation_seed( ...
                    20260808,canonical_model_index,canonical_market_index,l,qi);
                rng(evaluation_seed, 'twister');
                qd = queries{qi};
                qd = qd(1:end-1, :);              % drop trailing no-purchase row
                n  = size(qd,1);
                if n == 0, continue; end
                feats = qd(:, fea:end);          % n x 16
                r     = qd(:,7)';                 % revenues (1 x n)
                q     = standalone(X, th, feats);              % 1 x n

                Rplat = revenueX(X, th, true(1,n), r, feats, v0);
                if ~isfinite(Rplat) || Rplat <= 0
                    R(qi,:) = [Rplat,NaN,NaN];
                    continue
                end

                SG = run_heur(r, q, false);       % G: position (row) order
                SW = run_heur(r, q, true);        % W: ascending-revenue order
                RG = revenueX(X, th, SG, r, feats, v0);
                RW = revenueX(X, th, SW, r, feats, v0);

                R(qi,:) = [Rplat, RG, RW];         % raw revenues
            end
            validate_task_revenues(R,X,p,l);
            cnt = size(R, 1);
            expected_evaluated = min(queries_per_task,max_queries);
            if cnt ~= expected_evaluated || n_unique_queries ~= expected_evaluated
                error('revenue_misspec_trivago:InvalidQueryCount', ...
                    ['%s %s trial %d evaluated %d rows and %d unique query IDs; ' ...
                     'expected %d.'],X,p,l,cnt,n_unique_queries,expected_evaluated);
            end
            task = task + 1;
            model_col(task) = X; market_col(task) = p; trial_col(task) = l;
            n_query_col(task) = cnt;
            n_unique_query_col(task) = n_unique_queries;
            sum_platform(task) = sum(R(:,1)); sum_g(task) = sum(R(:,2)); sum_w(task) = sum(R(:,3));
            mean_g_ratio(task) = mean(R(:,2)./R(:,1));
            mean_w_ratio(task) = mean(R(:,3)./R(:,1));
            fprintf('%-4s %-2s trial %2d : G %.4f | W %.4f  (%d queries)\n', ...
                    X, p, l, mean(R(:,2)./R(:,1)), mean(R(:,3)./R(:,1)), cnt);
        end
    end
end
summary = table(model_col,market_col,trial_col,n_query_col,n_unique_query_col, ...
    sum_platform,sum_g,sum_w, ...
    mean_g_ratio,mean_w_ratio,'VariableNames',{'Model','Market','Trial','NQueries', ...
    'NUniqueQueries','PlatformRevenue','GRevenue','WRevenue','MeanGRatio', ...
    'MeanWRatio'});
validate_task_summary(summary,models,plats,trials,max_queries,queries_per_task);
write_publication_tables({summary}, ...
    {fullfile(output_dir,'counterfactual_trivago.csv')}, ...
    {models,plats,trials,max_queries}, ...
    {canonical_models,canonical_plats,1:10,Inf}, ...
    numel(canonical_models)*numel(canonical_plats)*10, ...
    'revenue_misspec_trivago:SubsetNotWritten', ...
    ['Subset results were returned in memory and did not replace the ' ...
     'canonical counterfactual CSV.'], ...
    @() validate_input_grid(data_dir,fit_dir,canonical_plats));
end


function [query_ids,queries] = validate_test_queries( ...
        test_data,expected_count,market,trial)
if ~isnumeric(test_data) || ~ismatrix(test_data) || isempty(test_data) ...
        || ~isreal(test_data) || size(test_data,2) ~= 23 ...
        || any(~isfinite(test_data),'all')
    error('revenue_misspec_trivago:InvalidTestData', ...
        'Test data for %s trial %d do not satisfy the 23-column matrix contract.', ...
        market,trial);
end
session_ids = test_data(:,1);
if any(session_ids ~= fix(session_ids)) || any(diff(session_ids) < 0)
    error('revenue_misspec_trivago:InvalidTestSession', ...
        'Test session IDs must be finite integers for %s trial %d.',market,trial);
end
is_query_start = [true; session_ids(2:end) ~= session_ids(1:end-1)];
query_ids = session_ids(is_query_start);
if numel(query_ids) ~= numel(unique(query_ids))
    error('revenue_misspec_trivago:DuplicateTestSession', ...
        ['A test session ID occurs in more than one query block for ' ...
         '%s trial %d.'],market,trial);
end
group = cumsum(is_query_start);
queries = splitapply(@(rows){rows},test_data,group);
query_ids = query_ids(:);
group_id_counts = splitapply(@(ids)numel(unique(ids)),session_ids,group);
if any(group_id_counts ~= 1) || numel(queries) ~= numel(query_ids)
    error('revenue_misspec_trivago:DuplicateTestSession', ...
        'Test query IDs are not unique after grouping for %s trial %d.',market,trial);
end
if any(~ismember(test_data(:,5),[0,1])) ...
        || any(splitapply(@sum,test_data(:,5),group) ~= 1) ...
        || any(~cellfun(@has_trailing_outside,queries))
    error('revenue_misspec_trivago:InvalidTestData', ...
        ['Test queries violate the choice or trailing-outside contract for ' ...
         '%s trial %d.'],market,trial);
end
if numel(queries) ~= expected_count
    error('revenue_misspec_trivago:InvalidQueryCount', ...
        'Test data for %s trial %d contain %d queries; expected %d.', ...
        market,trial,numel(queries),expected_count);
end
end


function valid = has_trailing_outside(rows)
valid = isnumeric(rows) && size(rows,1) >= 2 && size(rows,2) >= 7 ...
    && rows(end,2) == 0 && rows(end,4) == 100 && rows(end,7) == 0 ...
    && sum(rows(:,2) == 0 & rows(:,4) == 100 & rows(:,7) == 0) == 1;
end


function validate_input_grid(data_dir,fit_dir,markets)
expected_data = strings(numel(markets)*10,1);
expected_fit = strings(numel(markets)*10,1);
row = 0;
for market = string(markets)
    for trial = 1:10
        row = row+1;
        expected_data(row) = sprintf( ...
            '10k_sample_%s_%d_p2000.mat',char(market),trial);
        expected_fit(row) = sprintf('fit_%s_%d.mat',char(market),trial);
    end
end
validate_named_grid(data_dir,expected_data, ...
    '^10k_sample_[A-Z]{2}_\d+_p2000\.mat$','processed-data');
validate_named_grid(fit_dir,expected_fit, ...
    '^fit_[A-Z]{2}_\d+\.mat$','fit');
end


function validate_named_grid(folder,expected,pattern,label)
if ~isfolder(folder)
    error('revenue_misspec_trivago:InvalidInputGrid', ...
        'Missing Trivago %s directory: %s.',label,folder);
end
entries = dir(fullfile(folder,'*.mat'));
names = string({entries.name}).';
matches = ~cellfun(@isempty,regexp(cellstr(names),pattern,'once'));
observed = sort(names(matches));
if numel(observed) ~= numel(expected) ...
        || ~isequal(observed,sort(expected))
    error('revenue_misspec_trivago:InvalidInputGrid', ...
        'Expected the fixed 5-market by 10-trial Trivago %s grid.',label);
end
end


function validate_task_revenues(R,model,market,trial)
if ~isnumeric(R) || size(R,2) ~= 3
    error('revenue_misspec_trivago:InvalidRevenue', ...
        'Revenue output must have exactly three numeric columns.');
end
if any(~isfinite(R(:,1))) || any(R(:,1) <= 0)
    error('revenue_misspec_trivago:InvalidPlatformRevenue', ...
        'Platform revenue must be finite and positive for %s %s trial %d.', ...
        model,market,trial);
end
if any(~isfinite(R(:,2:3)),'all') || any(R(:,2:3) < 0,'all')
    error('revenue_misspec_trivago:InvalidCounterfactualRevenue', ...
        'G and W revenues must be finite and nonnegative for %s %s trial %d.', ...
        model,market,trial);
end
end


function validate_task_summary( ...
        summary,models,markets,trials,max_queries,queries_per_task)
expected_rows = numel(models)*numel(markets)*numel(trials);
keys = strcat(summary.Model,"|",summary.Market,"|",string(summary.Trial));
expected_queries = min(queries_per_task,max_queries);
numeric_values = summary{:,{'NQueries','NUniqueQueries','PlatformRevenue', ...
    'GRevenue','WRevenue','MeanGRatio','MeanWRatio'}};
if height(summary) ~= expected_rows || numel(unique(keys)) ~= expected_rows ...
        || any(summary.NQueries ~= expected_queries) ...
        || any(summary.NUniqueQueries ~= expected_queries) ...
        || any(~isfinite(numeric_values),'all') ...
        || any(summary.PlatformRevenue <= 0) ...
        || any(summary{:,{'GRevenue','WRevenue','MeanGRatio','MeanWRatio'}} < 0,'all')
    error('revenue_misspec_trivago:InvalidTaskSummary', ...
        'Counterfactual task summary failed completeness or metric checks.');
end
end


function values = validate_string_axis( ...
        values,canonical,invalid_id,duplicate_id,label)
values = string(values);
if isempty(values) || ~isvector(values) || any(ismissing(values)) ...
        || any(strlength(values) == 0)
    error(invalid_id,'%s must be a nonmissing vector.',label);
end
values = values(:).';
if numel(unique(values,'stable')) ~= numel(values)
    error(duplicate_id,'%s may be requested at most once per run.',label);
end
unknown = ~ismember(values,canonical);
if any(unknown)
    error(invalid_id,'Unknown %s: %s.',label,strjoin(values(unknown),', '));
end
end


function trials = validate_trials(trials)
if ~isnumeric(trials) || ~isvector(trials) || isempty(trials) ...
        || ~isreal(trials) || any(~isfinite(trials)) ...
        || any(trials ~= fix(trials)) || any(trials < 1 | trials > 10) ...
        || numel(unique(trials)) ~= numel(trials)
    error('revenue_misspec_trivago:InvalidTrials', ...
        'Trials must be unique integers from 1 through 10.');
end
trials = double(trials(:).');
end


function max_queries = validate_max_queries(max_queries)
valid = isnumeric(max_queries) && isreal(max_queries) ...
    && isscalar(max_queries) && ~isnan(max_queries) ...
    && ((isinf(max_queries) && max_queries > 0) ...
        || (isfinite(max_queries) && max_queries >= 1 ...
            && max_queries == fix(max_queries)));
if ~valid
    error('revenue_misspec_trivago:InvalidMaxQueries', ...
        'max_queries must be a positive integer or positive Inf.');
end
max_queries = double(max_queries);
end


% ======================================================================= %
%   parameters per ground-truth model                                     %
% ======================================================================= %
function th = load_params(X,fit_file,expected_market,expected_trial,data_test)
    S = load(fit_file,'fit');
    if ~isfield(S,'fit') || ~isstruct(S.fit) || ~isscalar(S.fit)
        error('revenue_misspec_trivago:InvalidFit', ...
            'Fit MAT must contain one scalar fit structure: %s.',fit_file);
    end
    models = validate_unified_fit( ...
        S.fit,'trivago',expected_market,expected_trial,[],data_test);
    switch X
        case "SCM"   % reference: reproduces the Section 6.2 result (validates pipeline)
            th.beta = models.scm;
        case "MNL"
            th.beta = models.mnl;
        case "MMNL"
            th.beta  = models.mmnl_beta;
            th.alpha = models.mmnl_alpha(:)';
        case "CMNL"
            cb = models.cmnl;  nf = numel(cb)/2;
            th.beta_v = cb(1:nf);                         % attraction
            th.beta_a = cb(nf+1:2*nf);                    % attention
        case "SeqINV"   % paginated sequential MNL (Seq_INV_function.m), truncated features
            th.beta = models.seqinv;
        case "NSMNL"    % stop-page-mixture MNL (NS_MNL_function.m), truncated features
            th.beta   = models.nsmnl;
            th.lambda = models.nsmnl_lambda(:)';
    end
end


% ======================================================================= %
%   standalone purchase probability  q_i = P_X(buy i | {i})               %
% ======================================================================= %
function q = standalone(X, th, feats)
    % Each row represents a different singleton experiment, so every hotel
    % occupies relative display position zero in its own singleton assortment.
    singleton_features = reindex_trivago_position_features( ...
        feats,zeros(size(feats,1),1));
    switch X
        case "SCM"                           % q_i = alpha_i * v_i/(1+v_i)
            nf = numel(th.beta)/2;           % 11 truncated features
            f  = singleton_features(:,1:nf);
            v  = exp(f*th.beta(1:nf));
            a  = exp(f*th.beta(nf+1:2*nf)); al = a./(1+a);
            q  = (al .* (v./(1+v)))';
        case "MNL"
            v = exp(singleton_features*th.beta); q = (v./(1+v))';
        case "MMNL"
            V = exp(singleton_features*th.beta); % n x m
            qk = V./(1+V);
            q = (qk*th.alpha')';             % 1 x n
        case "CMNL"
            v = exp(singleton_features*th.beta_v);
            a = exp(singleton_features*th.beta_a); al = a./(1+a);
            q = (al .* (v./(1+v)))';
        case {"SeqINV","NSMNL"}   % single-item: both collapse to MNL standalone v_i/(1+v_i)
            nf = numel(th.beta);             % 11 truncated features
            s  = max(min(singleton_features(:,1:nf)*th.beta, 50), -50);
            v  = exp(s);
            q  = (v./(1+v))';
    end
end


% ======================================================================= %
%   true expected revenue R_X(S)                                          %
% ======================================================================= %
function R = revenueX(X, th, S, r, feats, v0)
    S = logical(S);
    if ~any(S), R = 0; return; end
    rr = r(S);
    % Restrict pi_t to the retained rows, compact their relative positions,
    % and rebuild the five full-model position indicators. Logical indexing
    % preserves the recorded row order for both G and W assortments.
    offered_features = reindex_trivago_position_features( ...
        feats(S,:),(0:sum(S)-1)');
    switch X
        case "SCM"   % order-dependent cascade revenue (Seq_Click), position order
            nf = numel(th.beta)/2;
            p = Seq_Click_single_rev_prob(offered_features(:,1:nf), th.beta, 1);
            R = rr * p(:);
        case "MNL"
            v = exp(offered_features*th.beta)';
            R = sum(rr.*v)/(1+sum(v));
        case "MMNL"
            R = 0;
            for k = 1:numel(th.alpha)
                vk = exp(offered_features*th.beta(:,k))';
                R  = R + th.alpha(k)*sum(rr.*vk)/(1+sum(vk));
            end
        case "CMNL"
            v  = exp(offered_features*th.beta_v)';
            a  = exp(offered_features*th.beta_a)'; al = a./(1+a);
            p  = cmnl_prob(al, v, v0);
            R  = sum(rr.*p);
        case "SeqINV"   % offered set re-paginated (pos. order) into pages of k=10
            R = seqinv_revenue( ...
                offered_features(:,1:numel(th.beta)),rr,th.beta,10);
        case "NSMNL"    % stop-page mixture over lambda, MNL within viewed prefix
            R = nsmnl_revenue( ...
                offered_features(:,1:numel(th.beta)), ...
                rr,th.beta,th.lambda,10);
    end
end


function seed = counterfactual_evaluation_seed( ...
        base_seed,model_index,market_index,trial,query_index)
% Stable, collision-free seed over the full 6 x 5 x 10 x 4000 task grid.
% Strides exceed the complete range of every less-significant axis, and the
% canonical model/market indices make subset and reordered runs identical.
seed = base_seed + 10000000*model_index + 1000000*market_index ...
    + 10000*trial + query_index;
if seed < 0 || seed > double(intmax('uint32')) || seed ~= fix(seed)
    error('revenue_misspec_trivago:InvalidEvaluationSeed', ...
        'Counterfactual evaluation seed is outside the uint32 range.');
end
end


% ======================================================================= %
%   SeqINV true revenue: paginated sequential MNL (cf. Seq_INV_choice)     %
% ======================================================================= %
function R = seqinv_revenue(F, r, beta, k)
% Customer scans page by page (page size k); on each page makes an MNL choice
% among that page's items, otherwise continues. Verbatim weighting of
% Seq_INV_choice in the canonical Seq_INV_function.m, applied to the OFFERED set
% in its position order (k must match the public fit configuration).
    n = size(F,1);
    w = exp(max(min(F*beta,50),-50));            % n x 1 attraction weights
    pg = ceil((1:n)'/k);  np = pg(end);
    pmass = accumarray(pg, w, [np,1]);           % per-page weight mass
    prefix = ones(np,1);                         % P(no purchase before this page)
    if np > 1, prefix(2:end) = cumprod(1./(1+pmass(1:end-1))); end
    prob = prefix(pg) .* w ./ (1+pmass(pg));     % per-product purchase prob
    R = r * prob;                                % r is 1 x n
end


% ======================================================================= %
%   NSMNL true revenue: stop-page mixture of prefix-MNL (cf. NS_MNL_choice)%
% ======================================================================= %
function R = nsmnl_revenue(F, r, beta, lambda, k)
% A customer who stops at page g (prob lambda(g)) views the first min(g*k,n)
% products and makes a single MNL choice among them. We marginalize over lambda
% (the population stop-page distribution from NS_MNL_function) rather than use a
% per-query click-based stop page, since clicks do not transfer to a
% counterfactual assortment -- this keeps R_X assortment-only, as for the other
% ground-truth models. k must match the public fit configuration.
    n = size(F,1);
    w = exp(max(min(F*beta,50),-50));            % n x 1
    R = 0;
    for g = 1:numel(lambda)
        m = min(g*k, n);                         % products viewed if stop at page g
        R = R + lambda(g) * ( r(1:m) * w(1:m) ) / (1 + sum(w(1:m)));
    end
end


% ======================================================================= %
%   heuristics: cascade DP under an evaluation order                      %
% ======================================================================= %
function S = run_heur(r, q, ascending)
    n = length(r);
    if ascending
        [~, I] = sort(r, 'ascend');           % pi_{t,L} = W-heuristic
        [Sind, ~] = rcs_optimal_assortment(r(I), q(I), n);
        S = false(1,n); S(I(Sind==1)) = true;
    else
        [Sind, ~] = rcs_optimal_assortment(r, q, n); % position order = G-heuristic
        S = Sind == 1;
    end
end


% ======================================================================= %
%   CMNL purchase probabilities  P_CMNL(i,S)  (exact small / sampled big) %
% ======================================================================= %
function p = cmnl_prob(al, v, v0)
    al = al(:)'; v = v(:)'; m = numel(v);
    if m <= 12
        M = double(dec2bin(0:2^m-1) - '0');          % 2^m x m
        probpat = prod(M.*al + (1-M).*(1-al), 2);    % P(T)
        wsum    = v0 + M*v';
        p = v .* sum(M .* (probpat./wsum), 1);       % 1 x m
    else
        ns = 2000;
        B  = rand(ns,m) < al;                        % consideration draws
        den = v0 + B*v';
        p  = mean(B .* (v ./ den), 1);
    end
end
