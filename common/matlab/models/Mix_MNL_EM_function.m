function [alpha, beta, LL, score, error, error_1, mape, t1, diagnostics] = Mix_MNL_EM_function(alpha, beta, fea, train_data, test_data,options, test, em_options)
    %MIX_MNL_EM_FUNCTION Fit/evaluate finite-mixture MNL by session-level EM/GEM.
    %   ALPHA is weights, BETA is feature-by-component, and TEST selects evaluation.
    %   Outputs include predictive metrics, runtime, and diagnostics.
    tic;
    if nargin < 8
        em_options = [];
    end
    em_config = validate_em_options(em_options);
    validate_choice_encoding(test_data, 'test_data', fea);
    [alpha, beta] = validate_parameters(alpha, beta, fea, test_data);

    if test == 0
        validate_choice_encoding(train_data, 'train_data', fea);
        if size(train_data,2) ~= size(test_data,2)
            builtin('error','Mix_MNL_EM_function:InconsistentData', ...
                'train_data and test_data must have the same columns.');
        end
        [alpha,beta,diagnostics] = EM_MMNL_LL( ...
            alpha,beta,train_data,fea,options,em_config);
    else
        diagnostics = struct( ...
            'estimator_version', 2, ...
            'algorithm', 'chosen_choice_generalized_em', ...
            'fitted', false, ...
            'max_iterations', em_config.MaxIterations, ...
            'relative_tolerance', em_config.RelativeTolerance, ...
            'required_consecutive', 3, ...
            'consecutive', 0, ...
            'estimation_complete', false, ...
            'hit_iteration_cap', false, ...
            'em_options', em_config);
    end

    LL = MMNL_LL(alpha,beta, fea, test_data);
    [score] = mixed_MNL_hit(beta,alpha,test_data,fea);
    [error, error_1, mape] = MMNL_pred(alpha,beta,test_data,fea);
    t1 = toc;
end


function [alpha, beta, diagnostics] = EM_MMNL_LL( ...
        alpha,beta,data,fea,options,em_config)
    max_em_iterations = em_config.MaxIterations;
    relative_tolerance = em_config.RelativeTolerance;
    required_consecutive = 3;
    consecutive = 0;
    group = findgroups(data(:,1));
    current_nll = LL_mixed_mmnl(beta,alpha,data,fea,group);
    nll_history = nan(max_em_iterations + 1,1);
    nll_history(1) = current_nll;
    mstep_exitflag = nan(max_em_iterations,1);
    mstep_beta_accepted = false(max_em_iterations,1);
    relative_improvement_history = nan(max_em_iterations,1);
    iterations_accepted = 0;
    converged = false;
    termination_reason = 'maximum_iterations';

    if isempty(options)
        options = optimoptions('fminunc', 'Display', 'off');
    end
    mstep_options = optimoptions( ...
        options,'SpecifyObjectiveGradient',true);

    for iteration = 1:max_em_iterations
        log_prob = component_log_probabilities(beta,data,fea,group);
        chosen = data(:,5) == 1;

        % Each session is one observation.  Its latent-class posterior is
        % therefore conditioned only on that session's observed choice.
        responsibility = posterior_responsibility(log_prob(chosen,:),alpha);
        alpha_candidate = mean(responsibility,1);
        alpha_candidate = alpha_candidate / sum(alpha_candidate);

        q_before = LL_weight_mmnl(beta,responsibility,data,fea,group);
        [beta_candidate, q_after, exit_flag] = fminunc( ...
            @(candidate)LL_weight_mmnl( ...
                candidate,responsibility,data,fea,group), ...
            beta,mstep_options);
        mstep_exitflag(iteration) = exit_flag;

        % fminunc may stop before finding an improving M-step.  Retaining
        % the old beta still gives a valid generalized-EM update through
        % alpha, whereas accepting a larger Q would forfeit monotonicity.
        q_tolerance = 1e-10 * (1 + abs(q_before));
        if isempty(exit_flag) || any(~isfinite(beta_candidate(:))) ...
                || ~isfinite(q_after) ...
                || q_after > q_before + q_tolerance
            beta_candidate = beta;
        else
            mstep_beta_accepted(iteration) = true;
        end

        candidate_nll = LL_mixed_mmnl( ...
            beta_candidate,alpha_candidate,data,fea,group);
        improvement = current_nll - candidate_nll;
        relative_improvement = improvement/max(1,abs(current_nll));
        monotonic_tolerance = 1e-10 * (1 + abs(current_nll));

        if improvement < -monotonic_tolerance
            error('Mix_MNL_EM_function:NonMonotoneEM', ...
                ['The EM candidate increased the observed-data NLL from ' ...
                 '%.16g to %.16g.'], current_nll, candidate_nll);
        elseif improvement < 0
            % Ignore a roundoff-sized increase; the previous iterate is the
            % monotone solution returned to the caller.
            termination_reason = 'roundoff_guard';
            break
        end

        % Accept every genuine improvement before applying the stopping
        % tolerance.  In particular, do not roll back a small improvement.
        alpha = alpha_candidate;
        beta = beta_candidate;
        current_nll = candidate_nll;
        iterations_accepted = iterations_accepted + 1;
        nll_history(iterations_accepted + 1) = current_nll;
        relative_improvement_history(iterations_accepted) = ...
            relative_improvement;

        if relative_improvement <= relative_tolerance
            consecutive = consecutive + 1;
        else
            consecutive = 0;
        end
        if consecutive >= required_consecutive
            converged = true;
            termination_reason = 'consecutive_relative_nll_tolerance';
            break
        end
    end

    assert_valid_simplex(alpha);
    if any(~isfinite(beta(:)))
        error('Mix_MNL_EM_function:NonFiniteParameter', ...
            'The fitted MMNL beta contains a nonfinite value.');
    end

    % Latent classes are observationally invariant to permutations.  Sort
    % the jointly paired (alpha,beta) components once, after EM, to make
    % persisted parameters reproducible without changing any probability.
    component_key = [-alpha(:), beta'];
    [~, component_order] = sortrows( ...
        component_key,1:size(component_key,2));
    alpha = alpha(component_order);
    beta = beta(:,component_order);

    hit_iteration_cap = strcmp( ...
        termination_reason,'maximum_iterations') ...
        && iteration == max_em_iterations;
    estimation_complete = converged || hit_iteration_cap;
    diagnostics = struct();
    diagnostics.estimator_version = 2;
    diagnostics.algorithm = 'chosen_choice_generalized_em';
    diagnostics.fitted = true;
    diagnostics.converged = converged;
    diagnostics.termination_reason = termination_reason;
    diagnostics.iterations_attempted = iteration;
    diagnostics.iterations_accepted = iterations_accepted;
    diagnostics.max_iterations = max_em_iterations;
    diagnostics.relative_tolerance = relative_tolerance;
    diagnostics.required_consecutive = required_consecutive;
    diagnostics.consecutive = consecutive;
    diagnostics.estimation_complete = estimation_complete;
    diagnostics.hit_iteration_cap = hit_iteration_cap;
    diagnostics.em_options = em_config;
    diagnostics.initial_train_nll = nll_history(1);
    diagnostics.final_train_nll = current_nll;
    diagnostics.train_nll_history = nll_history(1:iterations_accepted + 1);
    diagnostics.relative_nll_improvement_history = ...
        relative_improvement_history(1:iterations_accepted);
    diagnostics.mstep_exitflag = mstep_exitflag(1:iteration);
    diagnostics.mstep_beta_accepted = mstep_beta_accepted(1:iteration);
    diagnostics.component_order = component_order(:).';
end


function config = validate_em_options(em_options)
    config = struct( ...
        'MaxIterations', 100, ...
        'RelativeTolerance', 1e-7);
    if isempty(em_options)
        return
    end
    if ~isstruct(em_options) || ~isscalar(em_options)
        error('Mix_MNL_EM_function:InvalidEMOptions', ...
            'em_options must be an empty value or a scalar struct.');
    end

    allowed = {'MaxIterations';'RelativeTolerance'};
    supplied = fieldnames(em_options);
    unknown = setdiff(supplied,allowed);
    if ~isempty(unknown)
        error('Mix_MNL_EM_function:InvalidEMOptions', ...
            'Unknown em_options field(s): %s.',strjoin(unknown,', '));
    end

    for field_index = 1:numel(supplied)
        field = supplied{field_index};
        config.(field) = em_options.(field);
    end

    max_iterations = config.MaxIterations;
    if ~isnumeric(max_iterations) || ~isscalar(max_iterations) ...
            || ~isreal(max_iterations) || ~isfinite(max_iterations) ...
            || max_iterations < 1 || max_iterations > 100 ...
            || max_iterations ~= fix(max_iterations)
        error('Mix_MNL_EM_function:InvalidEMOptions', ...
            ['em_options.MaxIterations must be a finite integer ' ...
             'between 1 and 100.']);
    end
    relative_tolerance = config.RelativeTolerance;
    if ~isnumeric(relative_tolerance) || ~isscalar(relative_tolerance) ...
            || ~isreal(relative_tolerance) ...
            || ~isfinite(relative_tolerance) || relative_tolerance <= 0
        error('Mix_MNL_EM_function:InvalidEMOptions', ...
            'em_options.RelativeTolerance must be finite and positive.');
    end
end


function responsibility = posterior_responsibility(chosen_log_prob,alpha)
    log_joint = chosen_log_prob + log(alpha);
    row_max = max(log_joint,[],2);
    log_normalizer = row_max + log(sum(exp(log_joint-row_max),2));
    responsibility = exp(log_joint-log_normalizer);

    if any(~isfinite(responsibility(:)))
        error('Mix_MNL_EM_function:NonFinitePosterior', ...
            'MMNL latent-class posteriors must be finite.');
    end
end


function [LL, gradient] = LL_weight_mmnl( ...
        beta,responsibility,data,fea,group)
    log_prob = component_log_probabilities(beta,data,fea,group);
    chosen = data(:,5)==1;
    chosen_log_prob = log_prob(chosen,:);
    active = responsibility > 0;

    if any(~isfinite(chosen_log_prob(active)))
        LL = Inf;
        gradient = zeros(size(beta));
        return
    end
    LL = -sum(responsibility(active).*chosen_log_prob(active));

    if nargout > 1
        % For component k, the gradient of the minimized negative Q is
        %   d(-Q)/dbeta_k = sum_t tau_tk (E_pk[x|S_t] - x_t,chosen).
        % Map the one-posterior-per-session matrix back to the original row
        % order before forming all component gradients in one matrix product.
        n_session = max(group);
        n_component = size(beta,2);
        responsibility_by_session = zeros(n_session,n_component);
        responsibility_by_session(group(chosen),:) = responsibility;
        row_responsibility = responsibility_by_session(group,:);
        component_prob = exp(log_prob);
        feature = data(:,fea:end);
        gradient = feature'*(component_prob.*row_responsibility) ...
            - feature(chosen,:)'*responsibility;

        if any(~isfinite(gradient(:)))
            LL = Inf;
            gradient = zeros(size(beta));
        end
    end
end


function LL = LL_mixed_mmnl(beta,alpha,data,fea,group)
    if nargin < 5
        group = findgroups(data(:,1));
    end
    log_prob = component_log_probabilities(beta,data,fea,group);
    chosen_log_prob = log_prob(data(:,5)==1,:);
    log_joint = chosen_log_prob + log(alpha);
    row_max = max(log_joint,[],2);
    log_likelihood = row_max + log(sum(exp(log_joint-row_max),2));
    LL = -sum(log_likelihood);

    if ~isfinite(LL)
        error('Mix_MNL_EM_function:NonFiniteLikelihood', ...
            'The MMNL observed-data NLL is not finite.');
    end
end


function LL = MMNL_LL(alpha,beta, fea, data)
    LL = LL_mixed_mmnl(beta,alpha,data,fea);
end

function [prob] = MMNL_probability(alpha, beta, fea, data)
    % beta: d x m
    % alpha: 1 x m
    log_prob = component_log_probabilities(beta,data,fea);
    prob = exp(log_prob)*alpha';

    if any(~isfinite(prob(:)))
        error('Mix_MNL_EM_function:NonFiniteProbability', ...
            'MMNL choice probabilities must be finite.');
    end

end


function log_prob = component_log_probabilities(beta,data,fea,group)
    if nargin < 4
        group = findgroups(data(:,1));
    end
    score = data(:,fea:end)*beta;
    if any(~isfinite(score(:)))
        log_prob = -Inf(size(score));
        return
    end

    % Two grouped reductions give a stable log-softmax in O(NK), while
    % indexing by group maps the results back to the original row order.
    group_max = splitapply(@(value)max(value,[],1),score,group);
    shifted = score-group_max(group,:);
    group_sum_exp = splitapply(@(value)sum(value,1),exp(shifted),group);
    log_prob = shifted-log(group_sum_exp(group,:));

end


function [alpha, beta] = validate_parameters(alpha,beta,fea,data)
    if ~isnumeric(alpha) || isempty(alpha) || any(~isfinite(alpha(:))) ...
            || any(alpha(:) < 0)
        error('Mix_MNL_EM_function:InvalidAlpha', ...
            'MMNL alpha must contain finite nonnegative mixture weights.');
    end
    alpha = reshape(alpha,1,[]);
    alpha_sum = sum(alpha);
    if abs(alpha_sum-1) > 1e-8
        error('Mix_MNL_EM_function:InvalidAlpha', ...
            'MMNL alpha must sum to one.');
    end
    alpha = alpha/alpha_sum;

    n_feature = size(data,2)-fea+1;
    if ~isnumeric(beta) || any(~isfinite(beta(:))) ...
            || size(beta,1) ~= n_feature || size(beta,2) ~= numel(alpha)
        error('Mix_MNL_EM_function:InvalidBeta', ...
            ['MMNL beta must be a finite feature-by-component matrix ' ...
             'consistent with alpha and feature_start.']);
    end
end


function validate_choice_encoding(data,label,fea)
    if ~isnumeric(data) || isempty(data) || size(data,2) < 5 ...
            || any(~isfinite(data(:,1))) || any(~isfinite(data(:,5))) ...
            || any(~ismember(data(:,5),[0,1]))
        error('Mix_MNL_EM_function:InvalidChoiceEncoding', ...
            '%s must be nonempty and use finite binary choice indicators.',label);
    end
    if ~isscalar(fea) || ~isfinite(fea) || fea ~= fix(fea) ...
            || fea < 1 || fea > size(data,2) ...
            || any(~isfinite(data(:,fea:end)),'all')
        error('Mix_MNL_EM_function:InvalidDesignMatrix', ...
            '%s must contain finite features from feature_start onward.',label);
    end

    group = findgroups(data(:,1));
    chosen_per_session = accumarray(group,data(:,5),[],@sum);
    if any(chosen_per_session ~= 1)
        error('Mix_MNL_EM_function:InvalidChoiceEncoding', ...
            '%s must contain exactly one chosen row per session.',label);
    end
end


function assert_valid_simplex(alpha)
    if any(~isfinite(alpha)) || any(alpha < 0) ...
            || abs(sum(alpha)-1) > 1e-10
        error('Mix_MNL_EM_function:InvalidAlpha', ...
            'The fitted MMNL alpha is not a finite probability simplex.');
    end

end



function [score] = mixed_MNL_hit(beta,alpha,data,fea)

user_Id = unique(data(:,1));
all_user = length(user_Id);
hit = zeros(3,1);
cal_user = 0;
for index = 1:all_user

    i = user_Id(index);
    query_data = data(data(:,1)==i,:);
    [~,sort_index] = sort(query_data(:,4));
    query_data = query_data(sort_index,:);
    prob =  MMNL_probability(alpha, beta, fea, query_data);
    query_data = query_data(1:end-1,:); %exclude no-purchhase
    prob = prob(1:end-1,:); % exclude no-purchase

    if sum(query_data(:,5))>0
        [purchase_prod_idx,~] = find(query_data(:,5) == 1);
        [~,sort_index] = sort(-prob);
        hit = hit + identify_hit(purchase_prod_idx,sort_index,3);
        cal_user = cal_user + 1;
    end
end
score = hit/cal_user;
end

function [error, error_1, mape] = MMNL_pred(alpha,beta,data,fea)

    user_Id = unique(data(:,1));
    all_user = length(user_Id);
    book = 5;
    sum_prod = 0;
    error = 0;
    error_1 = 0;
    mape = 0;
    for index = 1:all_user
        i = user_Id(index);
        query_data = data(data(:,1)==i,:);
        n = size(query_data,1);
        prob = MMNL_probability(alpha, beta, fea, query_data);
        error = error + sum((prob - query_data(:,book)).^2);
        error_1 = error_1 + sum((prob - query_data(:,book)).^2./(0.5+query_data(:,book)));
        mape = mape + sum(abs(prob - query_data(:,book))./(query_data(:,book) + 0.5));
        sum_prod = sum_prod + n;
    end
    error = sqrt(error/sum_prod);
    error_1 = error_1/sum_prod;
    mape = mape/sum_prod;
end
