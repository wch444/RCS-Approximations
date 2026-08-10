function [beta, lambda, LL, score, rmse, error_1, mape, tt] = NS_MNL_function(beta, k, fea, train_data, test_data, options, test, lambda_input)
%NS_MNL_FUNCTION Fit or evaluate the stopping-page-mixture NSMNL model.
%   K is page size; LAMBDA_INPUT optionally supplies the stopping-page
%   distribution.
tic

if nargin < 8
    lambda_input = [];
end

if isempty(lambda_input)
    lambda = NS_MNL_lambda(train_data, k);
else
    lambda = lambda_input(:);
    if any(lambda < 0) || sum(lambda) <= 0
        error('NS_MNL_function:InvalidLambda', ...
            'The stopping-page probabilities must be nonnegative and have a positive sum.');
    end
    lambda = lambda / sum(lambda);
end
if test == 0
    beta = fminunc(@(b)NS_MNL_LL(train_data, k, b, fea, lambda), beta, options);
end

LL = NS_MNL_LL(test_data, k, beta, fea, lambda);
[rmse, error_1, mape, score] = NS_MNL_pred(beta, k, test_data, fea, lambda);
tt = toc;
end


function LL = NS_MNL_LL(data, k, beta, fea, lambda)
g = findgroups(data(:, 1));
nll = splitapply(@(query_data)NS_MNL_single_nll(query_data, k, beta, fea, lambda), data, g);
LL = sum(nll);
end


function nll = NS_MNL_single_nll(query_data, k, beta, fea, lambda)
book = 5;
[choice_prob, stop_page] = NS_MNL_choice(query_data, k, beta, fea);

stop_page = min(stop_page, length(lambda));
prod_data = query_data(1:end-1, :);
book_idx = find(prod_data(:, book) == 1, 1, 'first');
if isempty(book_idx)
    p_choice = choice_prob(end);
else
    p_choice = choice_prob(book_idx);
end

joint_prob = lambda(stop_page) * p_choice;
nll = -log(max(joint_prob, 1e-12));
end


function lambda = NS_MNL_lambda(data, k)
user_id = unique(data(:, 1));
n_user = length(user_id);
stop_pages = zeros(n_user, 1);
max_page = 1;

for idx = 1:n_user
    uid = user_id(idx);
    query_data = data(data(:, 1) == uid, :);
    stop_pages(idx) = proxy_stop_page(query_data, k);
    max_page = max(max_page, total_pages(query_data, k));
end

count = accumarray(stop_pages, 1, [max_page, 1]);
lambda = count / max(sum(count), 1);
end


function [prob, stop_page] = NS_MNL_choice(query_data, k, beta, fea)
prod_data = query_data(1:end-1, :);
n_prod = size(prod_data, 1);

if n_prod == 0
    prob = 1;
    stop_page = 1;
    return;
end

stop_page = proxy_stop_page(query_data, k);
last_view_idx = min(stop_page * k, n_prod);

score = prod_data(:, fea:end) * beta;
score = max(min(score, 50), -50);
weight = exp(score);

denom = 1 + sum(weight(1:last_view_idx));
prob_prod = zeros(n_prod, 1);
prob_prod(1:last_view_idx) = weight(1:last_view_idx) / denom;
prob_no_purchase = 1 / denom;

prob = [prob_prod; prob_no_purchase];
prob = prob / max(sum(prob), 1e-12);
end


function stop_page = proxy_stop_page(query_data, k)
click = 6;
prod_data = query_data(1:end-1, :);
n_prod = size(prod_data, 1);

if n_prod == 0
    stop_page = 1;
    return;
end

clicked_idx = find(prod_data(:, click) == 1, 1, 'last');
if isempty(clicked_idx)
    stop_page = total_pages(query_data, k);
else
    stop_page = ceil(clicked_idx / k);
end
end


function n_page = total_pages(query_data, k)
n_prod = size(query_data, 1) - 1;
n_page = max(1, ceil(max(n_prod, 1) / k));
end


function [error, error_1, mape, score] = NS_MNL_pred(beta, k, data, fea, lambda)
user_id = unique(data(:, 1));
all_user = length(user_id);
book = 5;
sum_prod = 0;
error = 0;
error_1 = 0;
mape = 0;
hit = zeros(3, 1);
cal_user = 0;

for idx = 1:all_user
    uid = user_id(idx);
    query_data = data(data(:, 1) == uid, :);
    prob = NS_MNL_marginal_choice(query_data, k, beta, fea, lambda);

    error = error + sum((prob - query_data(:, book)).^2);
    error_1 = error_1 + sum((prob - query_data(:, book)).^2 ./ (0.5 + query_data(:, book)));
    mape = mape + sum(abs(prob - query_data(:, book)) ./ (query_data(:, book) + 0.5));
    sum_prod = sum_prod + size(query_data, 1);

    prod_prob = prob(1:end-1);
    prod_data = query_data(1:end-1, :);
    if sum(prod_data(:, book)) > 0
        purchase_prod_idx = find(prod_data(:, book) == 1, 1, 'first');
        [~, sort_index] = sort(-prod_prob);
        hit = hit + identify_hit(purchase_prod_idx, sort_index, 3);
        cal_user = cal_user + 1;
    end
end

error = sqrt(error / sum_prod);
error_1 = error_1 / sum_prod;
mape = mape / sum_prod;
score = hit / max(cal_user, 1);
end


function prob = NS_MNL_marginal_choice(query_data, k, beta, fea, lambda)
% Integrate the conditional MNL probability over the stopping-page
% distribution. Evaluation therefore does not condition on observed clicks.
prod_data = query_data(1:end-1, :);
n_prod = size(prod_data, 1);

if n_prod == 0
    prob = 1;
    return;
end

score = prod_data(:, fea:end) * beta;
score = max(min(score, 50), -50);
weight = exp(score);
prob = zeros(n_prod + 1, 1);

for stop_page = 1:length(lambda)
    last_view_idx = min(stop_page * k, n_prod);
    denom = 1 + sum(weight(1:last_view_idx));
    prob(1:last_view_idx) = prob(1:last_view_idx) ...
        + lambda(stop_page) * weight(1:last_view_idx) / denom;
    prob(end) = prob(end) + lambda(stop_page) / denom;
end

prob = prob / max(sum(prob), 1e-12);
end
