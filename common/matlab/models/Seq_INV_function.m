function [beta, LL, score, error, error_1, mape, tt] = Seq_INV_function(beta, k, fea, train_data, test_data, options, test)
%SEQ_INV_FUNCTION Fit or evaluate the pagewise SeqINV model.
%   K is page size; TEST selects evaluation-only mode.
tic

if test == 0
    beta = fminunc(@(b)Seq_INV_LL(train_data, k, b, fea), beta, options);
end

LL = Seq_INV_LL(test_data, k, beta, fea);
[error, error_1, mape, score] = Seq_INV_pred(beta, k, test_data, fea);
tt = toc;
end


function LL = Seq_INV_LL(data, k, beta, fea)
g = findgroups(data(:, 1));
nll = splitapply(@(query_data)Seq_INV_single_nll(query_data, k, beta, fea), data, g);
LL = sum(nll);
end


function nll = Seq_INV_single_nll(query_data, k, beta, fea)
book = 5;
prob = Seq_INV_choice(query_data, k, beta, fea);
prod_data = query_data(1:end-1, :);

book_idx = find(prod_data(:, book) == 1, 1, 'first');
if isempty(book_idx)
    p = prob(end);
else
    p = prob(book_idx);
end

nll = -log(max(p, 1e-12));
end


function prob = Seq_INV_choice(query_data, k, beta, fea)
prod_data = query_data(1:end-1, :);
n_prod = size(prod_data, 1);

if n_prod == 0
    prob = 1;
    return;
end

score = prod_data(:, fea:end) * beta;
score = max(min(score, 50), -50);
weight = exp(score);

page_idx = ceil((1:n_prod)' / k);
n_page = page_idx(end);
page_mass = accumarray(page_idx, weight, [n_page, 1]);

prefix_no_purchase = ones(n_page, 1);
if n_page > 1
    prefix_no_purchase(2:end) = cumprod(1 ./ (1 + page_mass(1:end-1)));
end

prob_prod = prefix_no_purchase(page_idx) .* weight ./ (1 + page_mass(page_idx));
prob_no_purchase = prefix_no_purchase(end) / (1 + page_mass(end));

prob = [prob_prod; prob_no_purchase];
prob = prob / max(sum(prob), 1e-12);
end


function [error, error_1, mape, score] = Seq_INV_pred(beta, k, data, fea)
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
    prob = Seq_INV_choice(query_data, k, beta, fea);

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
