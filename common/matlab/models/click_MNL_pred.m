function [error, error_1, mape, score, LL] = click_MNL_pred(beta,data,fea)
%CLICK_MNL_PRED Compute pooled errors, Hit@1:3, and NLL metrics for CMNL.
%   BETA is [attraction; attention]; DATA has one outside row per session.

    user_Id = unique(data(:,1));
    all_user = length(user_Id);
    book = 5;
    sum_prod = 0;
    error = 0;
    hit = zeros(3,1);
    cal_user = 0;
    LL = 0;
    error_1 = 0;
    mape = 0;
    for index = 1:all_user
        i = user_Id(index);
        query_data = data(data(:,1)==i,:);
        n = size(query_data,1);
        prob = Click_MNL_single_prob(query_data,beta,fea);
        error = error + sum((prob - query_data(:,book)).^2);
        error_1 = error_1 + sum((prob - query_data(:,book)).^2./(0.5+query_data(:,book)));
        mape = mape + sum(abs(prob - query_data(:,book))./(query_data(:,book) + 0.5));
        [book_ind,~] = find(query_data(:,book) == 1);
        LL = LL - log(prob(book_ind));
        prob = prob(1:end-1,:);
        query_data = query_data(1:end-1,:);
        sum_prod = sum_prod + n;
        if sum(query_data(:,book))>0
            [purchase_prod_idx,~] = find(query_data(:,book) == 1);
            [~,sort_index] = sort(-prob);
            hit = hit + identify_hit(purchase_prod_idx,sort_index,3);
            cal_user = cal_user + 1;
        end
    end
    error = sqrt(error/sum_prod);
    score = hit/cal_user;
    error_1 = error_1/sum_prod;
    mape = mape/sum_prod;
end
