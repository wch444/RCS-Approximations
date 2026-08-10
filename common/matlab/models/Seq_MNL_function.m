function [beta, LL, score, error, error_1, mape, tt] = Seq_MNL_function(beta,k,fea, train_data, test_data,options,test)
%SEQ_MNL_FUNCTION Fit or evaluate the pagewise sequential MNL model.
%   K is page size; TEST selects evaluation-only mode.
tic
if test==0
    beta = fminunc(@(beta)SMNL_LL(train_data,k,beta,fea),beta,options);
end

LL = SMNL_LL(test_data,k,beta,fea);
score = SMNL_hit(beta,k,test_data,fea);

[error, error_1, mape] = SMNL_pred(beta,k,test_data,fea);
tt = toc;
end


function LL = SMNL_LL(data,k,beta,fea)
    prob = SMNL_all_prob(data,k,beta,fea);
    index = data(:,5)==1;
    LL_s = prob(index,:);
    LL = -sum(log(LL_s));
end

function prob = SMNL_all_prob(data,k,beta,fea)
    g = findgroups(data(:,1));
    fun = @(data){SMNL_single_choice(data,k,beta,fea)};
    bb = splitapply(fun,data,g);
    prob = cell2mat(bb);
end


function prob = SMNL_single_choice(data,k,beta,fea)

data = data(1:end-1,:); % exclude no-purchase
% data k x d, beta d x 1
uti = exp(data(:,fea:end)*beta);
uti = squeeze(uti);
len = size(data,1);
stage = floor(len/k);
prob = zeros(len,1);
prob = squeeze(prob);

if stage == 0
    prob(1:len) = uti(1:len)/(1+sum(uti(1:len)));
else
    prob(1:k) = uti(1:k)/(1+sum(uti(1:k)));

    for i = 1:stage
        if i<stage
            prob(i*k+1:(i+1)*k) = uti(i*k+1:(i+1)*k)/(1+sum(uti(1:i*k)))/(1+sum(uti(1:(i+1)*k)));
        else
            prob(i*k+1:end) = uti(i*k+1:end)/(1+sum(uti(1:i*k)))/(1+sum(uti));
        end
    end
end
% sum_purchase_probability = sum(prob);
prob(end+1,1) = 1/(1+sum(uti));
end

function [error, error_1, mape] = SMNL_pred(beta,k,data,fea)

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
        prob = SMNL_choice(query_data,k,beta,fea);
        error = error + sum((prob - query_data(:,book)).^2);
        error_1 = error_1 + sum((prob - query_data(:,book)).^2./(0.5+query_data(:,book)));
        mape = mape + sum(abs(prob - query_data(:,book))./(query_data(:,book) + 0.5));
        sum_prod = sum_prod + n;
    end
    error = sqrt(error/sum_prod);
    error_1 = error_1/sum_prod;
    mape = mape/sum_prod;
end


function score = SMNL_hit(beta,k,data,fea)

user_Id = unique(data(:,1));
all_user = length(user_Id);
hit = zeros(3,1);
cal_user = 0;
book = 5;
for index = 1:all_user

    i = user_Id(index);
    query_data = data(data(:,1)==i,:);
    prob = SMNL_choice(query_data,k,beta,fea);
    query_data = query_data(1:end-1,:); %exclude no-purchhase
    prob = prob(1:end-1,:);

    if sum(query_data(:,book))>0
        [purchase_prod_idx,~] = find(query_data(:,book) == 1);
        [~,sort_index] = sort(-prob);
        hit = hit + identify_hit(purchase_prod_idx,sort_index,3);
        cal_user = cal_user + 1;
    end

end
score = hit/cal_user;
end

function prob = SMNL_choice(data,k,beta,fea)
% data k x d, beta d x 1
data = data(1:end-1,:); % exclude no-purchase
uti = exp(data(:,fea:end)*beta);
uti = squeeze(uti);
len = size(data,1);
stage = floor(len/k);
prob = zeros(len,1);
prob = squeeze(prob);

if stage == 0
    prob(1:len) = uti(1:len)/(1+sum(uti(1:len)));
else
    prob(1:k) = uti(1:k)/(1+sum(uti(1:k)));

    for i = 1:stage
        if i<stage
            prob(i*k+1:(i+1)*k) = uti(i*k+1:(i+1)*k)/(1+sum(uti(1:i*k)))/(1+sum(uti(1:(i+1)*k)));
        else
            prob(i*k+1:end) = uti(i*k+1:end)/(1+sum(uti(1:i*k)))/(1+sum(uti(1:end)));
        end
    end
end
prob(end+1,1) = 1-sum(prob);
end
