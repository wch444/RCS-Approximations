function [beta1, LL, score, error, error_1, mape, tt] = SCM_MNL_function(beta,fea, train_data, test_data,options,test)
%SCM_MNL_FUNCTION Fit or evaluate the sequential click model (SCM).
%   Parameters are returned as [attraction; attention]; TEST selects
%   evaluation-only mode.

tic
if test
    beta1 = beta;
    [error, error_1, mape, score] = Seq_click_MNL_pred(beta1,test_data,fea);
    LL = 0;
else
    alpha = fminunc(@(alpha)SCM_Click_LL(train_data,alpha,fea,1),beta,options);
    beta = fminunc(@(beta)SCM_Click_LL(train_data,beta,fea,0),beta,options);
    LL = SCM_Click_LL(test_data,alpha,fea,1) + SCM_Click_LL(test_data,beta,fea,0);
    % Downstream SCM routines read attraction coefficients first and
    % attention coefficients second.
    beta1 = [beta; alpha];
    [error, error_1, mape, score] = Seq_click_MNL_pred(beta1,test_data,fea);
end
tt = toc;
end


function LL = SCM_Click_LL(data,beta,fea,kk)
    prob = SCM_Click_all_prob(data,beta,fea,kk);
    LL = sum(prob);
end

function prob = SCM_Click_all_prob(data,beta,fea,kk)
    g = findgroups(data(:,1));
    if kk == 1
        fun = @(data){SCM_Click_single_choice_alpha(data,beta,fea)};
    else
        fun = @(data){SCM_Click_single_choice_beta(data,beta,fea)};
    end
    bb = splitapply(fun,data,g);
    prob = cell2mat(bb);
end


function LL = SCM_Click_single_choice_alpha(data,alpha,fea)

data = data(1:end-1,:); % exclude no-purchase

book = 5;
click = 6;

query_data = data;

if sum(data(:,book))>0

    [book_ind,~] = find(query_data(:,book) == 1);
    query_data = query_data(1:book_ind,:);  % only consider the products before purchase

    vti = exp(query_data(:,fea:end)*alpha);
    vti =  squeeze(vti);
    vti = vti./(1+vti);

    ind_click = query_data(:,click) == 1;
    ind_no_click = query_data(:,click) == 0;
    prob_1 = prod(1-vti(ind_no_click))*prod(vti(ind_click));
else
    vti = exp(query_data(:,fea:end)*alpha);
    vti =  squeeze(vti);
    vti = vti./(1+vti);
    ind_click = query_data(:,click) == 1;
    ind_no_click = query_data(:,click) == 0;
    prob_1 = prod(1-vti(ind_no_click))*prod(vti(ind_click));
end

LL = -log(prob_1);
end


function LL = SCM_Click_single_choice_beta(data,beta,fea)

data = data(1:end-1,:); % exclude no-purchase

book = 5;
click = 6;

query_data = data;

if sum(data(:,book))>0

    [book_ind,~] = find(query_data(:,book) == 1);
    query_data = query_data(1:book_ind,:);   % only consider the products before purchase

    uti = exp(query_data(:,fea:end)*beta);

    prob_2 = 1;
    if book_ind == 1
        prob_2 = prob_2*uti(book_ind)/(1+uti(book_ind));
    else
        query_1 = query_data(1:book_ind-1,:);
        uti_1 = exp(query_1(:,fea:end)*beta);
        ind_click_1 = query_1(:,click) == 1;
        v_c = sum(uti_1(ind_click_1));
        prob_2 = 1/(1+v_c)*uti(book_ind)/(1+v_c+uti(book_ind));
    end

    LL = -log(prob_2);
else
    ind_click = query_data(:,click) == 1;
    if isempty(ind_click)
        LL = 0;
    else
        uti = exp(query_data(ind_click,fea:end)*beta);
        uti =  squeeze(uti);
        LL = log(1+sum(uti));
    end
end

end
