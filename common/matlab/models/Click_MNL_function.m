function [beta1, LL, score, error, error_1, mape, tt] = Click_MNL_function(beta,k,fea, train_data, test_data,options,test)
%CLICK_MNL_FUNCTION Fit or evaluate the consideration-set MNL (CMNL).
%   Parameters are returned as [attraction; attention]; TEST selects
%   evaluation-only mode.
tic

if test == 0
attn = fminunc(@(alpha)Click_MNL_LL(train_data,alpha,fea,1),beta,options);   % attention params
attr = fminunc(@(beta)Click_MNL_LL(train_data,beta,fea,0),beta,options);     % attraction params
beta = [attr; attn];   % full CMNL params: [attraction(1:num); attention(num+1:2*num)]
end


LL = 0;

[error, error_1, mape, score, LL] = click_MNL_pred(beta,test_data,fea);
beta1 = beta;   % return the FULL parameter vector. Previously this returned the
                % attraction half only, so the attention params were lost whenever a
                % caller saved fit.models.cmnl (consumed as [attr;attn] by
                % Click_MNL_single_prob, matching SCM_MNL_function's 26-vector).
tt = toc;
end


function LL = Click_MNL_LL(data,beta,fea,kk)
    prob = Click_MNL_all_prob(data,beta,fea,kk);
    LL = sum(prob);
end


function prob = Click_MNL_all_prob(data,beta,fea,kk)
    g = findgroups(data(:,1));
    if kk == 1
        fun = @(data){Click_MNL_single_choice_alpha(data,beta,fea)};
    else
        fun = @(data){Click_MNL_single_choice_beta(data,beta,fea)};
    end
    bb = splitapply(fun,data,g);
    prob = cell2mat(bb);
end

function LL = Click_MNL_single_choice_alpha(data,alpha,fea)

data = data(1:end-1,:); % exclude no-purchase


click = 6;
query_data = data;

vti = exp(query_data(:,fea:end)*alpha);
vti =  squeeze(vti);
vti = vti./(1+vti);
ind_click = query_data(:,click) == 1;
ind_no_click = query_data(:,click) == 0;
prob_1 = prod(1-vti(ind_no_click))*prod(vti(ind_click));
LL = -log(prob_1);
end


function LL = Click_MNL_single_choice_beta(data,beta,fea)

data = data(1:end-1,:); % exclude no-purchase

book = 5;
click = 6;

query_data = data;
ind_click = query_data(:,click) == 1;
if sum(data(:,book))>0
    [book_ind,~] = find(query_data(:,book) == 1);
    uti = exp(query_data(ind_click,fea:end)*beta);
    uti =  squeeze(uti);
    vti = query_data(book_ind,fea:end)*beta;
    LL = -vti + log(1+sum(uti));
else
    if isempty(ind_click)
        LL = 0;
    else
        uti = exp(query_data(ind_click,fea:end)*beta);
        uti =  squeeze(uti);
        LL = log(1+sum(uti));
    end
end

end
