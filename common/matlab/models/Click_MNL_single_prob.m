function prob = Click_MNL_single_prob(data,beta1,fea)
%CLICK_MNL_SINGLE_PROB Return choice probabilities for one session.
%   BETA1 is [attraction; attention]; DATA ends with the outside-option row.

data = data(1:end-1,:);

num = size(data(:,fea:end),2);
beta = beta1(1:num);
alpha = beta1(num+1:2*num);

uti = exp(data(:,fea:end)*beta);
vti = exp(data(:,fea:end)*alpha);

v = squeeze(uti); %v: n x 1
q = vti./(1+vti);

n = size(data,1);
if n<=10
    prob = zeros(2^n, n);
    for j = 0:2^n-1
        bin = dec2bin(j);
        bin = strcat(repmat('0',1,n-length(bin)),bin);
        S = char(num2cell(bin));
        S = reshape(str2num(S),1,[]);
        S = S(end:-1:1);
        ind_1 = S == 1; % the click data
        ind_2 = S == 0;
        S = S';
        v_s = 1 + sum(v(ind_1));
        p1 = prod(q(ind_1))*prod(1-q(ind_2))/v_s;
        prob(j+1, :) = v.*S*p1;
    end
    % prob(:, end+1) = max(1- sum(prob,2),0);
    prob = sum(prob,1);
    prob = squeeze(prob);
    s = sum(prob);
    if s < 1
        prob(end+1) = 1 - s;
    else
        prob = prob / s * (1 - 1e-10);
        prob(end+1) = 1 - sum(prob);
    end
else
    sample = 2000;
    p1 = repmat(q,1,sample);
    b = binornd(1,p1);
    S = reshape(b,n,sample)'; % sample x n
    v1 = repmat(v,1,sample); % n x sample
    v1 = v1';
    v2 = S.*v1; % sample x n
    v2(:,end+1) = 1;
    v2 = v2./sum(v2,2);
    prob = mean(v2,1);
    prob = squeeze(prob);
end
prob = reshape(prob,[],1);
end
