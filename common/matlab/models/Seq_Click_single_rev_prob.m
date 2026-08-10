function prob = Seq_Click_single_rev_prob(data,beta1,fea)
%SEQ_CLICK_SINGLE_REV_PROB Return SCM product probabilities for an offered set.
%   BETA1 is [attraction; attention], and DATA contains products only.

% data = data(1:end-1,:);

num = size(data(:,fea:end),2);
beta = beta1(1:num);
alpha = beta1(num+1:2*num);

uti = exp(data(:,fea:end)*beta);
vti = exp(data(:,fea:end)*alpha);

v = squeeze(uti);
q = vti./(1+vti);

n = size(data,1);

prob = zeros(n,1);
prob(1) = q(1)*v(1)/(1+v(1));

% Precompute log values for numerical stability
log_q = log(q);
log_1_q = log(1 - q);

for i = 2:n

    if i <= 10
        % Exact enumeration using bitwise operations (faster than string conversion)
        num_patterns = 2^(i-1);
        prob_i = 0;

        for j = 0:num_patterns-1  % consider all consideration set
            % Use bitget for fast binary decomposition
            S = bitget(j, 1:i-1);

            ind_1 = find(S == 1);
            ind_0 = find(S == 0);

            v_s = 1 + sum(v(ind_1));

            % Log-sum-exp for numerical stability
            log_term = sum(log_q(ind_1)) + sum(log_1_q(ind_0)) ...
                       - log(v_s + v(i)) - log(v_s);
            prob_i = prob_i + exp(log_term);
        end
        prob(i) = prob_i * q(i) * v(i);
    else
        prob(i) = sampling_prob(q,v,i)*q(i)*v(i);
    end
end
% prob(end+1) = max(1- sum(prob),0);
end
