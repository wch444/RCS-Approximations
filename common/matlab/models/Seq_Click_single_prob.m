function prob = Seq_Click_single_prob(data,beta1,fea)
%SEQ_CLICK_SINGLE_PROB Return SCM probabilities for one session.
%   BETA1 is [attraction; attention]; DATA ends with the outside-option row.

data = data(1:end-1,:);

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

        for j = 0:num_patterns-1
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
s = sum(prob);
if s < 1
    prob(end+1) = 1 - s;
else
    prob = prob / s * (1 - 1e-10);
    prob(end+1) = 1 - sum(prob);
end
prob = reshape(prob, n+1, 1);
end
