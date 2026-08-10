function prob = sampling_prob(q,v,i)
%SAMPLING_PROB Monte Carlo expectation for long-assortment SCM probabilities.
%   Q and V are attention and attraction values; I is the current product.

vi = v(1:i-1);
qi = q(1:i-1);
sample = 2000;
% prob = zeros(sample,1);
p1 = repmat(qi,1,sample);
b = binornd(1,p1);
b = reshape(b,i-1,sample)';
S = b==1;
v_s = S*vi + 1; % sample x 1
v_s1 = v_s + v(i);
prob = 1./(v_s.*v_s1);
prob = mean(prob);

end
