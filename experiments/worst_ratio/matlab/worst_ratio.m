% WORST_RATIO Numerical upper-bound construction for the SCM/RCS ratio.
% Run from the repository root:
%   run('experiments/worst_ratio/matlab/worst_ratio.m')
% Requires MATLAB with Optimization Toolbox (fmincon).
%
% For each product count n, x = [r(1:n), v(1:n)] contains revenues and
% attraction values. Attention probabilities are all one, so the singleton
% choice probabilities are q(i) = v(i)/(1 + v(i)). Revenue r(1) is normalized
% to one; all remaining variables have the positive lower bound epsilon.
%
% The objective divides the singleton revenue r(1)*q(1) by the RCS cascade
% benchmark in index order 1,...,n. For every prefix 1,...,k, the nonlinear
% constraints require its SCM revenue in reverse index order k,...,1 to be
% no larger than the singleton revenue. The script searches this construction
% using SQP from the all-ones starting point and plots the resulting ratios.
% The paper reports a value of approximately 0.407 at n = 15.
%
% Outputs in the workspace:
%   n_list, ratio_list: product counts and optimized objective values.
%   xx, fval: optimizer solution and objective for the final product count.

n_list = 2:1:50;
ratio_list = ones(1, length(n_list));

for i = 1:length(n_list)
    n = n_list(i);
    epsilon = 1e-5;
    % Fix r(1) = 1 and bound the other 2*n - 1 variables below by epsilon.
    lb = [1, ones(1, 2*n-1)*epsilon];
    ub = [1, ones(1, 2*n-1)*inf];

    % Preserve the solver settings used in the numerical construction.
    options = optimoptions('fmincon','Algorithm','sqp', ...
        'ConstraintTolerance', 1e-10, 'MaxFunctionEvaluations', 1e6, 'MaxIterations', 1e4);
    [xx, fval] = fmincon(@(x)myObj(x), ones(1, 2*n),[],[],[],[],lb,ub,@(x)nlcon(x), options);
    ratio_list(i) = fval;
end

box off
plot(n_list, ratio_list, 'LineWidth',1.5, 'LineStyle', '--', 'Color', 'k');
xlabel('$n$', 'FontSize',15, 'Interpreter','latex');
ylabel('Upper bound', 'FontSize',15);

% Objective: singleton SCM revenue divided by the RCS cascade benchmark.
function obj = myObj(x)
    n = length(x) / 2;
    r = x(1:n);
    v = x(n+1:2*n);
    q = v./ (1+v);

    temp = 0;
    tmp = 1;
    for i = 1:length(r)
        temp = temp + tmp * q(i) * r(i);
        tmp = tmp * (1 - q(i));
    end
    obj = (r(1) * v(1) / (1 + v(1))) / temp;
end

% fmincon uses nl <= 0. There are no nonlinear equality constraints.
function [nl,nleq] = nlcon(x)
    n = length(x) / 2;
    r = x(1:n);
    v = x(n+1:2*n);

    nl = [];
    nleq = [];
    for k = 2:length(r)
        nl = [nl; con(r, v, k)];
    end
end

% Prefix SCM revenue in order k,...,1, minus the singleton revenue.
% With attention equal to one, each purchase probability is the difference
% between the outside-option probabilities before and after adding product i.
function y = con(r, v, k)
    y = 0;
    for i = 1:k
        y = y + r(i)*v(i) / ((1+sum(v(i+1:k)))*(1+sum(v(i:k))));
    end
    y = y - r(1)*v(1) / (1+v(1));
end
