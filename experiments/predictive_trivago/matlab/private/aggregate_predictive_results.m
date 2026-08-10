function means = aggregate_predictive_results(results,markets,model_names)
%AGGREGATE_PREDICTIVE_RESULTS Validate and aggregate predictive trial rows.
%   CMNL and SCM test NLL are deliberately undefined under the maintained
%   Monte Carlo/observed-path evaluation contract.  Every task for those two
%   models must therefore carry NaN, while the other four models must carry a
%   finite NLL.  All displayed and auxiliary predictive metrics must be finite.

metric_names = {'NLL','RMSE','WeightedSquaredError','MAPE', ...
    'Hit1','Hit2','Hit3'};
required_names = [{'Market','Trial','Model'},metric_names];
if ~istable(results) || ~all(ismember(required_names, ...
        results.Properties.VariableNames))
    error('aggregate_predictive_results:InvalidResults', ...
        'Predictive results are missing required task or metric columns.');
end
markets = string(markets(:).');
model_names = string(model_names(:).');
if isempty(markets) || isempty(model_names) ...
        || numel(unique(markets)) ~= numel(markets) ...
        || numel(unique(model_names)) ~= numel(model_names)
    error('aggregate_predictive_results:InvalidAxes', ...
        'Market and model axes must be nonempty and unique.');
end

observed_markets = string(results.Market);
observed_models = string(results.Model);
if any(~ismember(observed_markets,markets)) ...
        || any(~ismember(observed_models,model_names))
    error('aggregate_predictive_results:InvalidResults', ...
        'Predictive results contain a task outside the requested axes.');
end
if ~isnumeric(results.Trial) || any(~isfinite(results.Trial)) ...
        || any(results.Trial ~= fix(results.Trial))
    error('aggregate_predictive_results:InvalidResults', ...
        'Predictive trial identifiers must be finite integers.');
end

n = numel(markets)*numel(model_names);
market_col = strings(n,1);
model_col = strings(n,1);
metric = nan(n,numel(metric_names));
rows_per_task = [];
row = 0;
undefined_nll_models = ["CMNL","SCM"];
for market = markets
    for model = model_names
        row = row+1;
        use = observed_markets == market & observed_models == model;
        count = sum(use);
        if count == 0 || numel(unique(results.Trial(use))) ~= count
            error('aggregate_predictive_results:IncompleteGrid', ...
                'Each market/model cell must contain unique nonempty trial rows.');
        end
        if isempty(rows_per_task)
            rows_per_task = count;
        elseif count ~= rows_per_task
            error('aggregate_predictive_results:IncompleteGrid', ...
                'Every market/model cell must contain the same number of trials.');
        end

        nll = results.NLL(use);
        other = results{use,metric_names(2:end)};
        if ~isnumeric(nll) || ~isreal(nll) || ~isnumeric(other) ...
                || ~isreal(other) || any(~isfinite(other),'all')
            error('aggregate_predictive_results:InvalidMetric', ...
                'All non-NLL predictive metrics must be finite real numbers.');
        end
        if ismember(model,undefined_nll_models)
            if ~all(isnan(nll))
                error('aggregate_predictive_results:InvalidNLLContract', ...
                    'CMNL and SCM task NLL must be uniformly undefined (NaN).');
            end
            mean_nll = NaN;
        else
            if any(~isfinite(nll))
                error('aggregate_predictive_results:InvalidNLLContract', ...
                    'MNL, MMNL, SeqINV, and NSMNL task NLL must all be finite.');
            end
            mean_nll = mean(nll);
        end
        market_col(row) = market;
        model_col(row) = model;
        metric(row,:) = [mean_nll,mean(other,1)];
    end
end

means = table(market_col,model_col,metric(:,1),metric(:,2),metric(:,3), ...
    metric(:,4),metric(:,5),metric(:,6),metric(:,7), ...
    'VariableNames',{'Market','Model','NLL','RMSE', ...
    'WeightedSquaredError','MAPE','Hit1','Hit2','Hit3'});
end
