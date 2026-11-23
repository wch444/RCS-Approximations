function diagnostics = extract_expedia_fit_diagnostics(model_id,model,fit_diagnostics)
%EXTRACT_EXPEDIA_FIT_DIAGNOSTICS Normalize optimizer and IM-RCS EM audits.

config = expedia_experiment_config();
model_id = string(model_id);
if nargin < 3 || isempty(fit_diagnostics), fit_diagnostics = struct(); end

diagnostics = struct();
diagnostics.status = 'not_reported';
diagnostics.converged = false;
diagnostics.iterations = scalar_field(fit_diagnostics, ...
    {'iterations','numIterations','numIter'},NaN);
diagnostics.maxIterations = scalar_field(fit_diagnostics, ...
    {'maxIterations','maxIter'},NaN);
diagnostics.optimizerExitFlag = scalar_field(fit_diagnostics, ...
    {'exitflag','exitFlag','optimizerExitFlag'},NaN);
diagnostics.hitIterationCap = false;
diagnostics.finalLogLikGain = NaN;
diagnostics.finalRelativeLogLikGain = NaN;
diagnostics.monotonicityViolations = 0;
diagnostics.worstLogLikDecrease = 0;
diagnostics.innerOptimizerFailures = 0;

reported_convergence = logical_field(fit_diagnostics,{'converged'},[]);
if ~isempty(reported_convergence)
    diagnostics.converged = reported_convergence;
    diagnostics.status = char(string_field(fit_diagnostics,{'status'}, ...
        ternary(reported_convergence,'converged','not_converged')));
elseif isfinite(diagnostics.optimizerExitFlag)
    diagnostics.converged = diagnostics.optimizerExitFlag > 0;
    diagnostics.status = ternary(diagnostics.converged,'converged','optimizer_not_converged');
end

if model_id ~= "imrcs"
    return
end

diagnostics.maxIterations = model_max_iterations(model,fit_diagnostics,config.imrcsOuterMaxIter);
if diagnostics.maxIterations ~= config.imrcsOuterMaxIter
    error('ExpediaDiagnostics:WrongIMRCSMaxIter', ...
        'IM-RCS maxIterations is %g; the frozen protocol requires exactly %d.', ...
        diagnostics.maxIterations,config.imrcsOuterMaxIter);
end

log_lik = history_vector(model,fit_diagnostics,{'logLik','log_lik'});
if isempty(log_lik) || any(~isfinite(log_lik))
    error('ExpediaDiagnostics:MissingIMRCSHistory', ...
        'IM-RCS must expose a finite observed-log-likelihood history.');
end
gain = diff(log_lik);
diagnostics.iterations = numel(gain);
if ~isempty(gain)
    previous_scale = max(1,abs(log_lik(1:end-1)));
    relative_gain = gain(:)./previous_scale(:);
    diagnostics.finalLogLikGain = gain(end);
    diagnostics.finalRelativeLogLikGain = relative_gain(end);
    material_drop = gain(:) < -config.monotonicityTolerance;
    diagnostics.monotonicityViolations = sum(material_drop);
    diagnostics.worstLogLikDecrease = min([0; gain(:)]);
end

exit_flags = history_vector(model,fit_diagnostics, ...
    {'thetaExitFlag','innerExitFlag','innerExitFlags'});
if ~isempty(exit_flags)
    diagnostics.innerOptimizerFailures = sum(isfinite(exit_flags) & exit_flags <= 0);
end
diagnostics.hitIterationCap = diagnostics.iterations >= diagnostics.maxIterations;

if isempty(reported_convergence)
    diagnostics.converged = ~diagnostics.hitIterationCap ...
        && diagnostics.iterations > 0 ...
        && diagnostics.finalLogLikGain >= -config.monotonicityTolerance ...
        && diagnostics.finalRelativeLogLikGain <= config.emRelativeTolerance;
end
if diagnostics.monotonicityViolations > 0
    diagnostics.status = 'monotonicity_violation';
    diagnostics.converged = false;
elseif diagnostics.innerOptimizerFailures > 0
    diagnostics.status = 'inner_optimizer_failure';
    diagnostics.converged = false;
elseif diagnostics.hitIterationCap
    diagnostics.converged = false;
    diagnostics.status = 'hit_cap';
elseif diagnostics.converged
    diagnostics.status = 'converged';
else
    diagnostics.status = 'stopped_not_converged';
end
end


function value = model_max_iterations(model,fit_diagnostics,fallback)
value = scalar_field(fit_diagnostics, ...
    {'maxIterations','maxIter','imrcsOuterMaxIter'},NaN);
if isfinite(value), return; end
if isstruct(model) && isfield(model,'opts') && isstruct(model.opts)
    value = scalar_field(model.opts, ...
        {'maxIterations','maxIter','imrcsOuterMaxIter'},NaN);
end
if ~isfinite(value), value = fallback; end
end


function value = history_vector(model,fit_diagnostics,names)
value = [];
containers = cell(3,1);
container_count = 0;
if isstruct(model) && isfield(model,'history') && isstruct(model.history)
    container_count = container_count+1;
    containers{container_count} = model.history;
end
if isstruct(fit_diagnostics)
    if isfield(fit_diagnostics,'history') && isstruct(fit_diagnostics.history)
        container_count = container_count+1;
        containers{container_count} = fit_diagnostics.history;
    end
    container_count = container_count+1;
    containers{container_count} = fit_diagnostics;
end
for c = 1:container_count
    for n = 1:numel(names)
        if isfield(containers{c},names{n})
            candidate = containers{c}.(names{n});
            if isnumeric(candidate) && isreal(candidate)
                value = candidate(:);
                return
            end
        end
    end
end
end


function value = scalar_field(container,names,fallback)
value = fallback;
if ~isstruct(container), return; end
for i = 1:numel(names)
    if isfield(container,names{i})
        candidate = container.(names{i});
        if isnumeric(candidate) && isreal(candidate) && isscalar(candidate)
            value = double(candidate);
            return
        end
    end
end
end


function value = logical_field(container,names,fallback)
value = fallback;
if ~isstruct(container), return; end
for i = 1:numel(names)
    if isfield(container,names{i})
        candidate = container.(names{i});
        if (islogical(candidate) || isnumeric(candidate)) && isscalar(candidate) ...
                && ismember(candidate,[0 1])
            value = logical(candidate);
            return
        end
    end
end
end


function value = string_field(container,names,fallback)
value = string(fallback);
if ~isstruct(container), return; end
for i = 1:numel(names)
    if isfield(container,names{i})
        candidate = container.(names{i});
        if (ischar(candidate) && isrow(candidate)) ...
                || (isstring(candidate) && isscalar(candidate))
            value = string(candidate);
            return
        end
    end
end
end


function value = ternary(condition,true_value,false_value)
if condition, value = true_value; else, value = false_value; end
end
