function tasks = build_expedia_task_grid(sites,trials,model_ids)
%BUILD_EXPEDIA_TASK_GRID Return the deterministic site/trial/model ordering.

config = expedia_experiment_config();
if nargin < 1 || isempty(sites), sites = config.sites; end
if nargin < 2 || isempty(trials), trials = config.trials; end
if nargin < 3 || isempty(model_ids), model_ids = config.modelIds; end

sites = validate_integer_selection(sites,config.sites,'sites');
trials = validate_integer_selection(trials,config.trials,'trials');
model_ids = string(model_ids(:)');
if isempty(model_ids) || numel(unique(model_ids)) ~= numel(model_ids)
    error('ExpediaExperiment:InvalidModels', ...
        'Models must be a nonempty unique selection.');
end
unknown = setdiff(model_ids,config.modelIds,'stable');
if ~isempty(unknown)
    error('ExpediaExperiment:InvalidModels', ...
        'Unknown model ids: %s',strjoin(unknown,', '));
end

n = numel(sites)*numel(trials)*numel(model_ids);
site_col = zeros(n,1);
trial_col = zeros(n,1);
model_col = strings(n,1);
seed_col = zeros(n,1);
row = 0;
for site = sites
    for trial = trials
        for model_id = model_ids
            row = row+1;
            model_index = find(config.modelIds == model_id,1);
            site_col(row) = site;
            trial_col(row) = trial;
            model_col(row) = model_id;
            seed_col(row) = config.seedBase + 1000*site + 10*trial + model_index;
        end
    end
end
tasks = table(site_col,trial_col,model_col,seed_col, ...
    'VariableNames',{'site','trial','model_id','seed'});
end


function values = validate_integer_selection(values,allowed,label)
if ~isnumeric(values) || ~isreal(values) || isempty(values) ...
        || any(~isfinite(values(:))) || any(values(:) ~= fix(values(:)))
    error('ExpediaExperiment:InvalidSelection', ...
        '%s must be a nonempty integer selection.',label);
end
values = double(values(:)');
if numel(unique(values)) ~= numel(values) || any(~ismember(values,allowed))
    error('ExpediaExperiment:InvalidSelection', ...
        '%s must be unique and drawn from [%s].', ...
        label,strjoin(string(allowed),', '));
end
end
