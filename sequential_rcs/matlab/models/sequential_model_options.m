function opts = sequential_model_options(opts)
%SEQUENTIAL_MODEL_OPTIONS Canonical options for the Expedia four-model core.
%
% Position is deliberately not a utility feature.  It is retained only to
% order products, form stages, construct the position ranking in IM-RCS, and
% break ranking ties.  The utility/matching feature set is fixed at columns
% 8:20 (intercept plus the twelve hotel/search attributes).

if nargin < 1 || isempty(opts)
    opts = struct();
end

% Accept the first implementation's generic names only as input aliases.
% Remove them immediately so saved model options use the unambiguous frozen
% protocol names below.
opts = migrate_alias(opts, 'maxIter', 'directMaxIter');
opts = migrate_alias(opts, 'emMaxIter', 'imrcsOuterMaxIter');
opts = migrate_alias(opts, 'tol', 'optimizerOptimalityTolerance');
opts = migrate_alias(opts, 'emTol', 'emRelativeTolerance');
opts = migrate_alias(opts, 'outsidePosition', 'noPurchasePosition');

defaults = struct();
defaults.queryCol = 1;
defaults.itemCol = 2;
defaults.positionCol = 4;
defaults.choiceCol = 5;
defaults.clickCol = 6;
defaults.featureCols = 8:20;
defaults.noPurchasePosition = 100;
defaults.stageSize = 10;
defaults.theta0 = [];
defaults.p0 = [];
defaults.beta0 = [];
defaults.lambda = [];
defaults.rankings = [];
defaults.directMaxIter = 500;
defaults.imrcsOuterMaxIter = 100;
defaults.innerMaxIter = 500;
defaults.maxFunctionEvaluations = 1e5;
defaults.optimizerOptimalityTolerance = 1e-6;
defaults.emRelativeTolerance = 1e-6;
defaults.monotonicityTolerance = 1e-7;
defaults.probabilityTolerance = 1e-8;
defaults.probFloor = 1e-12;
defaults.numStarts = 3;
defaults.rngSeed = 42;
defaults.verbose = false;
defaults.ridge = 0;
defaults.optimOptions = [];
defaults.innerOptimOptions = [];
defaults.gradientCheck = false;
defaults.gradientCheckStep = 1e-6;
defaults.gradientCheckTolerance = 1e-4;

names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = defaults.(name);
    end
end

if ~isequal(opts.featureCols(:)', 8:20)
    error('SequentialModels:FeatureSchema', ...
        ['The Expedia comparison is fixed to feature columns 8:20. ' ...
         'Position encodings or trailing columns are not permitted.']);
end
if ~isscalar(opts.stageSize) || opts.stageSize < 1 || opts.stageSize ~= floor(opts.stageSize)
    error('SequentialModels:StageSize', 'stageSize must be a positive integer.');
end
if ~isscalar(opts.imrcsOuterMaxIter) || opts.imrcsOuterMaxIter ~= 100
    error('SequentialModels:EMIterations', ...
        'The approved IM-RCS specification fixes imrcsOuterMaxIter at 100.');
end
if ~is_positive_integer(opts.directMaxIter)
    error('SequentialModels:DirectIterations', ...
        'directMaxIter must be a positive integer.');
end
if ~is_positive_integer(opts.innerMaxIter)
    error('SequentialModels:InnerIterations', ...
        'innerMaxIter must be a positive integer.');
end
if ~isscalar(opts.optimizerOptimalityTolerance) ...
        || ~isfinite(opts.optimizerOptimalityTolerance) ...
        || opts.optimizerOptimalityTolerance <= 0
    error('SequentialModels:OptimizerTolerance', ...
        'optimizerOptimalityTolerance must be a positive finite scalar.');
end
if ~isscalar(opts.emRelativeTolerance) || ~isfinite(opts.emRelativeTolerance) ...
        || opts.emRelativeTolerance <= 0
    error('SequentialModels:EMTolerance', ...
        'emRelativeTolerance must be a positive finite scalar.');
end
if ~isscalar(opts.monotonicityTolerance) || ~isfinite(opts.monotonicityTolerance) ...
        || opts.monotonicityTolerance < 0
    error('SequentialModels:MonotonicityTolerance', ...
        'monotonicityTolerance must be a nonnegative finite scalar.');
end
if ~is_positive_integer(opts.numStarts)
    error('SequentialModels:NumStarts', 'numStarts must be a positive integer.');
end
if ~isscalar(opts.probFloor) || ~isfinite(opts.probFloor) || ...
        opts.probFloor <= 0 || opts.probFloor >= 1e-3
    error('SequentialModels:ProbabilityFloor', ...
        'probFloor must be finite and lie strictly between 0 and 1e-3.');
end
if ~isscalar(opts.probabilityTolerance) || ~isfinite(opts.probabilityTolerance) || ...
        opts.probabilityTolerance <= 0 || opts.probabilityTolerance >= 1e-3
    error('SequentialModels:ProbabilityTolerance', ...
        'probabilityTolerance must be finite and lie strictly between 0 and 1e-3.');
end
if ~isscalar(opts.gradientCheck) || ~ismember(opts.gradientCheck, [0 1])
    error('SequentialModels:GradientCheckFlag', ...
        'gradientCheck must be a logical scalar.');
end
if ~isscalar(opts.gradientCheckStep) || ~isfinite(opts.gradientCheckStep) ...
        || opts.gradientCheckStep <= 0
    error('SequentialModels:GradientCheckStep', ...
        'gradientCheckStep must be a positive finite scalar.');
end
if ~isscalar(opts.gradientCheckTolerance) ...
        || ~isfinite(opts.gradientCheckTolerance) ...
        || opts.gradientCheckTolerance <= 0
    error('SequentialModels:GradientCheckTolerance', ...
        'gradientCheckTolerance must be a positive finite scalar.');
end
end

function opts = migrate_alias(opts, old_name, new_name)
if isfield(opts, old_name)
    if ~isfield(opts, new_name) || isempty(opts.(new_name))
        opts.(new_name) = opts.(old_name);
    end
    opts = rmfield(opts, old_name);
end
end

function value = is_positive_integer(value)
value = isnumeric(value) && isreal(value) && isscalar(value) ...
    && isfinite(value) && value >= 1 && value == floor(value);
end
