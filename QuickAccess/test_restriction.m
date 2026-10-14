%% test_restriction.m
% Run from the SGLDBench project root.
clearvars;
clc;

addpath(fullfile(pwd, 'QuickAccess'));

% Change this to the MAT-file containing your initialized mesh hierarchy.
hierarchyFile = fullfile(pwd, 'meshHierarchy.mat');
assert(isfile(hierarchyFile), ...
    ['Hierarchy file not found: %s\n' ...
     'Save meshHierarchy_ from your initialized MATLAB session first.'], ...
    hierarchyFile);

S = load(hierarchyFile);
if isfield(S, 'meshHierarchy_')
    H = S.meshHierarchy_;
elseif isfield(S, 'H')
    H = S.H;
else
    error('MAT-file must contain meshHierarchy_ or H.');
end

numGPUs = 2;
level = 1;       % Restrict from this level to level + 1
numSamples = 64;

assert(numel(H) >= level + 1, 'Hierarchy has insufficient levels.');
assert(isfield(H(level), 'partitions') && ...
    numel(H(level).partitions) >= numGPUs, ...
    'Fine level has fewer than %d GPU partitions.', numGPUs);

rng(42);
fineResidual = randn(double(H(level).numDOFs), 1);

fprintf('Validating restriction: level %d -> %d on %d GPUs\n', ...
    level, level + 1, numGPUs);
maxAbsError = Validate_RestrictResidual_MGPU( ...
    H, numGPUs, fineResidual, level, numSamples);
fprintf('Maximum sampled absolute error: %.16e\n', maxAbsError);
