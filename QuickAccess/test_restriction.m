%% test_restriction.m
% Run from the SGLDBench project root.
%% DEMO: Stress Analysis
clear all; clc;
addpath('../');
addpath('../src/');
addpath('../src/MEXfuncs/');

Data_GlobalVariables;
numGPUs = 2;
outPath_ = '../out/';
if ~exist(outPath_, 'dir'), mkdir(outPath_); end

%%1. Data Loading
tStart = tic;
MdlSelect = 'Bone';
IO_LoadBuiltInDatasets(MdlSelect, 512, numGPUs);
disp(['Prepare Voxel Model Costs: ', sprintf('%10.3g',toc(tStart)) 's']);

% figure; view(gca,3);
% Vis_DrawMesh3D(gca, meshHierarchy_(1).boundaryNodeCoords, meshHierarchy_(1).boundaryEleFaces, 0);
% Vis_ShowLoadingCondition(gca, loadingCond_);
% Vis_ShowFixingCondition(gca, fixingCond_);

%% 2. Setup FEA
tStart = tic;
FEA_ApplyBoundaryCondition();
FEA_SetupVoxelBased(numGPUs);
densityField = ones(meshHierarchy_(1).numElements,1); %%fully solid domain
meshHierarchy_(1).eleModulus = TopOpti_MaterialInterpolationSIMP(densityField(:));
disp(['Setup FEA Costs: ', sprintf('%10.3g',toc(tStart)) 's']);

%% 3. Assemble Computing Stencil
tStart = tic;
Solving_AssembleFEAstencil();
disp(['Assemble Computing Stencil Costs: ', sprintf('%10.3g',toc(tStart)) 's']);

addpath(fullfile(pwd, 'QuickAccess'));

H= meshHierarchy_;
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
    H, numGPUs, 1e-05);
fprintf('Maximum sampled absolute error: %.16e\n', maxAbsError);
