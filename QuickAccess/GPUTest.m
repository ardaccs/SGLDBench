%% DEMO: Stress Analysis
clear all; clc;
addpath('../');
addpath('../src/');
addpath('../src/MEXfuncs/');

Data_GlobalVariables;
outPath_ = '../out/';
if ~exist(outPath_, 'dir'), mkdir(outPath_); end

%%1. Data Loading
tStart = tic;
MdlSelect = 'Bone'; %% Bone, Part, Part2, Part3, Bracket_GE, Molar, Fertility, Hanger, TopOptiShape
IO_LoadBuiltInDatasets(MdlSelect);
disp(['Prepare Voxel Model Costs: ', sprintf('%10.3g',toc(tStart)) 's']);

% figure; view(gca,3);
% Vis_DrawMesh3D(gca, meshHierarchy_(1).boundaryNodeCoords, meshHierarchy_(1).boundaryEleFaces, 0);
% Vis_ShowLoadingCondition(gca, loadingCond_);
% Vis_ShowFixingCondition(gca, fixingCond_);

%% 2. Setup FEA
tStart = tic;
FEA_ApplyBoundaryCondition();
FEA_SetupVoxelBased();
densityField = ones(meshHierarchy_(1).numElements,1); %%fully solid domain
meshHierarchy_(1).eleModulus = TopOpti_MaterialInterpolationSIMP(densityField(:));
disp(['Setup FEA Costs: ', sprintf('%10.3g',toc(tStart)) 's']);

%% 3. Assemble Computing Stencil
tStart = tic;
Solving_AssembleFEAstencil();
disp(['Assemble Computing Stencil Costs: ', sprintf('%10.3g',toc(tStart)) 's']);

%% 4. Compare CPU and CUDA MGPCG
fprintf('\n====================================================\n');
fprintf('CPU vs CUDA MGPCG test\n');
fprintf('====================================================\n');

% Build the hierarchy struct used by the CUDA MEX.
H_ = Build_GPU_Hierarchy(meshHierarchy_);

% Both solvers must start from the same initial solution.
U0 = zeros(size(F_), 'double');

%% 4.1 Check GPU hierarchy input
fprintf('\n--- GPU hierarchy information ---\n');
fprintf('Number of levels:       %d\n', H_.numLevels);
fprintf('Finest-level DOFs:      %d\n', H_.numDOFs(1));
fprintf('Coarsest-level DOFs:    %d\n', H_.numDOFs(end));
fprintf('Coarsest free DOFs:     %d\n', numel(H_.coarseFreeDOFIds));
fprintf('Coarse matrix size:     %d x %d\n', ...
    size(H_.coarseKFree, 1), ...
    size(H_.coarseKFree, 2));
fprintf('Coarse matrix nnz:      %d\n', nnz(H_.coarseKFree));
fprintf('Jacobi omega:           %.6f\n', H_.jacobiOmega);

assert(issparse(H_.coarseKFree), ...
    'H_.coarseKFree must remain sparse.');

assert(size(H_.coarseKFree, 1) == ...
       numel(H_.coarseFreeDOFIds), ...
    'coarseKFree size does not match coarseFreeDOFIds.');

assert(all(isfinite(nonzeros(H_.coarseKFree))), ...
    'coarseKFree contains NaN or Inf.');

%% 4.3 CUDA solve
fprintf('\n--- Running CUDA MGPCG ---\n');

mexPath = which('Solving_MGPCG_GPU_coarse_pcg');

if isempty(mexPath)
    error([ ...
        'Solving_MGPCG_GPU was not found. ', ...
        'Compile the CUDA MEX and add its directory to the MATLAB path.']);
end

fprintf('Using CUDA MEX:\n%s\n', mexPath);

gpuTimer = tic;

[U_gpu, gpuIterations, gpuReportedRelres] = ...
    Solving_MGPCG_GPU_coarse_pcg( ...
        F_, ...
        tol_, ...
        maxIT_, ...
        U0, ...
        H_);

gpuTime = toc(gpuTimer);

fprintf('CUDA end-to-end time: %.6f s\n', gpuTime);
fprintf('CUDA iterations:      %d\n', gpuIterations);
fprintf('CUDA reported relres: %.6e\n', gpuReportedRelres);
%% 4.2 CPU solve
fprintf('\n--- Running CPU MGPCG ---\n');

cpuTimer = tic;

U_cpu = Solving_PreconditionedConjugateGradientSolver( ...
    @Solving_KbyU_MatrixFree, ...
    @Solving_Vcycle, ...
    F_, ...
    tol_, ...
    maxIT_, ...
    'printP_ON', ...
    U0);

cpuTime = toc(cpuTimer);

fprintf('CPU solve time: %.6f s\n', cpuTime);


%% 4.4 Basic solution checks
if any(~isfinite(U_cpu))
    error('CPU solution contains NaN or Inf.');
end

if any(~isfinite(U_gpu))
    error('CUDA solution contains NaN or Inf.');
end

%% 4.5 Verify both solutions with the trusted CPU K*U
fprintf('\n--- Verifying solutions with CPU K*U ---\n');

KU_cpu = Solving_KbyU_MatrixFree(U_cpu);
KU_gpu = Solving_KbyU_MatrixFree(U_gpu);

r_cpu = F_ - KU_cpu;
r_gpu = F_ - KU_gpu;

fixedDOFs = meshHierarchy_(1).fixedDOFs(:);

r_cpu(fixedDOFs) = 0;
r_gpu(fixedDOFs) = 0;

normF = norm(F_);

if normF == 0
    normF = 1;
end

cpuVerifiedRelres = norm(r_cpu) / normF;
gpuVerifiedRelres = norm(r_gpu) / normF;

%% 4.6 Compare CPU and GPU displacement vectors
absoluteDifference = norm(U_gpu - U_cpu);

relativeDifference = ...
    absoluteDifference / max(norm(U_cpu), eps);

maximumDifference = ...
    max(abs(U_gpu - U_cpu));

fprintf('\n--- Accuracy results ---\n');
fprintf('CPU verified relres:        %.6e\n', cpuVerifiedRelres);
fprintf('CUDA verified relres:       %.6e\n', gpuVerifiedRelres);
fprintf('CUDA internally reported:   %.6e\n', gpuReportedRelres);
fprintf('||U_gpu - U_cpu||:          %.6e\n', absoluteDifference);
fprintf('Relative solution difference: %.6e\n', relativeDifference);
fprintf('Maximum entry difference:   %.6e\n', maximumDifference);

%% 4.7 Runtime comparison
fprintf('\n--- Runtime results ---\n');
fprintf('CPU solve time:             %.6f s\n', cpuTime);
fprintf('CUDA end-to-end time:       %.6f s\n', gpuTime);
fprintf('Measured speedup:           %.3fx\n', cpuTime / gpuTime);

%% 4.8 Warnings
if gpuVerifiedRelres > max(10 * tol_, 1e-8)
    warning( ...
        ['CUDA verified residual %.6e is larger than ', ...
         'the requested tolerance %.6e.'], ...
        gpuVerifiedRelres, ...
        tol_);
end

if relativeDifference > 1e-5
    warning( ...
        ['CPU and CUDA solutions differ significantly. ', ...
         'Relative difference: %.6e.'], ...
        relativeDifference);
end

if abs(gpuReportedRelres - gpuVerifiedRelres) > ...
        max(1e-8, 1e-3 * gpuVerifiedRelres)
    warning( ...
        ['CUDA reported residual and externally verified ', ...
         'residual differ significantly.']);
end

% Use the GPU result for compliance and stress calculations below.
U_ = U_gpu;