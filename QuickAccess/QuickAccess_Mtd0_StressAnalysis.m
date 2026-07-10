%% DEMO: Stress Analysis
clearvars;
clc;

addpath('../');
addpath('../src/');
addpath('../src/MEXfuncs/');

Data_GlobalVariables;

global meshHierarchy_;
global weightFactorJacobi_;
global cholFac_;
global cholPermut_;
outPath_ = '../out/';
if ~exist(outPath_, 'dir')
    mkdir(outPath_);
end

%% 1. Data loading
tStart = tic;

MdlSelect = 'Bone';
IO_LoadBuiltInDatasets(MdlSelect);

fprintf('Prepare Voxel Model Costs: %10.3g s\n', toc(tStart));

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

tStart = tic;
Test_PCG_Vcycle_GPUCoarse();
disp(['CUDA Costs: ', sprintf('%10.3g',toc(tStart)) 's']);

% %% 4A. Test CUDA V-cycle against MATLAB V-cycle
% 
% global meshHierarchy_;
% global weightFactorJacobi_;
% 
% mh1 = meshHierarchy_(1);
% mh2 = meshHierarchy_(2);
% mh3 = meshHierarchy_(3);
% 
% rng(1);
% 
% r = randn(mh1.numDOFs, 1);
% r(mh1.fixedDOFs) = 0;
% 
% fixedMask1 = false(mh1.numDOFs,1);
% fixedMask1(mh1.fixedDOFs) = true;
% 
% fixedMask2 = false(mh2.numDOFs,1);
% fixedMask2(mh2.fixedDOFs) = true;
% 
% fixedMask3 = false(mh3.numDOFs,1);
% fixedMask3(mh3.fixedDOFs) = true;
% 
% fprintf('\nTesting standalone CUDA V-cycle...\n');
% 
% tStart = tic;
% 
% [z_cuda, r2_cuda, r3_cuda] = sgld_vcycle3_exact_cuda_mex_fixed( ...
%     full(double(r(:))), ...
%     mh2, ...
%     mh3, ...
%     full(double(mh1.diagK(:))), ...
%     fixedMask1, ...
%     full(double(mh2.diagK(:))), ...
%     fixedMask2, ...
%     fixedMask3, ...
%     weightFactorJacobi_, ...
%     @Debug_CoarseSolve);
% 
% fprintf('CUDA V-cycle cost: %.3f s\n', toc(tStart));
% 
% tStart = tic;
% z_matlab = Solving_Vcycle(r);
% fprintf('MATLAB V-cycle cost: %.3f s\n', toc(tStart));
% 
% fprintf('vcycle absErr: %.12e\n', norm(z_cuda - z_matlab));
% fprintf('vcycle relErr: %.12e\n', norm(z_cuda - z_matlab) / max(norm(z_matlab), 1e-30));
% fprintf('r''z CUDA:     %.12e\n', r' * z_cuda);
% fprintf('r''z MATLAB:   %.12e\n', r' * z_matlab);
% fprintf('r''z rel diff: %.12e\n', abs(r'*z_cuda - r'*z_matlab) / abs(r'*z_matlab));
% % Test_Vcycle3_Exact_CUDA();
% % mh1 = meshHierarchy_(1);
% % mh2 = meshHierarchy_(2);
% % mh3 = meshHierarchy_(3);
% % 
% % r = randn(mh1.numDOFs, 1);
% % r(mh1.fixedDOFs) = 0;
% % 
% % z_cuda = sgld_vcycle3_cuda_mex( ...
% %     full(double(r)), ...
% %     int32(mh1.nodMapBack), ...
% %     int32(mh2.nodMapForward), ...
% %     mh1.diagK, ...
% %     mh1.fixedDOFs, ...
% %     int32(mh2.nodMapBack), ...
% %     int32(mh3.nodMapForward), ...
% %     mh2.diagK, ...
% %     mh2.fixedDOFs, ...
% %     mh3.diagK, ...
% %     mh3.fixedDOFs, ...
% %     int32(mh3.nodeToElements), ...
% %     int32(mh3.eNodMat), ...
% %     mh3.eleModulus(:), ...
% %     mh3.Ks, ...
% %     mh1.resX, mh1.resY, mh1.resZ, ...
% %     mh2.resX, mh2.resY, mh2.resZ, ...
% %     mh3.resX, mh3.resY, mh3.resZ, ...
% %     4, ...                    % level 1 -> 2
% %     2, ...                    % level 2 -> 3
% %     weightFactorJacobi_, ...
% %     20 ...                    % coarse Jacobi iterations on level 3
% % );
% % z_matlab = Solving_Vcycle(r);
% fprintf('\nStarting PCG with MATLAB KbyU + CUDA V-cycle...\n');
% 
% tStart = tic;
% 
% [U_cuda_vcycle, its_cuda_vcycle] = ...
%     Solving_PreconditionedConjugateGradientSolver( ...
%         @Solving_KbyU_MatrixFree, ...
%         @KbyR_CUDA_Vcycle, ...
%         F_, ...
%         tol_, ...
%         maxIT_, ...
%         'printP_ON');
% 
% fprintf('CUDA V-cycle PCG completed.\n');
% fprintf('Iterations: %d\n', its_cuda_vcycle);
% fprintf('CUDA V-cycle PCG cost: %.3f s\n', toc(tStart));
% % %% 4. Solve using CUDA KbyU
% % fprintf('Starting CUDA PCG solver...\n');
% % 
% % KbyU_cuda = @(U) KbyU_cuda_wrapper(U);
% % 
% % tStart = tic;
% % 
% % [U_cuda, its_cuda] = ...
% %     Solving_PreconditionedConjugateGradientSolver( ...
% %         KbyU_cuda, ...
% %         @Solving_Vcycle, ...
% %         F_, ...
% %         tol_, ...
% %         maxIT_, ...
% %         'printP_ON');
% % 
% % fprintf('CUDA PCG completed.\n');
% % fprintf('Iterations: %d\n', its_cuda);
% % fprintf('CUDA linear-system solver cost: %10.3g s\n', toc(tStart));
% % 
% % U_ = U_cuda;
% % 
% % clear U_cuda;
% mh = meshHierarchy_(1);
% 
% [U_cuda, its_cuda, relres_cuda] = pcg_cuda_jacobi_mex( ...
%     full(double(F_)), ...
%     [], ...                              % U0, [] means zero initial guess
%     int32(mh.nodeToElements), ...
%     int32(mh.eNodMat), ...
%     mh.eleModulus(:), ...
%     mh.Ks, ...                           % IMPORTANT: use Ks, not Ke
%     mh.diagK, ...
%     mh.fixedDOFs, ...
%     tol_, ...
%     maxIT_, ...
%     mh.resX, ...
%     mh.resY, ...
%     mh.resZ, ...
%     'printP_ON' ...
% );
%% 4. Solving FEA Linear System via Conjugate Gradien Method
tStart = tic;
U_ = Solving_PreconditionedConjugateGradientSolver(@Solving_KbyU_MatrixFree, @Solving_Vcycle, F_, tol_, maxIT_, 'printP_ON');
disp(['Liner System Solver Costs: ', sprintf('%10.3g',toc(tStart)) 's']);

%% 5. Compute compliance
tStart = tic;

ceList = TopOpti_ComputeUnitCompliance();
c = meshHierarchy_(1).eleModulus * ceList;

fprintf('Compliance in total: %10.5e\n', c);
fprintf('Compute Compliance Costs: %10.3g s\n', toc(tStart));

%% 6. Compute stress field
tStart = tic;

[cartesianStressField_, vonMisesStressField_] = FEA_StressAnalysis();

fprintf('Compute Stress Field Costs: %10.3g s\n', toc(tStart));

function y = KbyU_cuda_wrapper(U)
    global meshHierarchy_;

    mh = meshHierarchy_(1);

    y = pcg_cuda_mex_2( ...
        full(double(U)), ...
        int32(mh.nodeToElements), ...
        int32(mh.eNodMat), ...
        mh.eleModulus(:), ...
        mh.Ks, ...                 % IMPORTANT: use Ks, not Ke
        mh.resX, ...
        mh.resY, ...
        mh.resZ ...
    );

    % IMPORTANT: match Solving_KbyU_MatrixFree behavior
    y(mh.fixedDOFs, 1) = 0;
end

function y = KbyU_cpu_mex_wrapper(U)
    global meshHierarchy_;

    mh = meshHierarchy_(1);

    y = Solving_KbyU_MatrixFree_mex( ...
        full(double(U)), ...
        int32(mh.eNodMat), ...
        mh.Ks, ...                 % IMPORTANT: use Ks
        mh.eleModulus(:), ...
        mh.colors ...
    );

    y(mh.fixedDOFs, 1) = 0;         % IMPORTANT
end