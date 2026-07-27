%% DEMO: Test Bone at resolutions 512 and 1200

clear all;
clc;

addpath('../');
addpath('../src/');
addpath('../src/MEXfuncs/');

resolutions = [512, 1200];

numTests = numel(resolutions);

resolutionResults = zeros(numTests, 1);
numElementsResults = zeros(numTests, 1);
numDOFsResults = zeros(numTests, 1);
gpuIterationsResults = zeros(numTests, 1);

voxelTimeResults = zeros(numTests, 1);
setupTimeResults = zeros(numTests, 1);
stencilTimeResults = zeros(numTests, 1);
gpuTimeResults = zeros(numTests, 1);
cpuTimeResults = zeros(numTests, 1);
speedupResults = zeros(numTests, 1);

gpuReportedRelresResults = zeros(numTests, 1);
gpuVerifiedRelresResults = zeros(numTests, 1);
cpuVerifiedRelresResults = zeros(numTests, 1);
relativeDifferenceResults = zeros(numTests, 1);
maximumDifferenceResults = zeros(numTests, 1);

for testIndex = 1:numTests

    resolution = resolutions(testIndex);

    fprintf('\n\n');
    fprintf('############################################################\n');
    fprintf('Testing Bone at resolution %d\n', resolution);
    fprintf('Test %d of %d\n', testIndex, numTests);
    fprintf('############################################################\n');

    % Reset all data from the previous resolution.
    clear global;

    Data_GlobalVariables;

    global meshHierarchy_;
    global F_;
    global U_;
    global tol_;
    global maxIT_;

    outPath_ = '../out/';

    if ~exist(outPath_, 'dir')
        mkdir(outPath_);
    end

    %% 1. Data loading

    fprintf('\n--- Loading Bone R%d ---\n', resolution);

    tStart = tic;

    IO_LoadBuiltInDatasets('Bone', resolution);

    voxelTime = toc(tStart);

    fprintf('Prepare Voxel Model Costs: %10.3g s\n', voxelTime);

    %% 2. Setup FEA

    fprintf('\n--- Setting up FEA ---\n');

    tStart = tic;

    FEA_ApplyBoundaryCondition();
    FEA_SetupVoxelBased();

    densityField = ones( ...
        meshHierarchy_(1).numElements, ...
        1);

    meshHierarchy_(1).eleModulus = ...
        TopOpti_MaterialInterpolationSIMP( ...
            densityField(:));

    setupTime = toc(tStart);

    fprintf('Setup FEA Costs: %10.3g s\n', setupTime);

    clear densityField;

    %% 3. Assemble computing stencil

    fprintf('\n--- Assembling computing stencil ---\n');

    tStart = tic;

    Solving_AssembleFEAstencil();

    stencilTime = toc(tStart);

    fprintf( ...
        'Assemble Computing Stencil Costs: %10.3g s\n', ...
        stencilTime);

    %% 4. Build GPU hierarchy

    fprintf('\n====================================================\n');
    fprintf('CPU vs CUDA MGPCG test: Bone R%d\n', resolution);
    fprintf('====================================================\n');

    H_ = Build_GPU_Hierarchy(meshHierarchy_);

    U0 = zeros(size(F_), 'double');

    fprintf('\n--- GPU hierarchy information ---\n');
    fprintf('Resolution:             %d\n', resolution);
    fprintf('Number of levels:       %d\n', H_.numLevels);
    fprintf('Finest-level elements:  %d\n', H_.numElements(1));
    fprintf('Finest-level nodes:     %d\n', H_.numNodes(1));
    fprintf('Finest-level DOFs:      %d\n', H_.numDOFs(1));
    fprintf('Coarsest-level DOFs:    %d\n', H_.numDOFs(end));
    fprintf('Coarsest free DOFs:     %d\n', ...
        numel(H_.coarseFreeDOFIds));
    fprintf('Coarse matrix size:     %d x %d\n', ...
        size(H_.coarseKFree, 1), ...
        size(H_.coarseKFree, 2));
    fprintf('Coarse matrix nnz:      %d\n', ...
        nnz(H_.coarseKFree));
    fprintf('Jacobi omega:           %.6f\n', ...
        H_.jacobiOmega);

    assert( ...
        issparse(H_.coarseKFree), ...
        'H_.coarseKFree must remain sparse.');

    assert( ...
        size(H_.coarseKFree, 1) == ...
        numel(H_.coarseFreeDOFIds), ...
        'coarseKFree size does not match coarseFreeDOFIds.');

    assert( ...
        all(isfinite(nonzeros(H_.coarseKFree))), ...
        'coarseKFree contains NaN or Inf.');

    %% 5. Prepare CUDA inputs

    mexPath = which('Solving_MGPCG_GPU_fixed');

    if isempty(mexPath)
        error( ...
            ['Solving_MGPCG_GPU_fixed was not found. ', ...
             'Compile the CUDA MEX and add its directory ', ...
             'to the MATLAB path.']);
    end

    F_ = full(double(F_(:)));
    U0 = full(double(U0(:)));

    for level = 1:H_.numLevels - 1
        H_.diagK{level} = ...
            full(double(H_.diagK{level}(:)));
    end

    H_.eleModulus{1} = ...
        full(double(H_.eleModulus{1}(:)));

    H_.Ke = full(double(H_.Ke));

    fprintf('\nUsing CUDA MEX:\n%s\n', mexPath);

    %% 6. CUDA solve

    fprintf('\n--- Running CUDA MGPCG ---\n');

    gpuTimer = tic;

    [U_gpu, gpuIterations, gpuReportedRelres] = ...
        Solving_MGPCG_GPU_fixed( ...
            F_, ...
            tol_, ...
            maxIT_, ...
            U0, ...
            H_);

    gpuTime = toc(gpuTimer);

    fprintf('CUDA end-to-end time: %.6f s\n', gpuTime);
    fprintf('CUDA iterations:      %d\n', gpuIterations);
    fprintf('CUDA reported relres: %.6e\n', ...
        gpuReportedRelres);

    %% 7. CPU solve

    fprintf('\n--- Running CPU MGPCG ---\n');

    cpuTimer = tic;

    U_cpu = ...
        Solving_PreconditionedConjugateGradientSolver( ...
            @Solving_KbyU_MatrixFree, ...
            @Solving_Vcycle, ...
            F_, ...
            tol_, ...
            maxIT_, ...
            'printP_ON', ...
            U0);

    cpuTime = toc(cpuTimer);

    fprintf('CPU solve time: %.6f s\n', cpuTime);

    %% 8. Basic solution checks

    if any(~isfinite(U_cpu))
        error( ...
            'CPU solution for Bone R%d contains NaN or Inf.', ...
            resolution);
    end

    if any(~isfinite(U_gpu))
        error( ...
            'CUDA solution for Bone R%d contains NaN or Inf.', ...
            resolution);
    end

    %% 9. Verify solutions with trusted CPU K*U

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

    %% 10. Compare displacement vectors

    absoluteDifference = norm(U_gpu - U_cpu);

    relativeDifference = ...
        absoluteDifference / max(norm(U_cpu), eps);

    maximumDifference = ...
        max(abs(U_gpu - U_cpu));

    fprintf('\n--- Accuracy results: Bone R%d ---\n', resolution);
    fprintf('CPU verified relres:          %.6e\n', ...
        cpuVerifiedRelres);
    fprintf('CUDA verified relres:         %.6e\n', ...
        gpuVerifiedRelres);
    fprintf('CUDA internally reported:     %.6e\n', ...
        gpuReportedRelres);
    fprintf('||U_gpu - U_cpu||:            %.6e\n', ...
        absoluteDifference);
    fprintf('Relative solution difference: %.6e\n', ...
        relativeDifference);
    fprintf('Maximum entry difference:     %.6e\n', ...
        maximumDifference);

    %% 11. Runtime comparison

    speedup = cpuTime / gpuTime;

    fprintf('\n--- Runtime results: Bone R%d ---\n', resolution);
    fprintf('CPU solve time:       %.6f s\n', cpuTime);
    fprintf('CUDA end-to-end time: %.6f s\n', gpuTime);
    fprintf('Measured speedup:     %.3fx\n', speedup);

    %% 12. Warnings

    if gpuVerifiedRelres > max(10 * tol_, 1e-8)
        warning( ...
            ['Bone R%d: CUDA verified residual %.6e is ', ...
             'larger than the requested tolerance %.6e.'], ...
            resolution, ...
            gpuVerifiedRelres, ...
            tol_);
    end

    if relativeDifference > 1e-5
        warning( ...
            ['Bone R%d: CPU and CUDA solutions differ ', ...
             'significantly. Relative difference: %.6e.'], ...
            resolution, ...
            relativeDifference);
    end

    if abs(gpuReportedRelres - gpuVerifiedRelres) > ...
            max(1e-8, 1e-3 * gpuVerifiedRelres)

        warning( ...
            ['Bone R%d: CUDA reported residual and ', ...
             'externally verified residual differ ', ...
             'significantly.'], ...
            resolution);
    end

    U_ = U_gpu;

    %% 13. Store results

    resolutionResults(testIndex) = resolution;
    numElementsResults(testIndex) = ...
        double(H_.numElements(1));
    numDOFsResults(testIndex) = ...
        double(H_.numDOFs(1));
    gpuIterationsResults(testIndex) = ...
        double(gpuIterations);

    voxelTimeResults(testIndex) = voxelTime;
    setupTimeResults(testIndex) = setupTime;
    stencilTimeResults(testIndex) = stencilTime;
    gpuTimeResults(testIndex) = gpuTime;
    cpuTimeResults(testIndex) = cpuTime;
    speedupResults(testIndex) = speedup;

    gpuReportedRelresResults(testIndex) = ...
        gpuReportedRelres;
    gpuVerifiedRelresResults(testIndex) = ...
        gpuVerifiedRelres;
    cpuVerifiedRelresResults(testIndex) = ...
        cpuVerifiedRelres;
    relativeDifferenceResults(testIndex) = ...
        relativeDifference;
    maximumDifferenceResults(testIndex) = ...
        maximumDifference;

    clear H_;
    clear U0;
    clear U_cpu;
    clear U_gpu;
    clear KU_cpu;
    clear KU_gpu;
    clear r_cpu;
    clear r_gpu;
end

%% 14. Final result table

results = table( ...
    resolutionResults, ...
    numElementsResults, ...
    numDOFsResults, ...
    gpuIterationsResults, ...
    voxelTimeResults, ...
    setupTimeResults, ...
    stencilTimeResults, ...
    gpuTimeResults, ...
    cpuTimeResults, ...
    speedupResults, ...
    gpuReportedRelresResults, ...
    gpuVerifiedRelresResults, ...
    cpuVerifiedRelresResults, ...
    relativeDifferenceResults, ...
    maximumDifferenceResults, ...
    'VariableNames', { ...
        'Resolution', ...
        'NumElements', ...
        'NumDOFs', ...
        'GPUIterations', ...
        'VoxelTimeSeconds', ...
        'SetupTimeSeconds', ...
        'StencilTimeSeconds', ...
        'GPUTimeSeconds', ...
        'CPUTimeSeconds', ...
        'Speedup', ...
        'GPUReportedRelres', ...
        'GPUVerifiedRelres', ...
        'CPUVerifiedRelres', ...
        'RelativeSolutionDifference', ...
        'MaximumEntryDifference'});

fprintf('\n\n');
fprintf('============================================================\n');
fprintf('FINAL BONE BENCHMARK RESULTS\n');
fprintf('============================================================\n');

disp(results);

save( ...
    fullfile(outPath_, 'Bone_512_1200_Results.mat'), ...
    'results');

writetable( ...
    results, ...
    fullfile(outPath_, 'Bone_512_1200_Results.csv'));

fprintf('\nResults saved to:\n');
fprintf('%s\n', ...
    fullfile(outPath_, 'Bone_512_1200_Results.mat'));
fprintf('%s\n', ...
    fullfile(outPath_, 'Bone_512_1200_Results.csv'));

