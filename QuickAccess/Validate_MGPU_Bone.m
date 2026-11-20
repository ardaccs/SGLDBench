%% VALIDATION: Multi-GPU CUDA MGPCG vs CPU, Bone R512 and R1200

clearvars;
clc;

addpath('../');
addpath('../src/');
addpath('../src/MEXfuncs/');

%% Multi-GPU configuration
% Compile the multi-GPU CUDA source with this MEX name, or change it here.
mgpuMexName = 'Solving_MGPCG_MGPU_local';
numGPUs = 2;            % Number of GPUs / partitions to validate (>= 2).
assert(numGPUs >= 2 && numGPUs == floor(numGPUs), ...
    'For multi-GPU validation, numGPUs must be an integer >= 2.');

% Required solver values are read from meshHierarchy_(1) or
% meshHierarchy_(end) by Build_GPU_Hierarchy, unless overridden here.
% Fill these with YOUR assembled values if they are not stored on the mesh:
%   solverData.Ke               = yourKe24x24;
%   solverData.jacobiOmega      = yourJacobiOmega;
%   solverData.coarseKFree      = yourSparseCoarseKFree;
%   solverData.coarseFreeDOFIds = int32(yourFreeCoarseDOFIds);
%   solverData.fixedDOFIds      = int32(yourGlobalFixedDOFIds);
solverData = struct();

% Report shared-node differences in the final MGPU solution.
% This is a relative infinity-norm tolerance; also consider the solver tol_.
sharedSolutionTolerance = 1e-8;

resolutions = [512, 1200];

numTests = numel(resolutions);

resolutionResults = zeros(numTests, 1);
numElementsResults = zeros(numTests, 1);
numDOFsResults = zeros(numTests, 1);
gpuIterationsResults = zeros(numTests, 1);
numGPUsResults = repmat(numGPUs, numTests, 1);
hierarchyTimeResults = zeros(numTests, 1);
sharedMismatchResults = nan(numTests, 1);

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
    global weightFactorJacobi_;

    outPath_ = '../out/';

    if ~exist(outPath_, 'dir')
        mkdir(outPath_);
    end

    %% 1. Data loading

    fprintf('\n--- Loading Bone R%d ---\n', resolution);

    tStart = tic;

    IO_LoadBuiltInDatasets('Bone', resolution, numGPUs);

    voxelTime = toc(tStart);

    fprintf('Prepare Voxel Model Costs: %10.3g s\n', voxelTime);

    %% 2. Setup FEA

    fprintf('\n--- Setting up FEA ---\n');

    tStart = tic;

    FEA_ApplyBoundaryCondition();
    FEA_SetupVoxelBased(numGPUs);

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

    %% 4. Build and validate the MULTI-GPU hierarchy

    fprintf('\n====================================================\n');
    fprintf('CPU vs CUDA MULTI-GPU MGPCG: Bone R%d, %d GPUs\n', ...
        resolution, numGPUs);
    fprintf('====================================================\n');

    % The hierarchy builder does NOT partition an unpartitioned mesh.
    % Every multigrid level must already have numGPUs partition entries.
    for level = 1:numel(meshHierarchy_)
        if ~isfield(meshHierarchy_(level), 'partitions') || ...
                ~iscell(meshHierarchy_(level).partitions) || ...
                numel(meshHierarchy_(level).partitions) ~= numGPUs
            error(['meshHierarchy_(%d).partitions must be a cell array ' ...
                   'with %d entries. Partition every multigrid level ' ...
                   'before running the MGPU validation.'], level, numGPUs);
        end
    end

    % The original CPU validation uses fixedDOFs (not fixedDOFIds).
    % Keep those same constraints when constructing the MGPU partitions.
    solverDataThisTest = solverData;
    if (~isfield(solverDataThisTest, 'fixedDOFIds') || ...
            isempty(solverDataThisTest.fixedDOFIds)) && ...
            isfield(meshHierarchy_(1), 'fixedDOFs')
        solverDataThisTest.fixedDOFIds = ...
            int32(meshHierarchy_(1).fixedDOFs(:));
    end

    hierarchyTimer = tic;
    fprintf('\n--- Checking multigrid span widths ---\n');
    coarse = meshHierarchy_(end);

    fprintf('\n--- Coarsest matrix diagnostics ---\n');
    fprintf('Total coarse DOFs: %d\n', coarse.numDOFs);
    fprintf('Number of free DOFs: %d\n', numel(coarse.freeDOFs));
    fprintf('Kfree dimensions: %d x %d\n', ...
        size(coarse.Kfree, 1), size(coarse.Kfree, 2));
    fprintf('Solver free DOF IDs: %d\n', ...
        numel(solverDataThisTest.coarseFreeDOFIds));
    fprintf('Solver matrix dimensions: %d x %d\n', ...
        size(solverDataThisTest.coarseKFree, 1), ...
        size(solverDataThisTest.coarseKFree, 2));

    fprintf('Free DOF range: %d to %d\n', ...
        min(coarse.freeDOFs), max(coarse.freeDOFs));
    % Convert destination-level spanWidth to source-level convention.
    for level = 1:numel(meshHierarchy_)-1
        meshHierarchy_(level).spanWidth = ...
            meshHierarchy_(level+1).spanWidth;
    end

    % Match the CPU Solving_Vcycle Jacobi damping factor exactly.
    % Data_GlobalVariables initializes weightFactorJacobi_ as a global.
    assert(isnumeric(weightFactorJacobi_) && isreal(weightFactorJacobi_) && ...
           isscalar(weightFactorJacobi_) && isfinite(weightFactorJacobi_) && ...
           weightFactorJacobi_ > 0, ...
           'Global weightFactorJacobi_ is not initialized correctly.');
    solverDataThisTest.jacobiOmega = double(weightFactorJacobi_);
    fprintf('Using CPU Jacobi omega = %.6f\n', solverDataThisTest.jacobiOmega);
    %% Coarsest-level sparse stiffness matrix for cuDSS

    coarse = meshHierarchy_(end);

    assert(isfield(coarse, 'Kfree') && ~isempty(coarse.Kfree), ...
        'Coarsest Kfree was not assembled.');

    assert(isfield(coarse, 'freeDOFs') && ~isempty(coarse.freeDOFs), ...
        'Coarsest freeDOFs are missing.');

    % Reuse exactly the matrix assembled for the CPU Cholesky solver.
    solverDataThisTest.coarseKFree = sparse( ...
        double(coarse.Kfree));

    solverDataThisTest.coarseFreeDOFIds = int32( ...
        coarse.freeDOFs(:));

    % Validate matrix dimensions.
    nFree = numel(solverDataThisTest.coarseFreeDOFIds);

    assert(isequal(size(solverDataThisTest.coarseKFree), ...
        [nFree, nFree]), ...
        'Coarsest matrix dimensions do not match free DOFs.');

    fprintf('Coarsest free DOFs: %d\n', nFree);
    fprintf('Coarsest matrix NNZ: %d\n', ...
        nnz(solverDataThisTest.coarseKFree));
    H_ = Build_GPU_Hierarchy_MGPU_2( ...
        meshHierarchy_, numGPUs, solverDataThisTest);
    hierarchyTime = toc(hierarchyTimer);

    % MATLAB sparse vectors can still be class 'double'. MEX code uses
    % mxGetData as if these were DENSE arrays; make that explicit on all GPUs.
    H_.Ke = full(double(H_.Ke));
    for g = 1:numGPUs
        P = H_.partitions{g};
        P.eleModulus{1} = full(double(P.eleModulus{1}(:)));
        for level = 1:numel(meshHierarchy_)-1
            P.diagK{level} = full(double(P.diagK{level}(:)));
        end
        H_.partitions{g} = P;
    end

    validateMGPUHierarchy(H_, numGPUs, numel(meshHierarchy_), ...
        numel(meshHierarchy_(1).nodGridId));

    numLevels = numel(meshHierarchy_);
    globalFinestNodes = numel(meshHierarchy_(1).nodGridId);
    globalFinestDOFs = 3 * globalFinestNodes;
    globalCoarsestNodes = double(H_.globalCoarseNumNodes);

    assert(numel(F_) == globalFinestDOFs, ...
        'Global F_ must have 3 DOFs per active finest-level node.');

    U0 = zeros(size(F_), 'double');

    fprintf('\n--- Multi-GPU hierarchy information ---\n');
    fprintf('Resolution:                 %d\n', resolution);
    fprintf('GPUs:                       %d\n', numGPUs);
    fprintf('Multigrid levels:           %d\n', numLevels);
    fprintf('Global finest-level nodes:  %d\n', globalFinestNodes);
    fprintf('Global finest-level DOFs:   %d\n', globalFinestDOFs);
    fprintf('Global coarsest-level DOFs: %d\n', 3*globalCoarsestNodes);
    fprintf('Coarsest free DOFs:         %d\n', numel(H_.coarseFreeDOFIds));
    fprintf('Coarse matrix size:         %d x %d\n', ...
        size(H_.coarseKFree,1), size(H_.coarseKFree,2));
    fprintf('Coarse matrix nnz:          %d\n', nnz(H_.coarseKFree));
    fprintf('Jacobi omega:               %.6f\n', H_.jacobiOmega);
    fprintf('Hierarchy build time:       %.6f s\n', hierarchyTime);

    fprintf('\n  GPU      NormalNodes      Elements      HaloNodes      OwnedNodes\n');
    for g = 1:numGPUs
        P = H_.partitions{g};
        fprintf('  %3d      %11d      %8d      %9d      %10d\n', ...
            g, double(P.numNodes(1)), double(P.numElements(1)), ...
            double(P.numBufferNodes(1)), nnz(P.ownedNodeMask{1}));
    end

    assert(issparse(H_.coarseKFree), ...
        'H_.coarseKFree must remain sparse.');
    assert(size(H_.coarseKFree,1) == numel(H_.coarseFreeDOFIds) && ...
           size(H_.coarseKFree,2) == numel(H_.coarseFreeDOFIds), ...
        'coarseKFree size does not match coarseFreeDOFIds.');
    assert(all(isfinite(nonzeros(H_.coarseKFree))), ...
        'coarseKFree contains NaN or Inf.');

    %% 5. Prepare per-GPU vectors and validate MEX availability

    mexPath = which(mgpuMexName);
    if isempty(mexPath)
        error(['Multi-GPU MEX ''%s'' was not found. Compile the ' ...
               'multi-GPU CUDA solver under this name and add it to the path.'], ...
               mgpuMexName);
    end
    if ~endsWith(mexPath, ['.' mexext])
        error('MEX %s resolves to %s, not a compiled MEX binary.', ...
            mgpuMexName, mexPath);
    end

    F_ = full(double(F_(:)));
    U0 = full(double(U0(:)));
    [bGPU, y0GPU] = Split_GPU_Vectors(F_, U0, H_);

    for g = 1:numGPUs
        nLocalDOFs = double(H_.partitions{g}.numDOFs(1));
        assert(isa(bGPU{g}, 'double') && iscolumn(bGPU{g}) && ...
               numel(bGPU{g}) == nLocalDOFs && ...
               numel(y0GPU{g}) == nLocalDOFs, ...
            'Invalid split input vector dimensions for GPU %d.', g);
    end

    fprintf('\nUsing multi-GPU CUDA MEX:\n%s\n', mexPath);

    %% 6. Multi-GPU CUDA solve and global reconstruction

    fprintf('\n--- Running CUDA MGPCG (%d GPUs) ---\n', numGPUs);
    gpuTimer = tic;

    % The integrated MGPU MEX must accept local b/y0 cell arrays and H.
    % Preferred first output: one local DOF vector per GPU (cell array).
    % A fully assembled global DOF vector is also accepted by this validator.
    [U_gpu_raw, gpuIterations, gpuReportedRelres] = feval( ...
        mgpuMexName, bGPU, tol_, maxIT_, y0GPU, H_);

    gpuTime = toc(gpuTimer);

    assert(isnumeric(gpuIterations) && isscalar(gpuIterations) && ...
           isfinite(gpuIterations) && gpuIterations >= 0 && ...
           gpuIterations == floor(gpuIterations), ...
        'MGPU iteration count must be a nonnegative integer scalar.');
    assert(isnumeric(gpuReportedRelres) && isscalar(gpuReportedRelres) && ...
           isreal(gpuReportedRelres) && isfinite(gpuReportedRelres) && ...
           gpuReportedRelres >= 0, ...
        'MGPU reported residual must be a finite nonnegative scalar.');

    [U_gpu, sharedMismatch] = assembleMGPUOutput( ...
        U_gpu_raw, H_, globalFinestDOFs);

    fprintf('CUDA end-to-end time:    %.6f s\n', gpuTime);
    fprintf('CUDA iterations:         %d\n', gpuIterations);
    fprintf('CUDA reported relres:    %.6e\n', gpuReportedRelres);
    if isfinite(sharedMismatch)
        fprintf('Interface disagreement:  %.6e (relative max)\n', ...
            sharedMismatch);
        if sharedMismatch > max(sharedSolutionTolerance, 10*tol_)
            warning('Bone R%d: shared GPU solutions differ by %.6e.', ...
                resolution, sharedMismatch);
        end
    else
        fprintf('Interface disagreement:  N/A (MEX returned global vector)\n');
    end

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
        double(meshHierarchy_(1).numElements);
    numDOFsResults(testIndex) = globalFinestDOFs;
    gpuIterationsResults(testIndex) = ...
        double(gpuIterations);
    hierarchyTimeResults(testIndex) = hierarchyTime;
    sharedMismatchResults(testIndex) = sharedMismatch;

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
    clear bGPU;
    clear y0GPU;
    clear U_gpu_raw;
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
    numGPUsResults, ...
    hierarchyTimeResults, ...
    sharedMismatchResults, ...
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
        'NumGPUs', ...
        'HierarchyTimeSeconds', ...
        'SharedNodeRelativeMismatch', ...
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
fprintf('FINAL MULTI-GPU BONE BENCHMARK RESULTS\n');
fprintf('============================================================\n');

disp(results);

save( ...
    fullfile(outPath_, 'Bone_512_1200_MGPU_Results.mat'), ...
    'results');

writetable( ...
    results, ...
    fullfile(outPath_, 'Bone_512_1200_MGPU_Results.csv'));

fprintf('\nResults saved to:\n');
fprintf('%s\n', ...
    fullfile(outPath_, 'Bone_512_1200_MGPU_Results.mat'));
fprintf('%s\n', ...
    fullfile(outPath_, 'Bone_512_1200_MGPU_Results.csv'));


%% Local validation functions (MATLAB R2016b or newer supports script functions).

function validateMGPUHierarchy(H, numGPUs, numLevels, globalFinestNodes)
% Validate information that both the local-vector splitter and the current
% multi-GPU CUDA hierarchy parser rely upon.
    assert(isstruct(H) && isscalar(H) && ...
           isfield(H,'partitions') && iscell(H.partitions) && ...
           numel(H.partitions)==numGPUs, ...
        'MGPU hierarchy must have H.partitions{1:numGPUs}.');

    allOwned = [];
    allCoarse = [];
    for g = 1:numGPUs
        P = H.partitions{g};
        assert(double(P.numLevels)==numLevels, ...
            'Incorrect number of levels on GPU %d.',g);
        assert(numel(P.globalNodeIds)==numLevels, ...
            'Missing level-global node IDs on GPU %d.',g);
        assert(numel(P.ownedNodeMask)==numLevels, ...
            'Missing ownedNodeMask on GPU %d.',g);
        assert(numel(P.numBufferNodes)==numLevels, ...
            'Missing halo counts on GPU %d.',g);

        for level = 1:numLevels
            ids = double(P.globalNodeIds{level}(:));
            n = double(P.numNodes(level));
            assert(numel(ids)==n && numel(unique(ids))==n, ...
                'GPU %d level %d normal node IDs are invalid.',g,level);
            assert(numel(P.nodGridId{level})==n, ...
                'GPU %d level %d nodGridId size mismatch.',g,level);
            assert(numel(P.ownedNodeMask{level})==n, ...
                'GPU %d level %d owned mask size mismatch.',g,level);
            assert(double(P.numDOFs(level)) == 3*n, ...
                'GPU %d level %d DOF count mismatch.',g,level);
            assert(all(P.nodMapForward{level}(:)>=0) && ...
                   all(double(P.nodMapForward{level}(:)) <= ...
                       n + double(P.numBufferNodes(level))), ...
                'GPU %d level %d node mapping exceeds halo storage.',g,level);
        end

        fineIds = double(P.globalNodeIds{1}(:));
        owned = logical(P.ownedNodeMask{1}(:));
        allOwned = [allOwned; fineIds(owned)]; %#ok<AGROW>
        allCoarse = [allCoarse; double(P.globalNodeIds{end}(:))]; %#ok<AGROW>
    end

    assert(all(allOwned>=1 & allOwned<=globalFinestNodes), ...
        'Finest ownership references out-of-range global nodes.');
    counts = accumarray(allOwned, 1, [globalFinestNodes,1]);
    assert(all(counts==1), ...
        'Every global finest-level node must have exactly one GPU owner.');
    nCoarse = double(H.globalCoarseNumNodes);
    assert(all(allCoarse>=1 & allCoarse<=nCoarse), ...
        'Coarsest ownership references out-of-range global nodes.');
    counts = accumarray(allCoarse, 1, [nCoarse,1]);
    assert(all(counts==1), ...
        'Every global coarse node must occur in exactly one GPU partition.');

    for g = 1:numGPUs-1
        a = H.partitions{g};
        b = H.partitions{g+1};
        assert(isequal(a.rightInterfaceSendLocal(:), ...
                       b.leftInterfaceSendDestLocal(:)) && ...
               isequal(a.rightInterfaceSendDestLocal(:), ...
                       b.leftInterfaceSendLocal(:)), ...
            'Finest right/left interface mappings mismatch at GPUs %d/%d.', ...
            g,g+1);
        for level = 1:numLevels
            assert(numel(a.rightHaloSendLocal{level}) == ...
                   numel(a.rightHaloSendDestLocal{level}) && ...
                   numel(b.leftHaloSendLocal{level}) == ...
                   numel(b.leftHaloSendDestLocal{level}), ...
                'Halo send-index/destination-index lengths disagree.');
        end
    end

    % Receiver halo destinations must occupy every appended storage slot
    % exactly once: [normal nodes | incoming halo nodes].
    for level = 1:numLevels
        for g = 1:numGPUs
            nNormal = double(H.partitions{g}.numNodes(level));
            nBuffer = double(H.partitions{g}.numBufferNodes(level));
            incomingDest = zeros(0,1);
            if g>1
                L = H.partitions{g-1};
                send = double(L.rightHaloSendLocal{level}(:));
                dest = double(L.rightHaloSendDestLocal{level}(:));
                assert(numel(send)==numel(dest) && ...
                       all(send>=1 & send<=double(L.numNodes(level))), ...
                    'GPU %d right halo indices are invalid at level %d.',g-1,level);
                incomingDest = [incomingDest; dest]; %#ok<AGROW>
            end
            if g<numGPUs
                R = H.partitions{g+1};
                send = double(R.leftHaloSendLocal{level}(:));
                dest = double(R.leftHaloSendDestLocal{level}(:));
                assert(numel(send)==numel(dest) && ...
                       all(send>=1 & send<=double(R.numNodes(level))), ...
                    'GPU %d left halo indices are invalid at level %d.',g+1,level);
                incomingDest = [incomingDest; dest]; %#ok<AGROW>
            end
            expectedDest = (nNormal + (1:nBuffer))';
            assert(isequal(sort(incomingDest), expectedDest), ...
                'GPU %d level %d halo destinations do not fill appended slots.', ...
                g, level);
        end
    end
end

function [U, maxSharedRelativeMismatch] = assembleMGPUOutput(Uraw, H, globalDOFs)
% Return a full global displacement vector using the explicit finest-level
% ownedNodeMask. Replicated interface nodes are compared against their owner.
% No halo/buffer entries should be returned in the local solution vectors.
    assert(mod(globalDOFs,3)==0, 'Expected 3 DOFs per global node.');

    if isnumeric(Uraw) && ~iscell(Uraw)
        assert(isa(Uraw,'double') && isreal(Uraw) && ...
               numel(Uraw)==globalDOFs, ...
            'Global MGPU output must be a real double vector of length %d.', ...
            globalDOFs);
        U = full(Uraw(:));
        maxSharedRelativeMismatch = NaN; % Nothing local to cross-check.
        return;
    end

    assert(iscell(Uraw) && numel(Uraw)==numel(H.partitions), ...
        ['MGPU output must be either a global real double vector or ' ...
         'a cell array of one normal-local DOF vector per GPU.']);

    nGlobalNodes = globalDOFs/3;
    uByNode = zeros(3,nGlobalNodes);
    ownershipCount = zeros(nGlobalNodes,1);

    % First pass: place exactly one value for every global active node.
    for g = 1:numel(H.partitions)
        P = H.partitions{g};
        ids = double(P.globalNodeIds{1}(:));
        local = Uraw{g};
        assert(isa(local,'double') && isreal(local) && ...
               numel(local)==3*numel(ids), ...
            'MGPU output for GPU %d has incorrect type or normal DOF count.',g);
        local = reshape(full(local(:)),3,[]);
        owner = logical(P.ownedNodeMask{1}(:));
        assert(numel(owner)==numel(ids) && ...
               all(ids>=1 & ids<=nGlobalNodes), ...
            'Invalid global IDs or owner mask on GPU %d.',g);
        ownedIds = ids(owner);
        uByNode(:,ownedIds) = local(:,owner);
        ownershipCount(ownedIds) = ownershipCount(ownedIds)+1;
    end
    assert(all(ownershipCount==1), ...
        'The MGPU output cannot be reconstructed: missing/duplicate ownership.');

    % Second pass: test agreement of non-owner (duplicated/shared) nodes.
    discrepancy = 0;
    for g = 1:numel(H.partitions)
        ids = double(H.partitions{g}.globalNodeIds{1}(:));
        local = reshape(full(Uraw{g}(:)),3,[]);
        difference = abs(local - uByNode(:,ids));
        if ~isempty(difference)
            discrepancy = max(discrepancy, max(difference(:)));
        end
    end
    U = reshape(uByNode,[],1);
    maxSharedRelativeMismatch = discrepancy / max(norm(U,inf), eps);
end
