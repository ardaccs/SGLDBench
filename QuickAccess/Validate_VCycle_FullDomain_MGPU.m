function results = Validate_VCycle_FullDomain_MGPU(H, numGPUs, tolerance, gpuApply, cpuApply, cpuAtX)
%VALIDATE_VCYCLE_FULLDOMAIN_MGPU Validate GPU V-cycle against EXISTING CPU Solving_Vcycle.
%
% This file does NOT change any CPU/MATLAB or CUDA/C++ implementation.
% It calls the original Solving_Vcycle(r) as its reference.
%
% Run after the existing MATLAB CPU hierarchy / factorization setup:
%   global meshHierarchy_ weightFactorJacobi_ numLevels_ cholFac_ cholPermut_
%   results = Validate_VCycle_FullDomain_MGPU(H, 2, 1e-9);
%
% H: the EXISTING packed scalar hierarchy passed to Solving_MGPCG_GPU.
%    The existing CPU globals must describe the SAME hierarchy.
%
% MODE 1 (default): only Solving_MGPCG_GPU is callable.
%   Executes exactly ONE PCG iteration with y0=0, b=r.
%   CPU reference uses z=Solving_Vcycle(r), A*z=Solving_KbyU_MatrixFree(z):
%        y_CPU = ((r'*z)/(z'*A*z))*z
%   Compares the entire finest-level vector, including interface DOFs.
%   IMPORTANT: A uniform multiplier error in the GPU V-cycle cancels in
%   one PCG iteration; this mode cannot test the raw V-cycle scale.
%   Operator errors may also cause a mismatch in this indirect mode.
%
% MODE 2: if an ALREADY EXISTING GPU entry point returns raw V-cycle output,
%   pass a handle as gpuApply. No new entry point is created by this test:
%   results = Validate_VCycle_FullDomain_MGPU(H, 2, 1e-9, @(r) gpuVCycle(r));
%   The returned GPU correction is compared directly to Solving_Vcycle(r).
%
% If Solving_Vcycle is a local function not visible from this file, pass a
% callable handle as the FIFTH argument from a scope where it is visible:
%   results = Validate_VCycle_FullDomain_MGPU(H,2,1e-9,[],@Solving_Vcycle);
% The optional SIXTH argument supplies the CPU matrix-vector product handle.
%
% Tests: random, constant, partition-boundary/center impulse, sparse random.
% No implementation data, globals, CUDA source, or kernels are modified.

if nargin < 3 || isempty(tolerance), tolerance = 1e-9; end
if nargin < 4, gpuApply = []; end
if nargin < 5 || isempty(cpuApply), cpuApply = @Solving_Vcycle; end
if nargin < 6 || isempty(cpuAtX), cpuAtX = @Solving_KbyU_MatrixFree; end

assert(isnumeric(tolerance) && isscalar(tolerance) && ...
    isfinite(tolerance) && tolerance > 0, 'tolerance must be positive and finite.');
assert(isnumeric(numGPUs) && isscalar(numGPUs) && numGPUs >= 1 && ...
    fix(numGPUs) == numGPUs, 'numGPUs must be a positive integer.');
assert(isempty(gpuApply) || isa(gpuApply,'function_handle'), ...
    'gpuApply must be an existing GPU function handle.');
assert(isa(cpuApply,'function_handle') && isa(cpuAtX,'function_handle'), ...
    'cpuApply and cpuAtX must be function handles.');

global meshHierarchy_ weightFactorJacobi_ numLevels_
assert(~isempty(meshHierarchy_) && ~isempty(numLevels_) && ...
    ~isempty(weightFactorJacobi_), ...
    'First run your ORIGINAL CPU setup to initialize the V-cycle globals.');
assert(numel(meshHierarchy_) == double(numLevels_), ...
    'CPU numLevels_ does not match numel(meshHierarchy_).');

n = double(meshHierarchy_(1).numDOFs);
assert(n >= 1 && fix(n) == n, 'Invalid finest-level numDOFs.');
fixed = double(meshHierarchy_(1).fixedDOFs(:));
assert(all(fixed >= 1 & fixed <= n) && numel(unique(fixed)) == numel(fixed), ...
    'Invalid CPU fixedDOFs.');
freeMask = true(n,1);
freeMask(fixed) = false;
assert(any(freeMask), 'No free finest-level DOFs to test.');

isDirect = ~isempty(gpuApply);
if ~isDirect
    assert(isstruct(H) && isscalar(H) && isfield(H,'numLevels'), ...
        'The existing Solving_MGPCG_GPU MEX requires the packed scalar H.');
    assert(exist('Solving_MGPCG_GPU','file') ~= 0, ...
        'Solving_MGPCG_GPU not found on the MATLAB path.');
end

% Read-only consistency checks; never change the CPU or GPU hierarchies.
if isstruct(H) && isscalar(H) && isfield(H,'numLevels')
    assert(double(H.numLevels) == double(numLevels_), ...
        'CPU and packed GPU hierarchy level counts differ.');
    assert(double(H.numDOFs(1)) == n, ...
        'CPU and packed GPU finest-level DOF counts differ.');
    if isfield(H,'fixedDOFIds')
        assert(isequal(sort(double(H.fixedDOFIds(:))),sort(fixed)), ...
            'CPU fixedDOFs and GPU H.fixedDOFIds differ.');
    end
    if isfield(H,'jacobiOmega')
        omegaGPU = double(H.jacobiOmega);
    elseif isfield(H,'weightFactorJacobi')
        omegaGPU = double(H.weightFactorJacobi);
    else
        omegaGPU = [];
    end
    if ~isempty(omegaGPU)
        assert(abs(omegaGPU - double(weightFactorJacobi_)) <= ...
            1e-13*max(1,abs(double(weightFactorJacobi_))), ...
            'The CPU and GPU Jacobi weights differ.');
    end
end

[interfaceDOFs, nearDOFs, impulseNode, isPartitionImpulse] = ...
    getInterfaceInfo(meshHierarchy_, H, numGPUs, fixed);

if isDirect
    modeName = 'DIRECT: GPU raw V-cycle vs CPU raw V-cycle';
else
    modeName = 'INDIRECT: GPU vs CPU first PCG iterate';
end
fprintf('\n=== FULL-DOMAIN V-CYCLE VALIDATION ===\n');
fprintf('%s\n',modeName);
fprintf('Levels: %d | GPUs requested: %d | finest DOFs: %d | fixed: %d\n', ...
    numLevels_, numGPUs, n, numel(fixed));
fprintf('Shared-interface DOFs: %d | near-boundary DOFs: %d\n', ...
    numel(interfaceDOFs),numel(nearDOFs));

rng(42,'twister');
results = struct('test',{},'mode',{},'maxAbs',{},'relL2',{}, ...
    'maxInterfaceAbs',{},'maxNearBoundaryAbs',{},'directionRelL2',{}, ...
    'fixedAbs',{},'gpuSeconds',{},'cpuSeconds',{},'pass',{});

for testId = 1:4
    r = zeros(n,1);
    switch testId
        case 1
            testName = 'random';
            r = randn(n,1);
        case 2
            testName = 'constant';
            r(:) = 1;
        case 3
            if isPartitionImpulse
                testName = 'boundary_impulse';
            else
                testName = 'center_impulse';
            end
            idx = 3*impulseNode - (2:-1:0);
            r(idx) = [1;-2;3];
        case 4
            testName = 'sparse_random';
            ids = randperm(n,min(n,3000));
            r(ids) = randn(numel(ids),1);
    end
    % MEX masks fixed residual DOFs before calling applyVcycle.
    r(fixed) = 0;
    assert(norm(r) > 0, '%s input became zero after masking.',testName);

    fprintf('  %-20s GPU...',testName);
    tGPU = tic;
    if isDirect
        gpu = gpuApply(r);
    else
        % PUBLIC UNCHANGED MEX: its first PCG iteration uses applyVcycle(r).
        [gpu, iterations] = Solving_MGPCG_GPU( ...
            r, 1e-14, 1, zeros(n,1), H);
        assert(iterations == 1, ...
            'Expected 1 GPU PCG iteration for %s; received %g.', ...
            testName,iterations);
    end
    gpuSeconds = toc(tGPU);

    fprintf(' CPU...');
    tCPU = tic;
    try
        zCPU = double(cpuApply(r)); % EXACT EXISTING CPU Solving_Vcycle
    catch ME
        error('Validate_VCycle:CPUCall', ...
            ['Cannot call the existing CPU Solving_Vcycle: %s\n', ...
             'If it is a local function, pass its callable handle as ', ...
             'the fifth argument. No implementation changes are required.'], ...
            ME.message);
    end
    zCPU = zCPU(:);
    assert(numel(zCPU)==n && all(isfinite(zCPU)), ...
        'CPU Solving_Vcycle returned invalid values for %s.',testName);

    if isDirect
        reference = zCPU;
    else
        % This is exactly the first step of the supplied original CPU PCG:
        % p = PtV(r), Ap = AtX(p), alpha = (p'*r)/(p'*Ap), y=alpha*p.
        Az = double(cpuAtX(zCPU));
        Az = Az(:);
        assert(numel(Az)==n && all(isfinite(Az)), ...
            'CPU Solving_KbyU_MatrixFree returned invalid values.');
        denom = zCPU' * Az;
        numer = zCPU' * r;
        assert(isfinite(denom) && denom > 0 && isfinite(numer), ...
            'CPU first PCG step has invalid z''*A*z or z''*r for %s.',testName);
        reference = (numer/denom) * zCPU;
    end
    cpuSeconds = toc(tCPU);

    gpu = double(gpu(:));
    assert(numel(gpu) == n && all(isfinite(gpu)), ...
        'GPU produced wrong-sized or nonfinite output for %s.',testName);

    delta = abs(gpu-reference);
    maxAbs = max(delta);
    relL2 = norm(gpu-reference)/max(norm(reference),eps);
    interfaceAbs = 0;
    nearAbs = 0;
    fixedAbs = 0;
    if ~isempty(interfaceDOFs), interfaceAbs = max(delta(interfaceDOFs)); end
    if ~isempty(nearDOFs), nearAbs = max(delta(nearDOFs)); end
    if ~isempty(fixed), fixedAbs = max(abs(gpu(fixed))); end

    % Direction-only diagnostic separates a wrong V-cycle correction shape
    % from the final PCG step size; this is NOT a raw-scale validation.
    gain = (zCPU'*gpu) / max(zCPU'*zCPU,eps);
    directionRelL2 = norm(gpu-gain*zCPU)/max(norm(gpu),eps);

    scale = max(1,max(abs(reference)));
    passed = maxAbs <= tolerance*scale && relL2 <= tolerance && ...
        fixedAbs <= tolerance*scale;
    k = numel(results)+1;
    results(k) = struct('test',testName,'mode',modeName, ...
        'maxAbs',maxAbs,'relL2',relL2, ...
        'maxInterfaceAbs',interfaceAbs, ...
        'maxNearBoundaryAbs',nearAbs, ...
        'directionRelL2',directionRelL2,'fixedAbs',fixedAbs, ...
        'gpuSeconds',gpuSeconds,'cpuSeconds',cpuSeconds,'pass',passed);
    if passed, status='PASS'; else, status='FAIL'; end
    fprintf([' maxAbs=%.3e relL2=%.3e interface=%.3e ', ...
        'near=%.3e direction=%.3e fixed=%.3e ', ...
        '[GPU %.2fs CPU %.2fs] %s\n'], ...
        maxAbs,relL2,interfaceAbs,nearAbs,directionRelL2,fixedAbs, ...
        gpuSeconds,cpuSeconds,status);
end

assert(all([results.pass]), ...
    'FULL-DOMAIN V-CYCLE VALIDATION FAILED; see the results above.');
fprintf('\nPASS: all four full-domain comparisons.\n');
if ~isDirect
    fprintf(['NOTE: This was an INDIRECT test through one PCG iteration. ', ...
        'A uniform GPU V-cycle scaling error is invisible in this mode.\n']);
end
end

function [interfaceDOFs, nearDOFs, impulseNode, onPartition] = ...
    getInterfaceInfo(cpuH, gpuH, numGPUs, fixed)
% Diagnostics only. Uses partition metadata if it already exists.
fine = cpuH(1);
nNodes = double(fine.numNodes);
assert(3*nNodes == double(fine.numDOFs));
ids = double(fine.nodGridId(:))-1;
ny = double(fine.resY)+1;
nx = double(fine.resX)+1;
x = mod(floor(ids/ny),nx);
y = mod(ids,ny);
z = floor(ids/(ny*nx));
coords = x;
border = double(fine.resX)/2;
interfaceDOFs = [];
nearDOFs = [];
onPartition = false;

p = [];
if isfield(fine,'partitions') && iscell(fine.partitions) && ...
        numel(fine.partitions)==numGPUs
    p = fine.partitions;
elseif isstruct(gpuH) && numel(gpuH)>1 && ...
        isfield(gpuH(1),'partitions') && ...
        iscell(gpuH(1).partitions) && ...
        numel(gpuH(1).partitions)==numGPUs
    p = gpuH(1).partitions;
end

if ~isempty(p)
    occur = zeros(nNodes,1);
    for g = 1:numGPUs
        localIds = double(p{g}.globalNodeIds(:));
        assert(all(localIds>=1 & localIds<=nNodes), ...
            'Invalid globalNodeIds in partition %d.',g);
        occur(localIds) = occur(localIds)+1;
    end
    assert(all(occur>0),'Fine partition node coverage incomplete.');
    shared = find(occur>1);
    interfaceDOFs = reshape(3*shared(:)' + (-2:0)',[],1);
    switch double(p{1}.splitAxis)
        case 1, coords=x;
        case 2, coords=y;
        case 3, coords=z;
        otherwise, error('Invalid partition split axis.');
    end
    if numGPUs>1
        border = double(p{1}.range(2));
        onPartition = true;
        radius = 2;
        if isfield(cpuH(2),'spanWidth')
            radius = max(1,double(cpuH(2).spanWidth));
        end
        nearNodes = find(abs(coords-border)<=radius);
        nearDOFs = reshape(3*nearNodes(:)' + (-2:0)',[],1);
    end
end

blocked = false(double(fine.numDOFs),1);
blocked(fixed) = true;
eligible = find(~all(reshape(blocked,3,[]),1)');
assert(~isempty(eligible),'No free node available for impulse test.');
[~,j] = min(abs(coords(eligible)-border));
impulseNode = eligible(j);
end
