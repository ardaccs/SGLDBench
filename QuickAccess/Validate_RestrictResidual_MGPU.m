function results = Validate_Restriction_FullDomain_MGPU(H, numGPUs, tolerance)
% Exhaustively compare MGPU restriction against an independent MATLAB stencil.
% Usage (AFTER your existing FEA_SetupVoxelBased(numGPUs)):
%   global meshHierarchy_
%   results = Validate_Restriction_FullDomain_MGPU(meshHierarchy_, 2);
%
% Tests every output DOF for every level transition. Specifically checks
% GPU-partition interface nodes and impulses near the fine-level split.
% Never modifies the CUDA restriction kernel or the mesh hierarchy.

if nargin < 3 || isempty(tolerance), tolerance = 1e-12; end
assert(isscalar(tolerance) && isfinite(tolerance) && tolerance > 0);
assert(numel(H) >= 2, 'Expected at least two hierarchy levels.');
assert(isscalar(numGPUs) && numGPUs >= 1 && fix(numGPUs) == numGPUs);
rng(42, 'twister');
results = struct('level', {}, 'test', {}, 'maxAbs', {}, 'relL2', {}, ...
    'maxInterfaceAbs', {}, 'maxHaloAffectedAbs', {}, 'pass', {});

for lev = 1:numel(H)-1
    fine = H(lev);
    coarse = H(lev+1);
    assert(numel(fine.partitions) == numGPUs && ...
        numel(coarse.partitions) == numGPUs, 'GPU partition count mismatch.');
    assert(double(coarse.spanWidth) >= 2 && ...
        fix(double(coarse.spanWidth)) == double(coarse.spanWidth));
    nFine = double(fine.numNodes);
    nCoarse = double(coarse.numNodes);
    fprintf('\n=== FULL-DOMAIN RESTRICTION: %d -> %d (span %d; %d coarse nodes) ===\n', ...
        lev, lev+1, double(coarse.spanWidth), nCoarse);

    % Shared coarse nodes (each partition computes these independently).
    occur = zeros(nCoarse,1,'uint8');
    for g = 1:numGPUs
        ids = double(coarse.partitions{g}.globalNodeIds(:));
        assert(all(ids >= 1 & ids <= nCoarse));
        occur(ids) = occur(ids) + 1;
    end
    assert(all(occur > 0), 'Coarse node coverage incomplete.');
    interfaceNodes = find(occur > 1);
    interfaceDOFs = reshape(3*interfaceNodes(:)' + (-2:0)', [], 1);

    % Fine halo influence check: coarse stencil crosses a partition edge.
    % This deliberately includes a wider region than shared coarse nodes.
    coarseGrid = double(coarse.nodGridId(:)) - 1;
    cy = mod(coarseGrid, double(coarse.resY)+1);
    cx = mod(floor(coarseGrid/(double(coarse.resY)+1)), double(coarse.resX)+1);
    cz = floor(coarseGrid/((double(coarse.resY)+1)*(double(coarse.resX)+1)));
    switch double(fine.partitions{1}.splitAxis)
        case 1, axisCoords = cx;
        case 2, axisCoords = cy;
        case 3, axisCoords = cz;
        otherwise, error('Invalid split axis.');
    end
    centers = double(coarse.spanWidth)*axisCoords + 1; % 1-based fine nodes
    radius = double(coarse.spanWidth)-1;
    near = false(nCoarse,1);
    for g = 1:numGPUs-1
        border = double(fine.partitions{g}.range(2))+1; % shared node coordinate
        near = near | (centers-radius <= border & centers+radius >= border);
    end
    haloAffectedNodes = find(near);
    haloDOFs = reshape(3*haloAffectedNodes(:)' + (-2:0)', [], 1);

    fprintf('Interface nodes: %d; boundary-stencil nodes: %d\n', ...
        numel(interfaceNodes), numel(haloAffectedNodes));
    if numGPUs > 1 && isempty(interfaceNodes)
        warning('No shared coarse nodes found; interface checks will be empty.');
    end

    % Choose a fine active node near the first slab boundary for a sparse
    % impulse test; a wrong halo mapping is easier to expose with this input.
    fineGrid = double(fine.nodGridId(:)) - 1;
    fy = mod(fineGrid, double(fine.resY)+1);
    fx = mod(floor(fineGrid/(double(fine.resY)+1)), double(fine.resX)+1);
    fz = floor(fineGrid/((double(fine.resY)+1)*(double(fine.resX)+1)));
    switch double(fine.partitions{1}.splitAxis)
        case 1, fAxis = fx;
        case 2, fAxis = fy;
        case 3, fAxis = fz;
    end
    border = double(fine.partitions{1}.range(2)); % zero-based shared plane
    [~, impulseNode] = min(abs(fAxis-border));

    for testId = 1:4
        switch testId
            case 1
                testName = 'random';
                input = randn(3*nFine,1);
            case 2
                testName = 'constant';
                input = ones(3*nFine,1);
            case 3
                testName = 'boundary_impulse';
                input = zeros(3*nFine,1);
                input(3*impulseNode-2:3*impulseNode) = [1;-2;3];
            case 4
                testName = 'sparse_random';
                input = zeros(3*nFine,1);
                idx = randperm(nFine,min(1000,nFine));
                dofs = reshape(3*idx(:)' + (-2:0)',[],1);
                input(dofs) = randn(numel(dofs),1);
        end
        fprintf('  %-19s computing GPU...',testName);
        t = tic;
        gpu = Solving_RestrictResidual_MGPU(H, numGPUs, input, lev);
        gpuSeconds = toc(t);
        fprintf(' CPU reference...');
        t = tic;
        cpu = cpuReference(fine, coarse, input);
        cpuSeconds = toc(t);
        assert(numel(gpu)==3*nCoarse && numel(cpu)==3*nCoarse);
        assert(all(isfinite(gpu(:))) && all(isfinite(cpu(:))), ...
            'Nonfinite value in restriction results.');
        diff = abs(gpu(:)-cpu(:));
        maxAbs = max(diff);
        relL2 = norm(gpu(:)-cpu(:)) / max(norm(cpu(:)), eps);
        intAbs = 0;
        haloAbs = 0;
        if ~isempty(interfaceDOFs), intAbs = max(diff(interfaceDOFs)); end
        if ~isempty(haloDOFs), haloAbs = max(diff(haloDOFs)); end
        % Scale-aware componentwise criterion plus global relative norm.
        scale = max(1, max(abs(cpu(:))));
        passed = maxAbs <= tolerance*scale && relL2 <= tolerance;
        k = numel(results)+1;
        results(k) = struct('level',lev,'test',testName, ...
            'maxAbs',maxAbs,'relL2',relL2, ...
            'maxInterfaceAbs',intAbs,'maxHaloAffectedAbs',haloAbs,'pass',passed);
        fprintf(' maxAbs=%.3e relL2=%.3e interface=%.3e nearBoundary=%.3e [GPU %.1fs CPU %.1fs] %s\n', ...
            maxAbs,relL2,intAbs,haloAbs,gpuSeconds,cpuSeconds,passLabel(passed));
    end
end
assert(all([results.pass]), 'FULL-DOMAIN VALIDATION FAILED; see test results above.');
fprintf('\nPASS: all DOFs, all restriction transitions, all input patterns.\n');
end

function y = cpuReference(fine,coarse,input)
% Independent vectorized CPU implementation of the unchanged CUDA stencil.
% Chunking limits temporary memory. Arithmetic order follows dz, dx, dy.
n = double(coarse.numNodes);
s = double(coarse.spanWidth);
r = s-1;
fNy = double(fine.resY)+1;
fNx = double(fine.resX)+1;
fNz = double(fine.resZ)+1;
cNy = double(coarse.resY)+1;
cNx = double(coarse.resX)+1;
fmap = fine.nodMapForward(:);
y = zeros(3*n,1);
input = reshape(input,3,[]);
chunk = 16384;
for first = 1:chunk:n
    nodes = first:min(first+chunk-1,n);
    gid = double(coarse.nodGridId(nodes))-1;
    yy = s*mod(gid,cNy);
    xx = s*mod(floor(gid/cNy),cNx);
    zz = s*floor(gid/(cNy*cNx));
    values = zeros(3,numel(nodes));
    for dz = -r:r
        fz = zz+dz;
        wz = 1-abs(dz)/s;
        for dx = -r:r
            fx = xx+dx;
            wx = 1-abs(dx)/s;
            for dy = -r:r
                fy = yy+dy;
                valid = fy>=0 & fy<fNy & fx>=0 & fx<fNx & fz>=0 & fz<fNz;
                if ~any(valid), continue; end
                loc = find(valid);
                fullGrid = 1+fy(loc)+fx(loc)*fNy+fz(loc)*fNy*fNx;
                ids = double(fmap(fullGrid));
                keep = ids>0;
                if ~any(keep), continue; end
                loc = loc(keep);
                ids = ids(keep);
                w = (1-abs(dy)/s)*wx*wz;
                % Mirrors the per-node sequence in the original CUDA kernel.
                values(:,loc) = values(:,loc) + w*input(:,ids);
            end
        end
    end
    outDofs = reshape(3*nodes(:)' + (-2:0)',[],1);
    y(outDofs) = values(:);
end
end

function s = passLabel(tf)
if tf, s='PASS'; else, s='FAIL'; end
end
