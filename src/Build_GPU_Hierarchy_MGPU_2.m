function H = Build_GPU_Hierarchy_MGPU_2(meshHierarchy_, numGPUs, solverData)
%BUILD_GPU_HIERARCHY_MGPU_2 Construct the input expected by multi-GPU initializeGPUHierarchy.
%
%   H = Build_GPU_Hierarchy(meshHierarchy_, numGPUs)
%   H = Build_GPU_Hierarchy(meshHierarchy_, numGPUs, solverData)
%
% INPUT CONTRACT
%   meshHierarchy_(l).partitions{g}.globalNodeIds   : 1-based *global active*
%                                                     node IDs, ordered like
%                                                     partition-local vectors
%   meshHierarchy_(l).nodGridId                   : 1-based global *dense grid*
%                                                     ID for every active node
%   meshHierarchy_(l).resX/.resY/.resZ            : full-domain grid dimensions
%   meshHierarchy_(l).diagK                       : global 3*N double vector,
%                                                     unless P.diagK is present
%   meshHierarchy_(l).spanWidth                   : transfer width l -> l+1,
%                                                     or inferred from dimensions
%   P.nodeToElements, P.eNodMat                   : LOCAL int32 connectivity
%                                                     (required at finest level)
%   P.elementIds                                  : GLOBAL element indices
%                                                     (or P.eleModulus directly)
%   meshHierarchy_(1).eleModulus                  : global element modulus
%
%   Ke, jacobiOmega / weightFactorJacobi, coarseKFree and coarseFreeDOFIds
%   can be fields of solverData, meshHierarchy_(1), or meshHierarchy_(end)
%   (searched in that order). Fixed DOF IDs can likewise be supplied globally
%   in solverData.fixedDOFIds or meshHierarchy_(1).fixedDOFIds, OR as local
%   P.fixedDOFIds (preferred, if present).
%
% OUTPUT CONTRACT (matches multigpu_initialization.inl)
%   H                         scalar struct, with global coarse solver data
%   H.partitions{g}            per-GPU scalar struct
%   H.partitions{g}.<array>    int32 per-level vectors / cell arrays
%   H.partitions{g}.globalNodeIds{l}  normal local nodes only (not ghosts)
%   Halo nodes are APPENDED after normal nodes and are included in
%   P.nodMapForward{l} with indices numNodes(l)+1, ... . They are not included
%   in P.numNodes or P.globalNodeIds.
%
% ASSUMPTIONS OF THE DEPLOYED CUDA CODE
%   - GPUs must be ordered along a *1D* chain; missing halo nodes must be
%     available on an immediately adjacent GPU.
%   - Finest-level shared nodes can appear anywhere: global reductions use
%     ownedNodeMask. Each global node must have exactly one owner.
%   - Coarsest normal-node IDs may overlap at interfaces.
%     Each global coarse node must have exactly one ownedNodeMask entry, because
%     gatherGlobalCoarseResidualKernel writes only owned entries.
%     Coarse RHS ownership is correct only if local coarse residuals have
%     already incorporated contributions from neighboring partitions.
%   - This builder DOES NOT change node numbering or assemble coarseKFree.
%
% MATLAB output is for the previous multi-GPU C++ initialization INL, *not*
% for the original empty initializeGPUHierarchy() stubs.
    if nargin < 3
        solverData = struct();
    end
    assert(isstruct(meshHierarchy_) && ~isempty(meshHierarchy_), ...
        'meshHierarchy_ must be a nonempty struct array.');
    assert(isstruct(solverData) && isscalar(solverData), ...
        'solverData must be a scalar struct.');
    assert(isnumeric(numGPUs) && isscalar(numGPUs) && ...
        isfinite(numGPUs) && numGPUs >= 1 && numGPUs == floor(numGPUs), ...
        'numGPUs must be a positive integer.');
    numLevels = numel(meshHierarchy_);
    assert(numLevels >= 2, 'At least two multigrid levels are required.');
    numGPUs = double(numGPUs);
    % Global dense-grid geometry and index maps at every level.
    geo = repmat(struct('nx', 0, 'ny', 0, 'nz', 0, 'gridIds', [], ...
                        'forward', [], 'numNodes', 0), numLevels, 1);
    for l = 1:numLevels
        M = meshHierarchy_(l);
        assert(isfield(M, 'partitions') && iscell(M.partitions) && ...
               numel(M.partitions) == numGPUs, ...
               'meshHierarchy_(%d).partitions must have %d GPU entries.', l, numGPUs);
        geo(l).nx = requiredNonnegativeInteger(M, 'resX', l);
        geo(l).ny = requiredNonnegativeInteger(M, 'resY', l);
        geo(l).nz = requiredNonnegativeInteger(M, 'resZ', l);
        gridSize = (geo(l).nx + 1) * (geo(l).ny + 1) * (geo(l).nz + 1);
        assert(gridSize <= double(intmax('int32')), ...
            'Dense grid at level %d exceeds int32 indexing.', l);
        assert(isfield(M, 'nodGridId'), ...
            'meshHierarchy_(%d).nodGridId is needed to build local maps/halos.', l);
        gridIds = checkedIds(M.nodGridId, gridSize, ...
            sprintf('meshHierarchy_(%d).nodGridId', l));
        assert(numel(unique(gridIds)) == numel(gridIds), ...
            'Duplicate global nodGridId entries at level %d.', l);
        geo(l).gridIds = gridIds;
        geo(l).numNodes = numel(gridIds);
        geo(l).forward = zeros(gridSize, 1, 'int32');
        geo(l).forward(double(gridIds)) = int32((1:numel(gridIds))');
        if isfield(M, 'nodMapForward') && ~isempty(M.nodMapForward)
            map = int32(M.nodMapForward(:));
            assert(numel(map) == gridSize && isequal(map, geo(l).forward), ...
                'meshHierarchy_(%d).nodMapForward disagrees with nodGridId.', l);
        end
    end
    span = zeros(numLevels-1, 1, 'int32');
    for l = 1:numLevels-1
        M = meshHierarchy_(l);
        if isfield(M, 'spanWidth') && ~isempty(M.spanWidth)
            s = double(M.spanWidth);
            assert(isscalar(s) && isfinite(s) && s >= 1 && s == floor(s), ...
                'spanWidth at level %d must be a positive integer.', l);
        else
            fineDims = [geo(l).nx, geo(l).ny, geo(l).nz];
            coarseDims = [geo(l+1).nx, geo(l+1).ny, geo(l+1).nz];
            nonzero = coarseDims > 0;
            ratios = fineDims(nonzero) ./ coarseDims(nonzero);
            assert(~isempty(ratios) && all(ratios == ratios(1)), ...
                'Cannot infer a unique spanWidth for level %d.', l);
            s = ratios(1);
        end
        assert(s <= double(intmax('int32')) && ...
               isequal([geo(l).nx, geo(l).ny, geo(l).nz], ...
                       s * [geo(l+1).nx, geo(l+1).ny, geo(l+1).nz]), ...
            'Grid dimensions are inconsistent with spanWidth at level %d.', l);
        span(l) = int32(s);
    end
    H = struct();
    H.Ke = requiredGlobalDouble(solverData, meshHierarchy_, 'Ke');
    assert(isequal(size(H.Ke), [24, 24]) && all(isfinite(H.Ke(:))), ...
        'Ke must be a finite double 24-by-24 matrix.');
    omega = firstGlobalField(solverData, meshHierarchy_, 'jacobiOmega');
    if isempty(omega)
        omega = firstGlobalField(solverData, meshHierarchy_, 'weightFactorJacobi');
    end
    assert(isnumeric(omega) && isreal(omega) && isscalar(omega) && ...
           isfinite(omega), 'jacobiOmega / weightFactorJacobi is required.');
    H.jacobiOmega = double(omega);
    H.coarseKFree = firstGlobalField(solverData, meshHierarchy_, 'coarseKFree');
    coarseFree = firstGlobalField(solverData, meshHierarchy_, 'coarseFreeDOFIds');
    assert(issparse(H.coarseKFree) && isa(H.coarseKFree, 'double') && ...
           isreal(H.coarseKFree), 'coarseKFree must be a real double sparse matrix.');
    assert(~isempty(coarseFree), 'coarseFreeDOFIds is required.');
    H.coarseFreeDOFIds = checkedIds(coarseFree, 3 * geo(end).numNodes, ...
                                    'coarseFreeDOFIds');
    assert(numel(unique(H.coarseFreeDOFIds)) == numel(H.coarseFreeDOFIds), ...
        'coarseFreeDOFIds contains duplicates.');
    assert(isequal(size(H.coarseKFree), ...
                   [numel(H.coarseFreeDOFIds), numel(H.coarseFreeDOFIds)]), ...
        'coarseKFree dimensions must match coarseFreeDOFIds.');
    H.globalCoarseNumNodes = int32(geo(end).numNodes);
    globalFixed = firstGlobalField(solverData, meshHierarchy_, 'fixedDOFIds');
    if isempty(globalFixed)
        globalFixed = int32([]);
    else
        globalFixed = checkedIds(globalFixed, 3 * geo(1).numNodes, 'fixedDOFIds');
    end
    H.partitions = cell(1, numGPUs);
    for g = 1:numGPUs
        Q = struct();
        Q.numLevels = int32(numLevels);
        Q.resX = int32(reshape([geo.nx], [], 1));
        Q.resY = int32(reshape([geo.ny], [], 1));
        Q.resZ = int32(reshape([geo.nz], [], 1));
        Q.numNodes = zeros(numLevels, 1, 'int32');
        Q.numElements = zeros(numLevels, 1, 'int32');
        Q.numDOFs = zeros(numLevels, 1, 'int32');
        Q.spanWidth = span;
        Q.numLeftSharedNodes = zeros(numLevels, 1, 'int32');
        Q.numBufferNodes = zeros(numLevels, 1, 'int32');
        Q.nodeToElements = cell(numLevels, 1);
        Q.eNodMat = cell(numLevels, 1);
        Q.nodGridId = cell(numLevels, 1);
        Q.nodMapForward = cell(numLevels, 1);
        Q.eleModulus = cell(numLevels, 1);
        Q.diagK = cell(numLevels, 1);
        Q.globalNodeIds = cell(numLevels, 1);
        Q.ownedNodeMask = cell(numLevels, 1);
        Q.leftHaloSendLocal = cell(numLevels, 1);
        Q.leftHaloSendDestLocal = cell(numLevels, 1);
        Q.rightHaloSendLocal = cell(numLevels, 1);
        Q.rightHaloSendDestLocal = cell(numLevels, 1);
        Q.leftInterfaceSendLocal = int32([]);
        Q.leftInterfaceSendDestLocal = int32([]);
        Q.rightInterfaceSendLocal = int32([]);
        Q.rightInterfaceSendDestLocal = int32([]);
        Q.sharedNodesLocal = cell(1, numGPUs);   % Original function's convention, finest only.
        for l = 1:numLevels
            M = meshHierarchy_(l);
            P = M.partitions{g};
            assert(isstruct(P) && isscalar(P), ...
                'Partition %d at level %d must be a scalar struct.', g, l);
            assert(isfield(P, 'globalNodeIds'), ...
                'Partition %d level %d needs globalNodeIds.', g, l);
            ids = checkedIds(P.globalNodeIds, geo(l).numNodes, ...
                              sprintf('partitions{%d}.globalNodeIds at level %d', g, l));
            assert(numel(unique(ids)) == numel(ids), ...
                'Repeated globalNodeIds in GPU %d level %d.', g, l);
            n = numel(ids);
            assert(n > 0 && 3*n <= double(intmax('int32')), ...
                'GPU %d level %d has invalid local node count.', g, l);
            if isfield(P, 'numNodes')
                assert(double(P.numNodes) == n, ...
                    'GPU %d level %d numNodes != numel(globalNodeIds).', g, l);
            end
            Q.globalNodeIds{l} = ids;
            Q.numNodes(l) = int32(n);
            Q.numDOFs(l) = int32(3*n);
            Q.nodGridId{l} = geo(l).gridIds(double(ids));
            haveConnectivity = isfield(P, 'nodeToElements') && ...
                               isfield(P, 'eNodMat') && ~isempty(P.nodeToElements);
            assert(l ~= 1 || haveConnectivity, ...
                'Finest partition %d needs local nodeToElements and eNodMat.', g);
            if haveConnectivity
                nodeToElem = int32(P.nodeToElements);
                elemToNode = int32(P.eNodMat);
                assert(isequal(size(nodeToElem), [n, 8]) && ...
                       size(elemToNode, 2) == 8, ...
                    'Connectivity shape mismatch for GPU %d level %d.', g, l);
                nElem = size(elemToNode, 1);
                assert(all(nodeToElem(:) >= 0 & nodeToElem(:) <= nElem) && ...
                       all(elemToNode(:) >= 1 & elemToNode(:) <= n), ...
                    'Invalid local connectivity IDs for GPU %d level %d.', g, l);
                Q.nodeToElements{l} = nodeToElem;
                Q.eNodMat{l} = elemToNode;
                Q.numElements(l) = int32(nElem);
            else
                % The uploaded CUDA kernels use connectivity only at level 1.
                Q.nodeToElements{l} = zeros(n, 8, 'int32');
                Q.eNodMat{l} = zeros(0, 8, 'int32');
                Q.numElements(l) = int32(0);
            end
            if isfield(P, 'numElements')
                assert(double(P.numElements) == double(Q.numElements(l)), ...
                    'GPU %d level %d numElements disagrees with connectivity.', g, l);
            end
            if l == 1
                if isfield(P, 'eleModulus') && ~isempty(P.eleModulus)
                    modulus = double(P.eleModulus(:));
                else
                    assert(isfield(M, 'eleModulus') && isfield(P, 'elementIds'), ...
                        'GPU %d needs P.eleModulus or (P.elementIds and global eleModulus).', g);
                    elementIds = checkedIds(P.elementIds, numel(M.eleModulus), ...
                                            sprintf('GPU %d elementIds', g));
                    modulus = double(M.eleModulus(double(elementIds)));
                end
                assert(numel(modulus) == double(Q.numElements(l)) && ...
                       all(isfinite(modulus)), ...
                    'Invalid finest-level eleModulus for GPU %d.', g);
                Q.eleModulus{l} = modulus;
            else
                Q.eleModulus{l} = zeros(0, 1);
            end
            if l < numLevels
                if isfield(P, 'diagK') && ~isempty(P.diagK)
                    localDiagonal = double(P.diagK(:));
                else
                    assert(isfield(M, 'diagK'), ...
                        'Missing diagK at level %d (global or GPU-local).', l);
                    diagGlobal = double(M.diagK(:));
                    assert(numel(diagGlobal) == 3*geo(l).numNodes, ...
                        'Global diagK size mismatch at level %d.', l);
                    byNode = reshape(diagGlobal, 3, []);
                    localDiagonal = reshape(byNode(:, double(ids)), [], 1);
                end
                assert(numel(localDiagonal) == 3*n && ...
                       all(isfinite(localDiagonal)) && all(localDiagonal ~= 0), ...
                    'Invalid diagK for GPU %d level %d.', g, l);
                Q.diagK{l} = localDiagonal;
            else
                Q.diagK{l} = zeros(0, 1);
            end
        end
        % Convert GLOBAL fixed DOFs to partition-LOCAL finest DOFs, or use the
        % explicit P.fixedDOFIds (already local). Always preserve 1-based IDs.
        P1 = meshHierarchy_(1).partitions{g};
        if isfield(P1, 'fixedDOFIds') && ~isempty(P1.fixedDOFIds)
            Q.fixedDOFIds = checkedIds(P1.fixedDOFIds, double(Q.numDOFs(1)), ...
                                       sprintf('GPU %d fixedDOFIds', g));
        else
            nodeIds = double(Q.globalNodeIds{1});
            globalDofByLocal = reshape([3*nodeIds.'-2; ...
                                        3*nodeIds.'-1; 3*nodeIds.'], [], 1);
            [tf, localDofs] = ismember(double(globalFixed), globalDofByLocal);
            Q.fixedDOFIds = int32(localDofs(tf));
        end
        H.partitions{g} = Q;
    end
    % Every global active node must appear among the normal nodes of at
    % least one GPU. Duplicate normals are permitted except at the final
    % coarse level, where the uploaded CUDA assembly requires ownership.
    for l = 1:numLevels
        idsAtLevel = cellfun(@(p) p.globalNodeIds{l}, H.partitions, ...
                             'UniformOutput', false);
        represented = unique(vertcat(idsAtLevel{:}));
        assert(isequal(represented, int32((1:geo(l).numNodes)')), ...
            'Some global active nodes at level %d have no GPU partition.', l);
    end
    % ---------------------------------------------------------------------
    % Shared/interface mappings. sendLocal and sendDestLocal have matching
    % ordering; the sender's destination entries refer to the OTHER GPU.
    % ---------------------------------------------------------------------
    for l = 1:numLevels
        for g1 = 1:numGPUs-1
            ids1 = H.partitions{g1}.globalNodeIds{l};
            for g2 = g1+1:numGPUs
                ids2 = H.partitions{g2}.globalNodeIds{l};
                common = intersect(ids1, ids2);
                assert(isempty(common) || g2 == g1+1, ...
                    'Nonadjacent GPU %d/%d share nodes at level %d.', g1, g2, l);
            end
        end
        for g = 1:numGPUs
            Q = H.partitions{g};
            ids = Q.globalNodeIds{l};
            if g > 1
                prev = H.partitions{g-1}.globalNodeIds{l};
                leftShared = intersect(ids, prev);
                if l == 1
                    % Shared nodes may occur at arbitrary local positions.
                    % CUDA globalDotProduct/globalNorm use ownedNodeMask,
                    % not a prefix-based ownership assumption.
                    assert(numel(unique(leftShared)) == numel(leftShared), ...
                        'Duplicate left-shared node IDs.');
                end
                Q.numLeftSharedNodes(l) = int32(numel(leftShared));
                Q.ownedNodeMask{l} = uint8(~ismember(ids, prev));
            else
                Q.ownedNodeMask{l} = ones(numel(ids), 1, 'uint8');
            end
            H.partitions{g} = Q;
        end
    end
    for g = 1:numGPUs-1
        a = H.partitions{g};
        b = H.partitions{g+1};
        [shared, ia, ib] = intersect(a.globalNodeIds{1}, b.globalNodeIds{1});
        assert(numel(shared) == numel(ia) && numel(ia) == numel(ib), ...
               'Finest shared-node correspondence mismatch.');
        a.rightInterfaceSendLocal = int32(ia(:));
        a.rightInterfaceSendDestLocal = int32(ib(:));
        b.leftInterfaceSendLocal = int32(ib(:));
        b.leftInterfaceSendDestLocal = int32(ia(:));
        a.sharedNodesLocal{g+1} = int32(ia(:));
        b.sharedNodesLocal{g} = int32(ib(:));
        H.partitions{g} = a;
        H.partitions{g+1} = b;
    end
    % Coarsest normal-node arrays MAY overlap across partition interfaces.
    % The deployed CUDA gatherGlobalCoarseResidualKernel accepts ownedMask
    % and writes only owned nodes into the global coarse RHS. Therefore it
    % requires exactly one OWNER per global coarse node, not exactly one
    % local appearance. Shared nodes remain available for local transfers.
    nGlobalCoarse = double(geo(end).numNodes);
    coarsePresence = zeros(nGlobalCoarse, 1, 'uint32');
    coarseOwnerCount = zeros(nGlobalCoarse, 1, 'uint32');
    for g = 1:numGPUs
        ids = double(H.partitions{g}.globalNodeIds{end}(:));
        owner = H.partitions{g}.ownedNodeMask{end}(:);
        assert(numel(ids) == numel(owner) && ...
               all(owner == 0 | owner == 1), ...
               'GPU %d: invalid coarsest ownedNodeMask.', g);
        assert(all(ids >= 1 & ids <= nGlobalCoarse), ...
               'GPU %d: invalid global coarse-node IDs.', g);
        coarsePresence(ids) = coarsePresence(ids) + uint32(1);
        ownedIds = ids(logical(owner));
        coarseOwnerCount(ownedIds) = coarseOwnerCount(ownedIds) + uint32(1);
    end
    assert(all(coarsePresence >= 1), ...
        'Some global coarse nodes are missing from all partitions.');
    assert(all(coarseOwnerCount == 1), ...
        'Every global coarse node must have exactly one ownedNodeMask owner.');
    fprintf('Coarse nodes: %d global, %d replicated appearances; unique ownership verified.\n', ...
        nGlobalCoarse, sum(double(coarsePresence)) - nGlobalCoarse);
    % ---------------------------------------------------------------------
    % Halo maps for restriction and interpolation. A halo node is a global
    % node required by this partition's transfer stencil but not present
    % among its NORMAL local nodes. Ghost nodes are appended in contiguous
    % blocks: [normal local | LEFT ghosts | RIGHT ghosts].
    % ---------------------------------------------------------------------
    for l = 1:numLevels
        for g = 1:numGPUs
            Q = H.partitions{g};
            required = transferDependencyIds(l, g, H.partitions, geo, span);
            needed = setdiff(required, Q.globalNodeIds{l}, 'stable');
            onLeft = false(numel(needed), 1);
            onRight = false(numel(needed), 1);
            if g > 1
                onLeft = ismember(needed, H.partitions{g-1}.globalNodeIds{l});
            end
            if g < numGPUs
                onRight = ismember(needed, H.partitions{g+1}.globalNodeIds{l});
            end
            bad = find(~onLeft & ~onRight, 1);
            if ~isempty(bad)
                error(['GPU %d level %d needs global node %d for a transfer, ' ...
                       'but neither adjacent GPU has it as a normal node. ' ...
                       'The current CUDA halo exchange supports adjacent GPUs only.'], ...
                      g, l, double(needed(bad)));
            end
            % A duplicated neighboring node can be sourced from either;
            % left gets deterministic priority.
            leftIds = needed(onLeft);
            rightIds = needed(~onLeft & onRight);
            nNormal = double(Q.numNodes(l));
            nLeft = numel(leftIds);
            Q.numBufferNodes(l) = int32(nLeft + numel(rightIds));
            localAndGhost = [Q.globalNodeIds{l}; leftIds; rightIds];
            denseGridIds = geo(l).gridIds(double(localAndGhost));
            localMap = zeros(numel(geo(l).forward), 1, 'int32');
            localMap(double(denseGridIds)) = int32((1:numel(localAndGhost))');
            Q.nodMapForward{l} = localMap;
            H.partitions{g} = Q;
            if nLeft > 0
                sender = H.partitions{g-1};
                [ok, sendLoc] = ismember(leftIds, sender.globalNodeIds{l});
                assert(all(ok), 'Internal LEFT halo mapping error.');
                sender.rightHaloSendLocal{l} = int32(sendLoc(:));
                sender.rightHaloSendDestLocal{l} = ...
                    int32((nNormal + (1:nLeft))');
                H.partitions{g-1} = sender;
            end
            if ~isempty(rightIds)
                sender = H.partitions{g+1};
                [ok, sendLoc] = ismember(rightIds, sender.globalNodeIds{l});
                assert(all(ok), 'Internal RIGHT halo mapping error.');
                sender.leftHaloSendLocal{l} = int32(sendLoc(:));
                sender.leftHaloSendDestLocal{l} = ...
                    int32((nNormal + nLeft + (1:numel(rightIds)))');
                H.partitions{g+1} = sender;
            end
        end
    end
    % Final checks: these are the constraints enforced by the CUDA parser.
    for g = 1:numGPUs
        Q = H.partitions{g};
        for l = 1:numLevels
            assert(isa(Q.nodMapForward{l}, 'int32') && ...
                   numel(Q.nodMapForward{l}) == numel(geo(l).forward));
            assert(all(Q.nodMapForward{l} >= 0) && ...
                   all(Q.nodMapForward{l} <= Q.numNodes(l)+Q.numBufferNodes(l)));
            assert(numel(Q.globalNodeIds{l}) == double(Q.numNodes(l)));
        end
    end
end
function ids = transferDependencyIds(l, g, parts, geo, span)
% All global normal/ghost node IDs required by CUDA transfer kernels at l.
    required = int32([]);
    L = numel(geo);
    if l < L
        % Restriction l -> l+1: gather fine residual near local coarse nodes.
        fine = geo(l);
        coarseIds = parts{g}.globalNodeIds{l+1};
        [y, x, z] = gridCoords(geo(l+1).gridIds(double(coarseIds)), geo(l+1));
        s = double(span(l));
        yf = s*y; xf = s*x; zf = s*z;
        for dz = -(s-1):(s-1)
            for dx = -(s-1):(s-1)
                for dy = -(s-1):(s-1)
                    yc = yf+dy; xc = xf+dx; zc = zf+dz;
                    valid = yc>=0 & yc<=fine.ny & ...
                            xc>=0 & xc<=fine.nx & zc>=0 & zc<=fine.nz;
                    idx = 1+yc(valid)+(fine.ny+1)*xc(valid)+ ...
                          (fine.ny+1)*(fine.nx+1)*zc(valid);
                    required = [required; fine.forward(idx(:))]; %#ok<AGROW>
                end
            end
        end
    end
    if l > 1
        % Interpolation l -> l-1: trilinear coarse stencil for local fine nodes.
        coarse = geo(l);
        fineIds = parts{g}.globalNodeIds{l-1};
        [y, x, z] = gridCoords(geo(l-1).gridIds(double(fineIds)), geo(l-1));
        s = double(span(l-1));
        by = floor(y/s); bx = floor(x/s); bz = floor(z/s);
        ry = mod(y,s); rx = mod(x,s); rz = mod(z,s);
        for dz = 0:1
            for dx = 0:1
                for dy = 0:1
                    yc = by+dy; xc = bx+dx; zc = bz+dz;
                    valid = yc>=0 & yc<=coarse.ny & ...
                            xc>=0 & xc<=coarse.nx & zc>=0 & zc<=coarse.nz & ...
                            (dy==0 | ry~=0) & (dx==0 | rx~=0) & ...
                            (dz==0 | rz~=0);
                    idx = 1+yc(valid)+(coarse.ny+1)*xc(valid)+ ...
                          (coarse.ny+1)*(coarse.nx+1)*zc(valid);
                    required = [required; coarse.forward(idx(:))]; %#ok<AGROW>
                end
            end
        end
    end
    ids = unique(required(required>0), 'stable');
end
function [y, x, z] = gridCoords(gridIds, G)
    zero = double(gridIds(:)) - 1;
    ny = G.ny + 1;
    nx = G.nx + 1;
    y = mod(zero, ny);
    x = mod(floor(zero/ny), nx);
    z = floor(zero/(ny*nx));
end
function ids = checkedIds(values, upper, description)
    assert(isnumeric(values) && isreal(values), '%s must be numeric.', description);
    raw = double(values(:));
    assert(all(isfinite(raw)) && all(raw>=1) && ...
           all(raw==floor(raw)) && all(raw<=upper) && ...
           all(raw <= double(intmax('int32'))), ...
        '%s contains invalid/out-of-range 1-based IDs.', description);
    ids = int32(raw);
end
function n = requiredNonnegativeInteger(M, name, l)
    assert(isfield(M, name), 'meshHierarchy_(%d).%s is required.', l, name);
    n = double(M.(name));
    assert(isscalar(n) && isnumeric(n) && isfinite(n) && ...
           n>=0 && n==floor(n) && n <= double(intmax('int32')), ...
        'meshHierarchy_(%d).%s must be a nonnegative int32-compatible integer.', l, name);
end
function value = requiredGlobalDouble(solverData, hierarchy, name)
    value = firstGlobalField(solverData, hierarchy, name);
    assert(isa(value, 'double') && isreal(value) && ~isempty(value), ...
        'Global %s must be supplied as a real double array.', name);
end
function value = firstGlobalField(solverData, hierarchy, name)
    value = [];
    if isfield(solverData, name) && ~isempty(solverData.(name))
        value = solverData.(name);
    elseif isfield(hierarchy(1), name) && ~isempty(hierarchy(1).(name))
        value = hierarchy(1).(name);
    elseif isfield(hierarchy(end), name) && ~isempty(hierarchy(end).(name))
        value = hierarchy(end).(name);
    end
end
