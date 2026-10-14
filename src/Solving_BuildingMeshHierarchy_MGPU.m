function Solving_BuildingMeshHierarchy_MGPU()

    global meshHierarchy_;

    global numLevels_;

    global eNodMatHalfTemp_;

    global nonDyadic_;

    if numel(meshHierarchy_)>1, return; end

    %%0. global ordering of nodes on each levels

    nodeVolume = 1:(int32(meshHierarchy_(1).resX)+1)*(int32(meshHierarchy_(1).resY)+1)*(int32(meshHierarchy_(1).resZ)+1);

    nodeVolume = reshape(nodeVolume(:), meshHierarchy_(1).resY+1, meshHierarchy_(1).resX+1, meshHierarchy_(1).resZ+1);

    if 1==nonDyadic_ && numLevels_>=4

        numLevels_ = numLevels_ - 1; 

    else

        nonDyadic_ = 0;

    end

    for ii=2:numLevels_

        %%1. adjust voxel resolution

        if ii==2 && 1==nonDyadic_

            spanWidth = 4;

        else

            spanWidth = 2;

        end

        nx = meshHierarchy_(ii-1).resX/spanWidth;

        ny = meshHierarchy_(ii-1).resY/spanWidth;

        nz = meshHierarchy_(ii-1).resZ/spanWidth;

        %%2. initialize mesh

        meshHierarchy_(ii) = Data_CartesianMeshStruct();

        meshHierarchy_(ii).resX = nx;

        meshHierarchy_(ii).resY = ny;

        meshHierarchy_(ii).resZ = nz;

        meshHierarchy_(ii).eleSize = meshHierarchy_(ii-1).eleSize*spanWidth;

        meshHierarchy_(ii).spanWidth = spanWidth;

        %%3. identify solid&void elements

        %%3.1 capture raw info.

        iEleVolume = reshape(meshHierarchy_(ii-1).eleMapForward, spanWidth*ny, spanWidth*nx, spanWidth*nz);

        iEleVolumeTemp = reshape((1:int32(spanWidth^3*nx*ny*nz))', spanWidth*ny, spanWidth*nx, spanWidth*nz);

        iFineNodVolumeTemp = reshape((1:int32((spanWidth*nx+1)*(spanWidth*ny+1)* (spanWidth*nz+1)))', spanWidth*ny+1, spanWidth*nx+1, spanWidth*nz+1);

        elementUpwardMap = zeros(nx*ny*nz,spanWidth^3,'int32');

        elementUpwardMapTemp = zeros(nx*ny*nz,spanWidth^3,'int32');

        transferMatTemp = zeros((spanWidth+1)^3,nx*ny*nz,'int32');  

        for jj=1:nz

            iFineEleGroup = iEleVolume(:,:,spanWidth*(jj-1)+1:spanWidth*(jj-1)+spanWidth);

            iFineEleGroupTemp = iEleVolumeTemp(:,:,spanWidth*(jj-1)+1:spanWidth*(jj-1)+spanWidth);

            iFineNodGroupTemp = iFineNodVolumeTemp(:,:,spanWidth*(jj-1)+1:spanWidth*jj+1);

            for kk=1:nx

                iFineEleSubGroup = iFineEleGroup(:,spanWidth*(kk-1)+1:spanWidth*(kk-1)+spanWidth,:);

                iFineEleSubGroupTemp = iFineEleGroupTemp(:,spanWidth*(kk-1)+1:spanWidth*(kk-1)+spanWidth,:);

                iFineNodSubGroupTemp = iFineNodGroupTemp(:,spanWidth*(kk-1)+1:spanWidth*kk+1,:);

                for gg=1:ny

                    iFineEles = iFineEleSubGroup(spanWidth*(gg-1)+1:spanWidth*(gg-1)+spanWidth,:,:);

                    iFineEles = reshape(iFineEles, spanWidth^3, 1)';

                    iFineElesTemp = iFineEleSubGroupTemp(spanWidth*(gg-1)+1:spanWidth*(gg-1)+spanWidth,:,:);

                    iFineElesTemp = reshape(iFineElesTemp, spanWidth^3, 1)';

                    iFineNodsTemp = iFineNodSubGroupTemp(spanWidth*(gg-1)+1:spanWidth*gg+1,:,:);

                    iFineNodsTemp = reshape(iFineNodsTemp, (spanWidth+1)^3, 1)';

                    eleIndex = (jj-1)*ny*nx + (kk-1)*ny + gg;

                    elementUpwardMap(eleIndex,:) = iFineEles;   

                    elementUpwardMapTemp(eleIndex,:) = iFineElesTemp;

                    transferMatTemp(:,eleIndex) = iFineNodsTemp;                        

                end

            end

        end

        %%3.2 building the mapping relation for following tri-linear interpolation                  

        %                    _______                         _______ _______

        %                   |       |                       |       |       |

        %           void    |solid  |                       |void   |solid  |

        %                   |       |                       |       |       |

        %            _______|_______|                       |_______|_______|       

        %           |       |       |       <----->         |       |       |

        %           |solid  |solid  |                       |solid  |solid  |   

        %           |       |       |                       |       |       |

        %           |_______|_______|                       |_______|_______|

        % elementsIncVoidLastLevelGlobalOrdering    elementsLastLevelGlobalOrdering

        unemptyElements = find(sum(elementUpwardMap,2)>0);

        elementUpwardMapTemp = elementUpwardMapTemp(unemptyElements,:);

        elementsIncVoidLastLevelGlobalOrdering = reshape(elementUpwardMapTemp, numel(elementUpwardMapTemp), 1);

        nodesIncVoidLastLevelGlobalOrdering = eNodMatHalfTemp_(elementsIncVoidLastLevelGlobalOrdering,:);

        nodesIncVoidLastLevelGlobalOrdering = Common_RecoverHalfeNodMat(nodesIncVoidLastLevelGlobalOrdering);

        nodesIncVoidLastLevelGlobalOrdering = unique(nodesIncVoidLastLevelGlobalOrdering);

        meshHierarchy_(ii).intermediateNumNodes = length(nodesIncVoidLastLevelGlobalOrdering);

        transferMatTemp = transferMatTemp(:,unemptyElements);

        temp = zeros((spanWidth*nx+1)*(spanWidth*ny+1)*(spanWidth*nz+1),1,'int32');     

        temp(nodesIncVoidLastLevelGlobalOrdering) = (1:meshHierarchy_(ii).intermediateNumNodes)';

        meshHierarchy_(ii).transferMat = temp(transferMatTemp);

        meshHierarchy_(ii).transferMatCoeffi = zeros(meshHierarchy_(ii).intermediateNumNodes,1);

        for kk=1:1:(spanWidth+1)^3

            solidNodesLastLevel = meshHierarchy_(ii).transferMat(kk,:);

            meshHierarchy_(ii).transferMatCoeffi(solidNodesLastLevel,1) = meshHierarchy_(ii).transferMatCoeffi(solidNodesLastLevel,1) + 1;

        end

        elementsLastLevelGlobalOrdering = meshHierarchy_(ii-1).eleMapBack;

        nodesLastLevelGlobalOrdering = eNodMatHalfTemp_(elementsLastLevelGlobalOrdering,:);

        nodesLastLevelGlobalOrdering = Common_RecoverHalfeNodMat(nodesLastLevelGlobalOrdering);

        nodesLastLevelGlobalOrdering = unique(nodesLastLevelGlobalOrdering);

        [~,meshHierarchy_(ii).solidNodeMapCoarser2Finer] = intersect(nodesIncVoidLastLevelGlobalOrdering, nodesLastLevelGlobalOrdering);

        meshHierarchy_(ii).solidNodeMapCoarser2Finer = int32(meshHierarchy_(ii).solidNodeMapCoarser2Finer);

        %%3.3 initialize the solid elements 

        meshHierarchy_(ii).eleMapForward = zeros(nx*ny*nz,1,'int32');

        meshHierarchy_(ii).eleMapBack = int32(unemptyElements);

        meshHierarchy_(ii).numElements = length(unemptyElements);

        meshHierarchy_(ii).eleMapForward(unemptyElements) = (1:meshHierarchy_(ii).numElements)';

        meshHierarchy_(ii).colors = Solving_Coloring(meshHierarchy_(ii).eleMapForward, nx, ny, nz);

        elementUpwardMap = elementUpwardMap(unemptyElements,:); 

        meshHierarchy_(ii).elementUpwardMap = elementUpwardMap; clear elementUpwardMap

        %%4. discretize

        nodenrs = reshape(1:int32((nx+1)*(ny+1)*(nz+1)), 1+meshHierarchy_(ii).resY, 1+meshHierarchy_(ii).resX, 1+meshHierarchy_(ii).resZ);

        eNodVec = reshape(nodenrs(1:end-1,1:end-1,1:end-1)+1,nx*ny*nz, 1);

        eNodMat = repmat(eNodVec(meshHierarchy_(ii).eleMapBack),1,8);

        eNodMatHalfTemp_ = repmat(eNodVec,1,8);

        tmp = [0 ny+[1 0] -1 (ny+1)*(nx+1)+[0 ny+[1 0] -1]]; tmp = int32(tmp);

        for jj=1:8

            eNodMat(:,jj) = eNodMat(:,jj) + repmat(tmp(jj), meshHierarchy_(ii).numElements,1);

            eNodMatHalfTemp_(:,jj) = eNodMatHalfTemp_(:,jj) + repmat(tmp(jj), nx*ny*nz,1);

        end

        eNodMatHalfTemp_ = eNodMatHalfTemp_(:,[3 4 7 8]);

        meshHierarchy_(ii).nodMapBack = unique(eNodMat);

        %%Arda:Addition for non-dyadic mesh hierarchy

        %% Keep the node numbering relative to this level's full Cartesian grid.

        meshHierarchy_(ii).nodGridId = int32(meshHierarchy_(ii).nodMapBack);

        meshHierarchy_(ii).numNodes = length(meshHierarchy_(ii).nodMapBack);

        meshHierarchy_(ii).numDOFs = meshHierarchy_(ii).numNodes*3;

        meshHierarchy_(ii).nodMapForward = zeros((nx+1)*(ny+1)*(nz+1),1,'int32');

        meshHierarchy_(ii).nodMapForward(meshHierarchy_(ii).nodMapBack) = (1:meshHierarchy_(ii).numNodes)';     

        for jj=1:8

            eNodMat(:,jj) = meshHierarchy_(ii).nodMapForward(eNodMat(:,jj));

        end

        if 1==nonDyadic_, kk = ii; else, kk = ii-1; end

        tmp = nodeVolume(1:2^kk:meshHierarchy_(1).resY+1, 1:2^kk:meshHierarchy_(1).resX+1, 1:2^kk:meshHierarchy_(1).resZ+1);

        tmp = reshape(tmp,numel(tmp),1);

        meshHierarchy_(ii).nodMapBack = tmp(meshHierarchy_(ii).nodMapBack);

        %%5. initialize multi-grid Restriction&Interpolation operator

        meshHierarchy_(ii).multiGridOperatorRI = Solving_Operator4MultiGridRestrictionAndInterpolation('inNODE', spanWidth);

        meshHierarchy_(ii).multiGridOperatorRIdense = full(meshHierarchy_(ii).multiGridOperatorRI);

        %%6. identify boundary info.

        % meshHierarchy_(ii).numNod2ElesVec = zeros(meshHierarchy_(ii).numNodes,1,'int32');

        % for jj=1:8

            % iNodes = eNodMat(:,jj);

            % meshHierarchy_(ii).numNod2ElesVec(iNodes,:) = meshHierarchy_(ii).numNod2ElesVec(iNodes) + 1;      

        % end

        % meshHierarchy_(ii).nodesOnBoundary = int32(find(meshHierarchy_(ii).numNod2ElesVec<8));

        % allNodes = zeros(meshHierarchy_(ii).numNodes,1,'int32');

        % allNodes(meshHierarchy_(ii).nodesOnBoundary) = 1; 

        % tmp = zeros(meshHierarchy_(ii).numElements,1,'int32');

        % for jj=1:8

            % tmp = tmp + allNodes(eNodMat(:,jj));

        % end

        % meshHierarchy_(ii).elementsOnBoundary = int32(find(tmp>0));

        % blockIndex = Solving_MissionPartition(meshHierarchy_(ii).numElements, 5.0e6);

        % for jj=1:size(blockIndex,1)               

            % rangeIndex = (blockIndex(jj,1):blockIndex(jj,2))';

            % patchIndices = eNodMat(rangeIndex, [4 3 2 1  5 6 7 8  1 2 6 5  8 7 3 4  5 8 4 1  2 3 7 6])';

            % patchIndices = reshape(patchIndices(:), 4, 6*numel(rangeIndex));

            % tmp = zeros(meshHierarchy_(ii).numNodes, 1);

            % tmp(meshHierarchy_(ii).nodesOnBoundary) = 1;

            % tmp = tmp(patchIndices); tmp = sum(tmp,1);

            % iBoundaryEleFaces = patchIndices(:,find(4==tmp));

            % meshHierarchy_(ii).boundaryEleFaces(end+1:end+size(iBoundaryEleFaces,2),:) = iBoundaryEleFaces';

        % end

        meshHierarchy_(ii).eNodMat = eNodMat;

        %%7: identify active voxels touching active nodes

        % nodeToElements(localNode, :) stores up to 8 active coarse elements

        % touching that active node. 0 means empty slot.

        meshHierarchy_(ii).eNodMat = int32(eNodMat);

        nodeToElements = zeros(meshHierarchy_(ii).numNodes, 8, 'int32');

        nodeToElementsCount = zeros(meshHierarchy_(ii).numNodes, 1, 'int32');

        for ee = 1:meshHierarchy_(ii).numElements

            for nn = 1:8

                iNode = eNodMat(ee, nn);  % already local active-node id

                nodeToElementsCount(iNode) = nodeToElementsCount(iNode) + 1;

                slot = nodeToElementsCount(iNode);

                if slot <= 8

                    nodeToElements(iNode, slot) = ee;

                else

                    error('Node touches more than 8 elements. This should not happen for a structured hex mesh.');

                end

            end

        end

        meshHierarchy_(ii).nodeToElements = nodeToElements;

            %% ============================================================

            %% 8. Multi-GPU partitioning

            %% ============================================================

            if isfield(meshHierarchy_(1), 'partitions') && ~isempty(meshHierarchy_(1).partitions)

                numGPUs = numel(meshHierarchy_(1).partitions);

                % Determine split axis from finest level

                finest = meshHierarchy_(1);

                finestDims = [finest.resX, finest.resY, finest.resZ];

                splitAxis = double(finest.partitionSplitAxis);

                finestEdges = double(finest.partitionEdges);

                finestAxisLength = finestDims(splitAxis);

                % Project finest partition onto CURRENT hierarchy level

                current = meshHierarchy_(ii);

                currentDims = [ current.resX, current.resY, current.resZ];

                currentAxisLength = currentDims(splitAxis);

                edges = round( finestEdges / double(finestAxisLength) * double(currentAxisLength));

                edges(1) = 0;

                edges(end) = currentAxisLength;

                % Ensure valid non-empty slabs if possible

                for g = 2:numGPUs

                    edges(g) = max(edges(g), edges(g-1) + 1);

                end

                for g = numGPUs:-1:2

                    edges(g) = min(edges(g), edges(g+1) - 1);

                end

                assert(all(diff(edges) > 0), ...

                    'Level %d is too coarse for %d GPU slabs.', ...

                    ii, numGPUs);

                % Calculate element coordinates for current level

                [iy, ix, iz] = ind2sub( [current.resY, current.resX, current.resZ], double(current.eleMapBack));

                switch splitAxis

                    case 1

                        eleCoord = ix;

                    case 2

                        eleCoord = iy;

                    case 3

                        eleCoord = iz;

                end

                fprintf('Level %d partition edges: ', ii);

                fprintf('%d ', edges);

                fprintf('\n');

            %% ------------------------------------------------------------

            %% 8.1 Build the normal element/node partitions

            %% ------------------------------------------------------------

            for g = 1:numGPUs

                lowerBound = edges(g) + 1;

                upperBound = edges(g+1);

                % Active elements owned by this GPU.

                elementIds = find(eleCoord >= lowerBound & eleCoord <= upperBound);

                elementIds = int32(elementIds(:));

                % Global-within-level active node IDs touched by these elements.

                globalNodeIds = unique(meshHierarchy_(ii).eNodMat(double(elementIds), :));

                globalNodeIds = int32(globalNodeIds(:));

                % Remap the partition eNodMat to compact local node numbering.

                globalToLocalNode = zeros(meshHierarchy_(ii).numNodes, 1, 'int32');

                globalToLocalNode(double(globalNodeIds)) = int32(1:numel(globalNodeIds));

                localENodMat = meshHierarchy_(ii).eNodMat(double(elementIds), :);

                localENodMat = globalToLocalNode(double(localENodMat));

                % Build local node-to-element adjacency.

                localNodeToElements = zeros(numel(globalNodeIds), 8, 'int32');

                localNodeToElementsCount = zeros(numel(globalNodeIds), 1, 'int32');

                for ee = 1:numel(elementIds)

                    for nn = 1:8

                        iNode = localENodMat(ee, nn);

                        localNodeToElementsCount(iNode) = ...

                            localNodeToElementsCount(iNode) + 1;

                        slot = localNodeToElementsCount(iNode);

                        if slot <= 8

                            localNodeToElements(iNode, slot) = ee;

                        else

                            error('Partition node touches more than 8 elements.');

                        end

                    end

                end

                P = struct();

                P.range = int32([lowerBound, upperBound]);

                P.splitAxis = int32(splitAxis);

                P.elementIds = elementIds;

                P.numElements = numel(elementIds);

                P.globalNodeIds = globalNodeIds;

                P.numNodes = numel(globalNodeIds);

                P.eNodMat = int32(localENodMat);

                P.nodeToElements = localNodeToElements;

                % Geometry map for this partition.

                % globalNodeIds are compact active-node IDs within this level.

                % nodGridId gives the corresponding full Cartesian-grid IDs.

                % This is intentionally solver-stage agnostic: restriction,

                % interpolation, smoothing, etc. can all use the same geometry.

                P.nodGridId = int32( ...

                    meshHierarchy_(ii).nodGridId(double(globalNodeIds)));

                % Local-only Cartesian slot -> compact local node.

                % The generic halo builder below creates nodMapForwardWithHalo

                % without changing this existing local-only map.

                P.nodMapForward = zeros( ...

                    (current.resX+1) * (current.resY+1) * (current.resZ+1), ...

                    1, 'int32');

                P.nodMapForward(double(P.nodGridId)) = ...

                    int32(1:P.numNodes);

                % Store P only after every field above has been added.

                meshHierarchy_(ii).partitions{g} = P;

            end

            clear globalToLocalNode localNodeToElementsCount

        end

    end 

    %% ============================================================

    %% 9. Build generic left/right node halos on EVERY hierarchy level

    %% ============================================================

    % The halo is deliberately NOT tied to restriction.  It is a reusable

    % partition communication layer:

    %

    %   local buffer = [ normal local nodes | left halo | right halo ]

    %

    % Restriction, interpolation, smoothing, matrix-free operators, etc.

    % may all consume the same halo metadata.  Shared interface nodes that

    % are already present in P.globalNodeIds are NOT duplicated in the halo.

    %

    % For the current multigrid stencil, the geometric reach outside a slab

    % is spanWidth-1 node layers, possibly enlarged if the rounded

    % partition edges on neighboring levels do not align exactly.

    if isfield(meshHierarchy_(1), 'partitions') && ...

            ~isempty(meshHierarchy_(1).partitions)

        % Halo width belongs to the NEXT restriction transition, not a
        % constant shared by every level. In a non-dyadic hierarchy the first
        % transition can have spanWidth=4 and subsequent transitions 2.
        for levelId = 1:numel(meshHierarchy_)
            if levelId < numel(meshHierarchy_)
                haloWidth = max(0, double(meshHierarchy_(levelId+1).spanWidth) - 1);
            else
                haloWidth = 0; % coarsest level has no outgoing restriction
            end

            if ~isfield(meshHierarchy_(levelId), 'partitions') || ...

                    isempty(meshHierarchy_(levelId).partitions)

                continue;

            end

            level = meshHierarchy_(levelId);

            % On coarse levels nodMapBack has been remapped to finest-grid

            % numbering, so use nodGridId.  On the finest level nodMapBack

            % already is the current-level Cartesian-grid numbering.

            if isfield(level, 'nodGridId') && ~isempty(level.nodGridId)

                levelNodeGridId = int32(level.nodGridId(:));

            else

                levelNodeGridId = int32(level.nodMapBack(:));

            end

            partitions = level.partitions;

            % Make the finest-level partition geometry explicit as well if

            % its builder did not already store range/splitAxis.

            if levelId == 1

                finestEdges = double(meshHierarchy_(1).partitionEdges);

                finestSplitAxis = int32(meshHierarchy_(1).partitionSplitAxis);

                for g = 1:numel(partitions)

                    if ~isfield(partitions{g}, 'range') || isempty(partitions{g}.range)

                        partitions{g}.range = int32( ...

                            [finestEdges(g)+1, finestEdges(g+1)]);

                    end

                    if ~isfield(partitions{g}, 'splitAxis') || isempty(partitions{g}.splitAxis)

                        partitions{g}.splitAxis = finestSplitAxis;

                    end

                end

            end

            % If projected partition edges round differently between
            % levels, the coarse nodes' fine-grid stencil can extend
            % farther than spanWidth-1 beyond a partition's local nodes.
            % Increase the halo just enough to cover these stencils.
            if levelId < numel(meshHierarchy_)
                spanWidth = double(meshHierarchy_(levelId+1).spanWidth);
                radius = spanWidth - 1;
                nextPartitions = meshHierarchy_(levelId+1).partitions;
                for g = 1:numel(partitions)
                    fineRange = double(partitions{g}.range);
                    coarseRange = double(nextPartitions{g}.range);
                    minStencilNode = spanWidth * (coarseRange(1)-1) + 1 - radius;
                    maxStencilNode = spanWidth * coarseRange(2) + 1 + radius;
                    haloWidth = max([haloWidth, ...
                        fineRange(1) - minStencilNode, ...
                        maxStencilNode - (fineRange(2)+1)]);
                end
            end

            partitions = BuildGenericLeftRightNodeHalos( ...

                level, partitions, levelNodeGridId, haloWidth);

            meshHierarchy_(levelId).partitions = partitions;

            meshHierarchy_(levelId).haloWidth = int32(haloWidth);

        end

    end

    clear -global eNodMatHalfTemp_

    %% Print Mesh Hierarchy

    disp('Mesh Hierarchy...');

    disp('             #Resolutions         #Elements       #DOFs');

    for ii = 1:numel(meshHierarchy_)

        fprintf('...Level %i: %4i x %4i x %4i %11i %11i\n', ...

            ii, ...

            meshHierarchy_(ii).resX, ...

            meshHierarchy_(ii).resY, ...

            meshHierarchy_(ii).resZ, ...

            meshHierarchy_(ii).numElements, ...

            meshHierarchy_(ii).numDOFs);

    end

end

function partitions = BuildGenericLeftRightNodeHalos( ...

    level, partitions, levelNodeGridId, haloWidth)

%BUILDGENERICLEFTRIGHTNODEHALOS

% Build solver-stage-independent node halos for a 1-D slab decomposition.

%

% Existing partition nodes stay in:

%   P.globalNodeIds            level-active node IDs

%   P.nodGridId                current-level Cartesian-grid IDs

%

% Generic halo-aware buffer numbering is:

%   1 : P.numLocalNodes                         normal local nodes

%   ...                                         left halo

%   ... : P.numBufferNodes                      right halo

%

% The halo structures contain both receive-side and send-side metadata:

%

%   P.halo.left/right.globalNodeIds   received halo active-node IDs

%   P.halo.left/right.nodGridId       received halo Cartesian-grid IDs

%   P.halo.left/right.recvLocal       destination in this GPU's halo buffer

%   P.halo.left/right.sourceLocal     source index in neighbor normal buffer

%   P.halo.left/right.sendLocal       indices this GPU sends to that neighbor

%   P.halo.left/right.sendDestLocal   destination indices on that neighbor

%

% Interface nodes already in P.globalNodeIds are intentionally excluded.

% They are shared/local nodes, not halo nodes.  Any additive synchronization

% of duplicated interface nodes is a separate communication operation.

    numGPUs = numel(partitions);

    numLevelNodes = double(level.numNodes);

    assert(numel(levelNodeGridId) == numLevelNodes, ...

        'levelNodeGridId must contain one Cartesian-grid ID per active node.');

    nodeDims = [double(level.resY)+1, ...

                double(level.resX)+1, ...

                double(level.resZ)+1];

    [nodeY, nodeX, nodeZ] = ind2sub(nodeDims, double(levelNodeGridId));

    splitAxis = double(partitions{1}.splitAxis);

    switch splitAxis

        case 1

            nodeCoord = nodeX;

            axisNodeCount = double(level.resX) + 1;

        case 2

            nodeCoord = nodeY;

            axisNodeCount = double(level.resY) + 1;

        case 3

            nodeCoord = nodeZ;

            axisNodeCount = double(level.resZ) + 1;

        otherwise

            error('splitAxis must be 1 (X), 2 (Y), or 3 (Z).');

    end

    allActiveNodeIds = int32((1:numLevelNodes)');

    fullGridNodeCount = prod(nodeDims);

    %% Pass 1: choose geometric left/right halo nodes and build buffer maps.

    for g = 1:numGPUs

        P = partitions{g};

        assert(isfield(P, 'globalNodeIds'), ...

            'Partition %d is missing globalNodeIds.', g);

        assert(isfield(P, 'range') && numel(P.range) == 2, ...

            'Partition %d is missing its element range.', g);

        assert(double(P.splitAxis) == splitAxis, ...

            'All partitions on a level must use the same splitAxis.');

        localIds = int32(P.globalNodeIds(:));

        localMask = false(numLevelNodes, 1);

        localMask(double(localIds)) = true;

        lowerElement = double(P.range(1));

        upperElement = double(P.range(2));

        % Owned element range [lowerElement, upperElement] touches node

        % coordinates [lowerElement, upperElement+1].

        leftMin  = max(1, lowerElement - haloWidth);

        leftMax  = lowerElement - 1;

        rightMin = upperElement + 2;

        rightMax = min(axisNodeCount, upperElement + 1 + haloWidth);

        if isempty(localIds)

            % No active work lives in this slab, so it does not need halo data.

            leftIds = zeros(0,1,'int32');

            rightIds = zeros(0,1,'int32');

        else

            if haloWidth > 0 && leftMin <= leftMax

                leftMask = nodeCoord >= leftMin & ...

                           nodeCoord <= leftMax & ...

                           ~localMask;

                leftIds = allActiveNodeIds(leftMask);

            else

                leftIds = zeros(0,1,'int32');

            end

            if haloWidth > 0 && rightMin <= rightMax

                rightMask = nodeCoord >= rightMin & ...

                            nodeCoord <= rightMax & ...

                            ~localMask;

                rightIds = allActiveNodeIds(rightMask);

            else

                rightIds = zeros(0,1,'int32');

            end

        end

        haloIds = [leftIds; rightIds];

        bufferIds = [localIds; haloIds];

        P.numLocalNodes = numel(localIds);

        P.haloNodeIds = int32(haloIds);

        P.numHaloNodes = numel(haloIds);

        P.bufferGlobalNodeIds = int32(bufferIds);

        P.numBufferNodes = numel(bufferIds);

        P.bufferNodGridId = int32(levelNodeGridId(double(bufferIds)));

        % Ensure the same geometry fields exist on the finest level too.

        P.nodGridId = int32(levelNodeGridId(double(localIds)));

        if ~isfield(P, 'nodMapForward') || isempty(P.nodMapForward)

            P.nodMapForward = zeros(fullGridNodeCount, 1, 'int32');

            P.nodMapForward(double(P.nodGridId)) = ...

                int32(1:numel(localIds));

        end

        % Keep the existing local-only P.nodMapForward untouched.  Kernels

        % that use the halo-extended vector should use this map instead.

        P.nodMapForwardWithHalo = zeros(fullGridNodeCount, 1, 'int32');

        P.nodMapForwardWithHalo(double(P.bufferNodGridId)) = ...

            int32(1:numel(bufferIds));

        nLocal = numel(localIds);

        nLeft = numel(leftIds);

        nRight = numel(rightIds);

        P.halo.width = int32(haloWidth);

        P.halo.left.neighborGPU = int32(max(g-1, 0));

        P.halo.left.globalNodeIds = int32(leftIds);

        P.halo.left.nodGridId = int32(levelNodeGridId(double(leftIds)));

        P.halo.left.recvLocal = int32((nLocal + (1:nLeft))');

        P.halo.left.sourceLocal = zeros(nLeft,1,'int32');

        P.halo.left.sendLocal = zeros(0,1,'int32');

        P.halo.left.sendDestLocal = zeros(0,1,'int32');

        P.halo.left.sendGlobalNodeIds = zeros(0,1,'int32');

        P.halo.right.neighborGPU = int32(min(g+1, numGPUs+1));

        if g == numGPUs

            P.halo.right.neighborGPU = int32(0);

        end

        P.halo.right.globalNodeIds = int32(rightIds);

        P.halo.right.nodGridId = int32(levelNodeGridId(double(rightIds)));

        P.halo.right.recvLocal = int32((nLocal + nLeft + (1:nRight))');

        P.halo.right.sourceLocal = zeros(nRight,1,'int32');

        P.halo.right.sendLocal = zeros(0,1,'int32');

        P.halo.right.sendDestLocal = zeros(0,1,'int32');

        P.halo.right.sendGlobalNodeIds = zeros(0,1,'int32');

        partitions{g} = P;

    end

    %% Pass 2: map each receive halo entry to the adjacent GPU's local node.

    for g = 1:numGPUs

        P = partitions{g};

        if ~isempty(P.halo.left.globalNodeIds)

            assert(g > 1, ...

                'Partition %d has a left halo but no left neighbor.', g);

            %% sourceLocal _> The global ids of the nodes that they share

            donorIds = int32(partitions{g-1}.globalNodeIds(:));

            [found, sourceLocal] = ismember( ...

                P.halo.left.globalNodeIds, donorIds);

            assert(all(found), ...

                ['Left halo of partition %d reaches beyond the immediate ' ...

                 'left neighbor. Reduce haloWidth/increase slab width, or ' ...

                 'extend the exchange logic to multiple-hop neighbors.'], g);

            P.halo.left.sourceLocal = int32(sourceLocal(:));

        end

        if ~isempty(P.halo.right.globalNodeIds)

            assert(g < numGPUs, ...

                'Partition %d has a right halo but no right neighbor.', g);

            donorIds = int32(partitions{g+1}.globalNodeIds(:));

            [found, sourceLocal] = ismember( ...

                P.halo.right.globalNodeIds, donorIds);

            assert(all(found), ...

                ['Right halo of partition %d reaches beyond the immediate ' ...

                 'right neighbor. Reduce haloWidth/increase slab width, or ' ...

                 'extend the exchange logic to multiple-hop neighbors.'], g);

            P.halo.right.sourceLocal = int32(sourceLocal(:));

        end

        partitions{g} = P;

    end

    %% Pass 3: create sender-side lists directly from neighbor receive lists.

    for g = 1:numGPUs

        P = partitions{g};

        if g > 1

            leftNeighbor = partitions{g-1};

            P.halo.left.sendLocal = int32( ...

                leftNeighbor.halo.right.sourceLocal(:));

            P.halo.left.sendDestLocal = int32( ...

                leftNeighbor.halo.right.recvLocal(:));

            P.halo.left.sendGlobalNodeIds = int32( ...

                leftNeighbor.halo.right.globalNodeIds(:));

        end

        if g < numGPUs

            rightNeighbor = partitions{g+1};

            P.halo.right.sendLocal = int32( ...

                rightNeighbor.halo.left.sourceLocal(:));

            P.halo.right.sendDestLocal = int32( ...

                rightNeighbor.halo.left.recvLocal(:));

            P.halo.right.sendGlobalNodeIds = int32( ...

                rightNeighbor.halo.left.globalNodeIds(:));

        end

        partitions{g} = P;

    end

end