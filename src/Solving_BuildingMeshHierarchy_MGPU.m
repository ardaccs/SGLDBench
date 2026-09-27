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
		%					 _______						 _______ _______
		%					|		|						|		|		|
		%			void	|solid	|						|void	|solid	|
		%					|		|						|		|		|
		%			 _______|_______|						|_______|_______|		
		%			|		|		|		<----->			|		|		|
		%			|solid	|solid	|						|solid	|solid	|	
		%			|		|		|						|		|		|
		%			|_______|_______|						|_______|_______|
		% elementsIncVoidLastLevelGlobalOrdering	elementsLastLevelGlobalOrdering
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

		


        %%8. Multi-GPU partitioning for this hierarchy level
        % Use the same number of GPUs already used to partition the finest level.
        %
        % Besides the KbyU partition data, this section also builds everything
        % needed to perform restriction from level (ii-1) -> level ii without
        % creating a second hierarchy structure:
        %
        %   partition.nodGridId
        %       Local coarse node -> full coarse Cartesian-grid node ID.
        %
        %   partition.restrictionFineNodeIds
        %       Active fine-level node IDs required by this GPU's restriction
        %       stencil. This includes the local fine nodes plus the halo.
        %
        %   partition.restrictionFineNodeMapForward
        %       LOCAL slab-grid ID -> local index in restrictionFineNodeIds.
        %       0 means an inactive fine-grid node.
        %
        %   partition.restrictionFineGridStart
        %       [x0 y0 z0], 0-based offset of the local fine slab in the full
        %       fine Cartesian node grid. Only the split axis is cropped.
        %
        %   partition.restrictionRecvSourceLocal{srcGPU}
        %       Fine-partition local node IDs to gather from source GPU.
        %
        %   partition.restrictionRecvDestLocal{srcGPU}
        %       Destination local node IDs in the restriction input buffer.
        %
        % A node already present in this GPU's fine partition is always sourced
        % locally. Only true halo nodes are assigned to another GPU.

        if isfield(meshHierarchy_(1), 'partitions') && ~isempty(meshHierarchy_(1).partitions)

            numGPUs = numel(meshHierarchy_(1).partitions);

            % Split along the longest axis of this level.
            dims = [meshHierarchy_(ii).resX, ...
                    meshHierarchy_(ii).resY, ...
                    meshHierarchy_(ii).resZ];

            [~, splitAxis] = max(dims);
            edges = round(linspace(0, dims(splitAxis), numGPUs + 1));

            % Coordinates of active elements in the full Cartesian grid.
            % eleMapBack uses MATLAB ordering: [Y, X, Z].
            [iy, ix, iz] = ind2sub( ...
                [meshHierarchy_(ii).resY, ...
                 meshHierarchy_(ii).resX, ...
                 meshHierarchy_(ii).resZ], ...
                double(meshHierarchy_(ii).eleMapBack));

            switch splitAxis
                case 1
                    eleCoord = ix;
                case 2
                    eleCoord = iy;
                case 3
                    eleCoord = iz;
            end

            meshHierarchy_(ii).partitions = cell(numGPUs,1);

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

                % IMPORTANT FOR RESTRICTION:
                % globalNodeIds are active-node IDs within this hierarchy level.
                % nodGridId converts those IDs to the full Cartesian-grid IDs
                % expected by the restriction kernel.
                P.nodGridId = int32( ...
                    meshHierarchy_(ii).nodGridId(double(globalNodeIds)));

                meshHierarchy_(ii).partitions{g} = P;
            end

            clear globalToLocalNode localNodeToElementsCount

            %% ------------------------------------------------------------
            %% 8.2 Build restriction slab + halo metadata
            %%     Restriction is from fine level (ii-1) -> coarse level ii.
            %% ------------------------------------------------------------

            fineLevel = ii - 1;
            fineMesh = meshHierarchy_(fineLevel);

            if ~isfield(fineMesh, 'partitions') || isempty(fineMesh.partitions)
                error(['Fine hierarchy level %d is not partitioned. ' ...
                       'Restriction metadata cannot be built.'], fineLevel);
            end

            if numel(fineMesh.partitions) ~= numGPUs
                error('Fine and coarse levels have different GPU counts.');
            end

            fineRes = double([fineMesh.resX, fineMesh.resY, fineMesh.resZ]);
            radius = double(spanWidth - 1);

            % Full fine Cartesian active-node map. This is only used while
            % constructing the partition metadata. It is NOT copied into every
            % partition. Each GPU stores only its own slab + halo map.
            fineMapVolume = reshape( ...
                fineMesh.nodMapForward, ...
                fineMesh.resY + 1, ...
                fineMesh.resX + 1, ...
                fineMesh.resZ + 1);

            fprintf('\nRestriction partitioning: level %d -> %d\n', ...
                fineLevel, ii);

            for g = 1:numGPUs

                P = meshHierarchy_(ii).partitions{g};

                lowerBound = double(P.range(1));
                upperBound = double(P.range(2));

                % Owned coarse elements [lowerBound, upperBound] touch coarse
                % node coordinates [lowerBound-1, upperBound] on splitAxis.
                % Convert those centers to the fine grid and extend by the
                % restriction stencil radius = spanWidth - 1.
                fineStart0 = [0, 0, 0];
                fineEnd0   = fineRes;

                coarseNodeStart0 = lowerBound - 1;
                coarseNodeEnd0   = upperBound;

                fineStart0(splitAxis) = max( ...
                    0, ...
                    double(spanWidth) * coarseNodeStart0 - radius);

                fineEnd0(splitAxis) = min( ...
                    fineRes(splitAxis), ...
                    double(spanWidth) * coarseNodeEnd0 + radius);

                % Extract only the required slab from the fine nodMapForward.
                % MATLAB storage order is [Y, X, Z].
                switch splitAxis
                    case 1  % split in X
                        fineSlabGlobalMap = fineMapVolume( ...
                            :, ...
                            fineStart0(1)+1:fineEnd0(1)+1, ...
                            :);

                    case 2  % split in Y
                        fineSlabGlobalMap = fineMapVolume( ...
                            fineStart0(2)+1:fineEnd0(2)+1, ...
                            :, ...
                            :);

                    case 3  % split in Z
                        fineSlabGlobalMap = fineMapVolume( ...
                            :, ...
                            :, ...
                            fineStart0(3)+1:fineEnd0(3)+1);
                end

                fineSlabGlobalMap = int32(fineSlabGlobalMap);
                activeMask = fineSlabGlobalMap ~= 0;

                % Active fine nodes that must exist in this GPU's restriction
                % input vector. This is local data + true halo data.
                restrictionFineNodeIds = unique(fineSlabGlobalMap(activeMask));
                restrictionFineNodeIds = int32(restrictionFineNodeIds(:));

                % Remap the slab's GLOBAL active fine-node IDs to compact LOCAL
                % restriction-buffer IDs. The temporary lookup is destroyed
                % immediately after this partition is built.
                globalToRestrictionLocal = zeros( ...
                    fineMesh.numNodes, 1, 'int32');

                globalToRestrictionLocal(double(restrictionFineNodeIds)) = ...
                    int32(1:numel(restrictionFineNodeIds));

                restrictionFineNodeMapForward = ...
                    zeros(size(fineSlabGlobalMap), 'int32');

                restrictionFineNodeMapForward(activeMask) = ...
                    globalToRestrictionLocal( ...
                        double(fineSlabGlobalMap(activeMask)));

                clear globalToRestrictionLocal

                %% --------------------------------------------------------
                %% 8.2.1 Locate every required fine node in the distributed
                %%       fine-level vectors.
                %% --------------------------------------------------------

                numRestrictionFineNodes = numel(restrictionFineNodeIds);

                restrictionFineNodeSourceGPU = ...
                    zeros(numRestrictionFineNodes, 1, 'int32');

                restrictionFineNodeSourceLocal = ...
                    zeros(numRestrictionFineNodes, 1, 'int32');

                % Prefer this GPU's own fine partition. Therefore duplicated
                % interface nodes do not require communication.
                sourceOrder = [g, 1:g-1, g+1:numGPUs];

                for srcGPU = sourceOrder

                    sourceGlobalNodeIds = ...
                        fineMesh.partitions{srcGPU}.globalNodeIds;

                    [isPresent, sourceLocal] = ismember( ...
                        restrictionFineNodeIds, ...
                        sourceGlobalNodeIds);

                    take = isPresent & ...
                           (restrictionFineNodeSourceGPU == 0);

                    restrictionFineNodeSourceGPU(take) = int32(srcGPU);
                    restrictionFineNodeSourceLocal(take) = ...
                        int32(sourceLocal(take));
                end

                if any(restrictionFineNodeSourceGPU == 0)
                    error(['GPU %d level %d restriction halo contains active ' ...
                           'fine nodes that are missing from every fine partition.'], ...
                           g-1, ii);
                end

                % Communication lists. For srcGPU == g these are simply the
                % local gather list. For srcGPU ~= g they describe halo receives:
                % gather sourceLocal on srcGPU -> copy -> scatter to destLocal.
                restrictionRecvSourceLocal = cell(1, numGPUs);
                restrictionRecvDestLocal = cell(1, numGPUs);

                for srcGPU = 1:numGPUs
                    destLocal = find( ...
                        restrictionFineNodeSourceGPU == srcGPU);

                    restrictionRecvDestLocal{srcGPU} = ...
                        int32(destLocal(:));

                    restrictionRecvSourceLocal{srcGPU} = ...
                        int32(restrictionFineNodeSourceLocal(destLocal));
                end

                numHaloNodes = nnz( ...
                    restrictionFineNodeSourceGPU ~= g);

                %% --------------------------------------------------------
                %% 8.2.2 Store restriction metadata directly in the existing
                %%       coarse partition -- no extra H hierarchy.
                %% --------------------------------------------------------

                P.restrictionSpanWidth = int32(spanWidth);
                P.restrictionRadius = int32(radius);

                % Local fine slab position in FULL fine-grid coordinates.
                % Coordinates are 0-based [X Y Z], matching the CUDA kernel.
                P.restrictionFineGridStart = int32(fineStart0);
                P.restrictionFineGridEnd = int32(fineEnd0);
                P.restrictionFineGridSize = int32( ...
                    fineEnd0 - fineStart0 + 1);

                % Local slab map. Stored flattened in MATLAB [Y,X,Z] ordering,
                % which is exactly the ordering used by the CUDA linear index.
                P.restrictionFineNodeMapForward = int32( ...
                    restrictionFineNodeMapForward(:));

                P.restrictionFineNodeIds = restrictionFineNodeIds;
                P.numRestrictionFineNodes = numRestrictionFineNodes;
                P.numRestrictionHaloNodes = numHaloNodes;

                % Where the local restriction input vector gets its values.
                P.restrictionFineNodeSourceGPU = ...
                    restrictionFineNodeSourceGPU;

                P.restrictionFineNodeSourceLocal = ...
                    restrictionFineNodeSourceLocal;

                P.restrictionRecvSourceLocal = ...
                    restrictionRecvSourceLocal;

                P.restrictionRecvDestLocal = ...
                    restrictionRecvDestLocal;

                %% Basic checks
                assert(numel(P.nodGridId) == P.numNodes, ...
                    'GPU %d: nodGridId size does not match numNodes.', g-1);

                assert(all(P.restrictionFineNodeMapForward >= 0), ...
                    'GPU %d: invalid negative restriction map entry.', g-1);

                if ~isempty(P.restrictionFineNodeMapForward)
                    assert(max(P.restrictionFineNodeMapForward) <= ...
                           P.numRestrictionFineNodes, ...
                        'GPU %d: restriction map references invalid local node.', ...
                        g-1);
                end

                assert(sum(cellfun(@numel, P.restrictionRecvDestLocal)) == ...
                       P.numRestrictionFineNodes, ...
                    'GPU %d: restriction source mapping is incomplete.', g-1);

                for srcGPU = 1:numGPUs
                    assert(numel(P.restrictionRecvSourceLocal{srcGPU}) == ...
                           numel(P.restrictionRecvDestLocal{srcGPU}), ...
                        'GPU %d: restriction communication list mismatch.', g-1);

                    % Verify that every communication pair refers to the same
                    % physical active fine node.
                    if ~isempty(P.restrictionRecvSourceLocal{srcGPU})
                        sourceGlobalCheck = fineMesh.partitions{srcGPU}.globalNodeIds( ...
                            double(P.restrictionRecvSourceLocal{srcGPU}));

                        destGlobalCheck = P.restrictionFineNodeIds( ...
                            double(P.restrictionRecvDestLocal{srcGPU}));

                        assert(isequal(int32(sourceGlobalCheck(:)), ...
                                       int32(destGlobalCheck(:))), ...
                            ['GPU %d: restriction source/destination mapping ' ...
                             'does not refer to the same fine node.'], g-1);
                    end
                end

                meshHierarchy_(ii).partitions{g} = P;

                fprintf(['  GPU %d: coarse nodes = %d, ' ...
                         'fine restriction nodes = %d, halo = %d, ' ...
                         'slab = [%d %d %d] nodes\n'], ...
                    g-1, ...
                    P.numNodes, ...
                    P.numRestrictionFineNodes, ...
                    P.numRestrictionHaloNodes, ...
                    P.restrictionFineGridSize(1), ...
                    P.restrictionFineGridSize(2), ...
                    P.restrictionFineGridSize(3));

                clear fineSlabGlobalMap activeMask ...
                      restrictionFineNodeIds restrictionFineNodeMapForward ...
                      restrictionFineNodeSourceGPU restrictionFineNodeSourceLocal ...
                      restrictionRecvSourceLocal restrictionRecvDestLocal
            end

            clear fineMapVolume
        end




    end 

    clear -global eNodMatHalfTemp_



    %%Print Mesh Hierarchy

    disp('Mesh Hierarchy...');

    disp('             #Resolutions         #Elements   #DOFs');

    for ii=1:numel(meshHierarchy_)

        disp([sprintf('...Level %i', ii), sprintf(': %4i x %4i x %4i', [meshHierarchy_(ii).resX meshHierarchy_(ii).resY ...

            meshHierarchy_(ii).resZ]), sprintf(' %11i', meshHierarchy_(ii).numElements), sprintf(' %11i', meshHierarchy_(ii).numDOFs)]);

    end

end