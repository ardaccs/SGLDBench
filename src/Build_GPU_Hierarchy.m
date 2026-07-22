function H = Build_GPU_Hierarchy( ...
    meshHierarchy_)

    numLevels = numel(meshHierarchy_);

    H.nodeToElements = cell(1, numLevels);
    H.eNodMat        = cell(1, numLevels);
    H.eleModulus     = cell(1, numLevels);

    H.nodGridId      = cell(1, numLevels);
    H.nodMapForward  = cell(1, numLevels);
    H.diagK          = cell(1, numLevels);

    H.resX        = zeros(1, numLevels, 'int32');
    H.resY        = zeros(1, numLevels, 'int32');
    H.resZ        = zeros(1, numLevels, 'int32');
    H.numNodes    = zeros(1, numLevels, 'int32');
    H.numElements = zeros(1, numLevels, 'int32');
    H.numDOFs     = zeros(1, numLevels, 'int32');

    for level = 1:numLevels

        H.nodeToElements{level} = ...
            int32(meshHierarchy_(level).nodeToElements);

        H.eNodMat{level} = ...
            int32(meshHierarchy_(level).eNodMat);

        H.nodGridId{level} = ...
            int32(meshHierarchy_(level).nodGridId(:));

        H.nodMapForward{level} = ...
            int32(meshHierarchy_(level).nodMapForward(:));

        H.resX(level) = ...
            int32(meshHierarchy_(level).resX);

        H.resY(level) = ...
            int32(meshHierarchy_(level).resY);

        H.resZ(level) = ...
            int32(meshHierarchy_(level).resZ);

        H.numNodes(level) = ...
            int32(meshHierarchy_(level).numNodes);

        H.numElements(level) = ...
            int32(meshHierarchy_(level).numElements);

        H.numDOFs(level) = ...
            int32(meshHierarchy_(level).numDOFs);

        % Only non-coarsest levels use diagK.
        if level < numLevels
            H.diagK{level} = ...
                double(meshHierarchy_(level).diagK(:));
        else
            H.diagK{level} = [];
        end

        % eleModulus is required by K*U.
        %
        % Your coarse levels currently show eleModulus = [].
        % Therefore this must be resolved before applying kbyu_kernel
        % on those levels. The current V-cycle does not need coarse
        % K*U, so only level 1 requires eleModulus for MGPCG.
        if isempty(meshHierarchy_(level).eleModulus)
            H.eleModulus{level} = [];
        else
            H.eleModulus{level} = ...
                double(meshHierarchy_(level).eleModulus(:));
        end
    end

    H.spanWidth = zeros(1, numLevels - 1, 'int32');

    for fineLevel = 1:numLevels - 1
        H.spanWidth(fineLevel) = ...
            int32(meshHierarchy_(fineLevel + 1).spanWidth);
    end

    H.Ke = double(meshHierarchy_(1).Ke);

    H.fixedDOFIds = ...
        int32(find(meshHierarchy_(1).fixedDOFs));

    H.coarseFreeDOFIds = ...
        int32(find(meshHierarchy_(end).freeDOFs));

end