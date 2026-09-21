function H = Build_GPU_Hierarchy(meshHierarchy_, numGPUs)

    % For now we only build data required by finest-level KbyU.
    mesh = meshHierarchy_(1);

    H = repmat(struct(), 1, numGPUs);

    %% ============================================================
    % 1. Build local GPU data
    %% ============================================================

    for g = 1:numGPUs

        P = mesh.partitions{g};

        % Local node -> local elements
        H(g).nodeToElements = int32(P.nodeToElements);

        % Local element -> local nodes
        H(g).eNodMat = int32(P.eNodMat);

        % Element modulus belonging only to this partition
        H(g).eleModulus = double(mesh.eleModulus(P.elementIds));

        % Local sizes
        H(g).numNodes = int32(P.numNodes);
        H(g).numElements = int32(P.numElements);

        % sharedNodesLocal{otherGPU}
        H(g).sharedNodesLocal = cell(1, numGPUs);

    end


    %% ============================================================
    % 2. Find shared/interface nodes
    %% ============================================================

    for g1 = 1:numGPUs

        for g2 = g1+1:numGPUs

            globalNodes1 = mesh.partitions{g1}.globalNodeIds;

            globalNodes2 = mesh.partitions{g2}.globalNodeIds;


            % Shared GLOBAL node IDs.
            %
            % We only use these temporarily here.
            % They are NOT stored inside H.
            sharedGlobalNodes = intersect( globalNodes1, globalNodes2);

            % Find where those shared nodes live in GPU g1's
            % local node numbering.
            [~, localNodes1] = ismember( sharedGlobalNodes, globalNodes1);


            % Same thing for GPU g2.
            [~, localNodes2] = ismember( sharedGlobalNodes, globalNodes2);


            % Store only LOCAL node IDs.
            %
            % IMPORTANT:
            %
            % H(g1).sharedNodesLocal{g2}(k)
            %
            % and
            %
            % H(g2).sharedNodesLocal{g1}(k)
            %
            % refer to the SAME physical/global node.

            H(g1).sharedNodesLocal{g2} = int32(localNodes1);

            H(g2).sharedNodesLocal{g1} = int32(localNodes2);

        end
    end


    %% ============================================================
    % 3. Basic checks
    %% ============================================================

    for g = 1:numGPUs

        assert(size(H(g).nodeToElements, 1) == ...
               double(H(g).numNodes));

        assert(size(H(g).nodeToElements, 2) == 8);

        assert(size(H(g).eNodMat, 1) == ...
               double(H(g).numElements));

        assert(size(H(g).eNodMat, 2) == 8);

        assert(numel(H(g).eleModulus) == ...
               double(H(g).numElements));

    end


    %% Check shared-node correspondence

    for g1 = 1:numGPUs

        for g2 = g1+1:numGPUs

            n1 = H(g1).sharedNodesLocal{g2};
            n2 = H(g2).sharedNodesLocal{g1};

            assert(numel(n1) == numel(n2), ...
                'Shared-node mappings have inconsistent sizes.');

        end

    end

end