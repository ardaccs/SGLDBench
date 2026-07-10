function Y = Debug_KbyU_VertexBased_MATLAB(U, nodeToElements, eNodMat, E, Ke)
    numNodes = size(nodeToElements, 1);
    numElements = size(eNodMat, 1);

    Y = zeros(3*numNodes, 1);

    for node = 1:numNodes
        sum3 = zeros(3,1);

        for a = 1:8
            elem = nodeToElements(node, a);

            if elem <= 0 || elem > numElements
                continue;
            end

            elemNodes = eNodMat(elem, :);

            localNode = find(elemNodes == node, 1);

            if isempty(localNode)
                continue;
            end

            Ue = zeros(24,1);

            for j = 1:8
                n = elemNodes(j);
                base = 3*(n-1);

                Ue(3*j - 2) = U(base + 1);
                Ue(3*j - 1) = U(base + 2);
                Ue(3*j    ) = U(base + 3);
            end

            row0 = 3*(localNode-1) + 1;

            Ye3 = E(elem) * Ke(row0:row0+2, :) * Ue;

            sum3 = sum3 + Ye3;
        end

        out = 3*(node-1);
        Y(out+1) = sum3(1);
        Y(out+2) = sum3(2);
        Y(out+3) = sum3(3);
    end
end