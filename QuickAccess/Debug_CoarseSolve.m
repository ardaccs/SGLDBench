function xCoarsest = Debug_CoarseSolve(rCoarsest)
% Exact coarsest solve used by Solving_Vcycle.
    global meshHierarchy_;
    global cholFac_;
    global cholPermut_;

    mh = meshHierarchy_(end);
    xCoarsest = zeros(mh.numDOFs,1);
    rhsFree = rCoarsest(mh.freeDOFs);
    xCoarsest(mh.freeDOFs) = ...
        cholPermut_ * (cholFac_' \ (cholFac_ \ (cholPermut_' * rhsFree)));
end
