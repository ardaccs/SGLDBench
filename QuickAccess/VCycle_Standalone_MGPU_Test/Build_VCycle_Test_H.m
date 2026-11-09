function H = Build_VCycle_Test_H(coarseKFree)
%BUILD_VCYCLE_TEST_H Pack the EXISTING CPU global mesh into the raw test MEX.
% Run AFTER your original MATLAB FEA / V-cycle setup.
%   H = Build_VCycle_Test_H();
% Or if a verified coarsest free-DOF matrix is already available:
%   H = Build_VCycle_Test_H(coarseKFree);
%
% Does not change any CPU global or GPU computational function.
global meshHierarchy_ numLevels_ weightFactorJacobi_ cholFac_ cholPermut_
assert(~isempty(meshHierarchy_) && ~isempty(numLevels_), 'CPU hierarchy must be initialized.');
assert(numel(meshHierarchy_) == double(numLevels_), 'Inconsistent numLevels_.');
assert(~isempty(weightFactorJacobi_), 'Missing weightFactorJacobi_.');
N = double(numLevels_);
H.numLevels = int32(N);
H.resX = zeros(N,1,'int32');
H.resY = zeros(N,1,'int32');
H.resZ = zeros(N,1,'int32');
H.numNodes = zeros(N,1,'int32');
H.numDOFs = zeros(N,1,'int32');
H.spanWidth = zeros(N-1,1,'int32');
H.nodGridId = cell(N,1);
H.nodMapForward = cell(N,1);
H.diagK = cell(N-1,1);
H.jacobiOmega = double(weightFactorJacobi_);

for lev = 1:N
    L = meshHierarchy_(lev);
    H.resX(lev) = int32(L.resX);
    H.resY(lev) = int32(L.resY);
    H.resZ(lev) = int32(L.resZ);
    H.numNodes(lev) = int32(L.numNodes);
    H.numDOFs(lev) = int32(L.numDOFs);
    H.nodGridId{lev} = int32(L.nodGridId(:));
    H.nodMapForward{lev} = int32(L.nodMapForward(:));
    if lev < N
        H.diagK{lev} = double(L.diagK(:));
        H.spanWidth(lev) = int32(meshHierarchy_(lev+1).spanWidth);
    end
end
H.fixedDOFIds = int32(meshHierarchy_(1).fixedDOFs(:));
H.coarseFreeDOFIds = int32(meshHierarchy_(N).freeDOFs(:));

if nargin < 1 || isempty(coarseKFree)
    % The original CPU solve does: P * (L' \ (L \ (P' * b))).
    % Thus its matrix is K = P*(L*L')*P'. Reconstruct ONLY the coarse
    % matrix for the test MEX; the original CPU functions are unchanged.
    assert(~isempty(cholFac_) && ~isempty(cholPermut_), ...
        'Provide coarseKFree or initialize original cholFac_/cholPermut_.');
    coarseKFree = cholPermut_ * (cholFac_ * cholFac_') * cholPermut_';
end
H.coarseKFree = sparse(double(coarseKFree));
assert(isequal(size(H.coarseKFree), repmat(numel(H.coarseFreeDOFIds),1,2)), ...
    'coarseKFree dimension must match the coarsest free DOFs.');
end
