%% DEMO: Stress Analysis
clear all; clc;
addpath('../');
addpath('../src/');
addpath('../src/MEXfuncs/');

Data_GlobalVariables;
numGPUs = 2;
outPath_ = '../out/';
if ~exist(outPath_, 'dir'), mkdir(outPath_); end

%%1. Data Loading
tStart = tic;
MdlSelect = 'Bone';
IO_LoadBuiltInDatasets(MdlSelect, 512, numGPUs);
disp(['Prepare Voxel Model Costs: ', sprintf('%10.3g',toc(tStart)) 's']);

% figure; view(gca,3);
% Vis_DrawMesh3D(gca, meshHierarchy_(1).boundaryNodeCoords, meshHierarchy_(1).boundaryEleFaces, 0);
% Vis_ShowLoadingCondition(gca, loadingCond_);
% Vis_ShowFixingCondition(gca, fixingCond_);

%% 2. Setup FEA
tStart = tic;
FEA_ApplyBoundaryCondition();
FEA_SetupVoxelBased(numGPUs);
densityField = ones(meshHierarchy_(1).numElements,1); %%fully solid domain
meshHierarchy_(1).eleModulus = TopOpti_MaterialInterpolationSIMP(densityField(:));
disp(['Setup FEA Costs: ', sprintf('%10.3g',toc(tStart)) 's']);

%% 3. Assemble Computing Stencil
tStart = tic;
Solving_AssembleFEAstencil();
disp(['Assemble Computing Stencil Costs: ', sprintf('%10.3g',toc(tStart)) 's']);

%% 4. Multiple GPU Implementation

%% Get the longest axis of the mesh

resX = meshHierarchy_(1).resX;
resY = meshHierarchy_(1).resY;
resZ = meshHierarchy_(1).resZ;

values = [resX, resY, resZ];
names = {'resX', 'resY', 'resZ'};

[maxValue, idx] = max(values);
maxName = names{idx};

fprintf('\nLongest axis: %s = %g\n', maxName, maxValue);

%% Split the mesh along the longest axis

numGPUs = 2;

edges = round(linspace(0, maxValue, numGPUs + 1));

fprintf('Partition edges: ');
fprintf('%d ', edges);
fprintf('\n');


%% ------------------------------------------------------------
%% SANITY CHECKS FOR MULTI-GPU PARTITION
%% ------------------------------------------------------------

fprintf('\n');
fprintf('========================================\n');
fprintf(' MULTI-GPU PARTITION SANITY CHECK\n');
fprintf('========================================\n');

mesh = meshHierarchy_(1);
P = mesh.partitions;


%% 1. Number of partitions

assert(numel(P) == numGPUs, ...
    'Number of partitions does not match numGPUs.');

fprintf('[PASS] Number of partitions = %d\n', numGPUs);


%% 2. Print partition sizes

fprintf('\nPartition sizes:\n');

totalPartitionElements = 0;

for g = 1:numGPUs

    fprintf(['GPU %d:\n' ...
             '    Elements : %d\n' ...
             '    Nodes    : %d\n' ...
             '    Range    : [%d, %d]\n'], ...
             g-1, ...
             P{g}.numElements, ...
             P{g}.numNodes, ...
             P{g}.range(1), ...
             P{g}.range(2));

    totalPartitionElements = ...
        totalPartitionElements + P{g}.numElements;
end


%% 3. Every global active element must appear exactly once

allElementIds = [];

for g = 1:numGPUs
    allElementIds = ...
        [allElementIds; double(P{g}.elementIds(:))];
end

assert(numel(allElementIds) == mesh.numElements, ...
    ['Partition element count does not equal global ' ...
     'element count.']);

assert(numel(unique(allElementIds)) == mesh.numElements, ...
    'Some elements appear in more than one partition.');

assert(all(sort(allElementIds) == (1:mesh.numElements)'), ...
    'Some global active elements are missing.');

fprintf('[PASS] Every active element belongs to exactly one GPU.\n');


%% 4. Element count must equal global element count

assert(totalPartitionElements == mesh.numElements, ...
    'Sum of partition element counts != global numElements.');

fprintf('[PASS] Element counts: %d == %d\n', ...
    totalPartitionElements, mesh.numElements);


%% 5. Check partition ranges

for g = 1:numGPUs

    expectedLower = edges(g) + 1;
    expectedUpper = edges(g+1);

    assert(P{g}.range(1) == expectedLower, ...
        'GPU %d has incorrect lower range.', g-1);

    assert(P{g}.range(2) == expectedUpper, ...
        'GPU %d has incorrect upper range.', g-1);

end

fprintf('[PASS] Partition ranges match requested edges.\n');


%% 6. eNodMat dimensions

for g = 1:numGPUs

    assert(size(P{g}.eNodMat,1) == P{g}.numElements, ...
        'GPU %d: eNodMat row count incorrect.', g-1);

    assert(size(P{g}.eNodMat,2) == 8, ...
        'GPU %d: eNodMat must have 8 columns.', g-1);

end

fprintf('[PASS] All local eNodMat arrays have correct dimensions.\n');


%% 7. Local eNodMat node indices must be valid

for g = 1:numGPUs

    localNodes = P{g}.eNodMat(:);

    assert(all(localNodes >= 1), ...
        'GPU %d: eNodMat contains node index < 1.', g-1);

    assert(all(localNodes <= P{g}.numNodes), ...
        'GPU %d: eNodMat contains node index > numNodes.', g-1);

end

fprintf('[PASS] All local eNodMat node indices are valid.\n');


%% 8. Every local node should actually be used

for g = 1:numGPUs

    usedNodes = unique(P{g}.eNodMat(:));

    expectedNodes = int32((1:P{g}.numNodes)');

    assert(isequal(usedNodes, expectedNodes), ...
        'GPU %d contains unused or missing local node IDs.', g-1);

end

fprintf('[PASS] Local node numbering is compact: 1...numNodes.\n');


%% 9. globalNodeIds must contain valid global node IDs

for g = 1:numGPUs

    ids = P{g}.globalNodeIds;

    assert(all(ids >= 1), ...
        'GPU %d contains global node ID < 1.', g-1);

    assert(all(ids <= mesh.numNodes), ...
        'GPU %d contains global node ID > global numNodes.', g-1);

    assert(numel(unique(ids)) == numel(ids), ...
        'GPU %d globalNodeIds contains duplicates.', g-1);

end

fprintf('[PASS] All globalNodeIds are valid and unique per GPU.\n');


%% 10. Check that local eNodMat maps back to global eNodMat

for g = 1:numGPUs

    globalElementIds = double(P{g}.elementIds);

    expectedGlobalENodMat = ...
        mesh.eNodMat(globalElementIds,:);

    reconstructedGlobalENodMat = ...
        P{g}.globalNodeIds(P{g}.eNodMat);

    assert(isequal( ...
        int32(reconstructedGlobalENodMat), ...
        int32(expectedGlobalENodMat)), ...
        'GPU %d: local/global eNodMat mapping is incorrect.', g-1);

end

fprintf('[PASS] Local eNodMat correctly reproduces global eNodMat.\n');


%% 11. nodeToElements dimensions and bounds

for g = 1:numGPUs

    N2E = P{g}.nodeToElements;

    assert(size(N2E,1) == P{g}.numNodes, ...
        'GPU %d: nodeToElements row count incorrect.', g-1);

    assert(size(N2E,2) == 8, ...
        'GPU %d: nodeToElements must have 8 columns.', g-1);

    validEntries = N2E(N2E ~= 0);

    assert(all(validEntries >= 1), ...
        'GPU %d: nodeToElements contains negative IDs.', g-1);

    assert(all(validEntries <= P{g}.numElements), ...
        'GPU %d: nodeToElements references invalid element.', g-1);

end

fprintf('[PASS] nodeToElements dimensions and element IDs are valid.\n');


%% 12. Verify nodeToElements against eNodMat
% Efficient version for very large meshes

for g = 1:numGPUs

    eNod = P{g}.eNodMat;
    N2E  = P{g}.nodeToElements;

    % Every nonzero nodeToElements(node,k) must point to an element
    % that actually contains that node.

    [nodeIds, slots] = find(N2E ~= 0);

    linearIds = sub2ind(size(N2E), nodeIds, slots);
    elementIds = N2E(linearIds);

    % Process in chunks to avoid huge temporary allocations
    chunkSize = 1e6;

    for startIdx = 1:chunkSize:numel(nodeIds)

        stopIdx = min(startIdx + chunkSize - 1, numel(nodeIds));

        nodesChunk = nodeIds(startIdx:stopIdx);
        elemsChunk = double(elementIds(startIdx:stopIdx));

        referencedENod = eNod(elemsChunk,:);

        valid = any(referencedENod == nodesChunk, 2);

        assert(all(valid), ...
            ['GPU %d: nodeToElements contains a node/element ' ...
             'pair inconsistent with eNodMat.'], ...
             g-1);

    end

    % Number of incidences must also match exactly
    expectedIncidences = numel(eNod);
    actualIncidences   = nnz(N2E);

    assert(actualIncidences == expectedIncidences, ...
        ['GPU %d: nodeToElements contains incorrect number ' ...
         'of node-element incidences.'], ...
         g-1);

end

fprintf('[PASS] nodeToElements agrees with eNodMat.\n');


%% 13. Check shared/interface nodes

fprintf('\nInterface information:\n');

for g = 1:numGPUs-1

    sharedNodes = intersect( ...
        P{g}.globalNodeIds, ...
        P{g+1}.globalNodeIds);

    fprintf('GPU %d <-> GPU %d: %d shared nodes\n', ...
        g-1, g, numel(sharedNodes));

    assert(~isempty(sharedNodes), ...
        ['No shared nodes between neighboring GPUs. ' ...
         'This is suspicious for a connected mesh.']);

end

fprintf('[PASS] Neighboring partitions share interface nodes.\n');


%% 14. Non-neighboring GPUs should normally not share nodes
% Relevant when using > 2 GPUs.

if numGPUs > 2

    for g1 = 1:numGPUs
        for g2 = g1+2:numGPUs

            sharedNodes = intersect( ...
                P{g1}.globalNodeIds, ...
                P{g2}.globalNodeIds);

            assert(isempty(sharedNodes), ...
                ['GPU %d and GPU %d are non-neighbors but ' ...
                 'share nodes.'], ...
                 g1-1, g2-1);

        end
    end

    fprintf('[PASS] Non-neighboring partitions do not share nodes.\n');

end


%% 15. Compare total local node count with unique global nodes

allGPUNodeIds = [];

for g = 1:numGPUs
    allGPUNodeIds = ...
        [allGPUNodeIds; double(P{g}.globalNodeIds(:))];
end

uniquePartitionNodes = unique(allGPUNodeIds);

assert(numel(uniquePartitionNodes) == mesh.numNodes, ...
    ['Union of partition nodes does not reproduce all ' ...
     'global active nodes.']);

fprintf('[PASS] Union of GPU nodes reproduces all global nodes.\n');


%% 16. Show duplication caused by interface nodes

totalLocalNodes = 0;

for g = 1:numGPUs
    totalLocalNodes = totalLocalNodes + P{g}.numNodes;
end

duplicatedNodes = totalLocalNodes - mesh.numNodes;

fprintf('\nNode statistics:\n');
fprintf('    Global nodes             : %d\n', mesh.numNodes);
fprintf('    Sum of GPU-local nodes   : %d\n', totalLocalNodes);
fprintf('    Interface duplication    : %d\n', duplicatedNodes);


fprintf('\n========================================\n');
fprintf(' ALL PARTITION SANITY CHECKS PASSED\n');
fprintf('========================================\n\n');

%% 17. MULTI-GPU K*U TEST (append after partition sanity checks)
% Compile separately, from the folder containing Solving_KbyU_MatrixFree_MGPU.cu:
%   mexcuda -output Solving_KbyU_MatrixFree_MGPU Solving_KbyU_MatrixFree_MGPU.cu
% This test uses the existing mesh and partitions; it does not rebuild them.
assert(numGPUs == 2, 'This MEX currently supports exactly two GPUs.');
assert(exist('Solving_KbyU_MatrixFree_MGPU', 'file') == 3, ...
    'Compile Solving_KbyU_MatrixFree_MGPU.cu with mexcuda and add its output folder to the MATLAB path.');

% Use the EXACT 24x24 stiffness matrix used by Solving_AssembleFEAstencil.
% If your project uses a different variable/field name, set KeTest here.
KeTest = [];
if isfield(mesh, 'Ke'),          KeTest = mesh.Ke;
elseif isfield(mesh, 'Ke_'),     KeTest = mesh.Ke_;
elseif exist('Ke_', 'var'),      KeTest = Ke_;
elseif exist('Ke', 'var'),       KeTest = Ke;
elseif exist('KE_', 'var'),      KeTest = KE_;
elseif exist('KE', 'var'),       KeTest = KE;
end
assert(isa(KeTest, 'double') && isreal(KeTest) && ...
       ~issparse(KeTest) && isequal(size(KeTest), [24,24]), ...
       ['Cannot locate the 24x24 double element stiffness matrix. ' ...
        'Set KeTest to the Ke used in Solving_AssembleFEAstencil.']);
assert(numel(mesh.eleModulus) == mesh.numElements, ...
    'Global eleModulus must have one entry per active element.');

% Build the H(1:2) struct expected by the current MEX. IMPORTANT:
% sharedNodesLocal{otherGPU} is indexed in the SAME global-node order
% for both partitions; intersect returns sorted matching global IDs.
H = repmat(struct('numNodes', 0, 'numElements', 0, ...
    'nodeToElements', [], 'eNodMat', [], 'eleModulus', [], ...
    'sharedNodesLocal', {{[], []}}), 1, numGPUs);
[sharedGlobal, sharedLocal1, sharedLocal2] = intersect( ...
    P{1}.globalNodeIds(:), P{2}.globalNodeIds(:));
assert(~isempty(sharedGlobal), 'No shared interface nodes.');
for g = 1:numGPUs
    H(g).numNodes       = double(P{g}.numNodes);
    H(g).numElements    = double(P{g}.numElements);
    H(g).nodeToElements = int32(P{g}.nodeToElements);
    H(g).eNodMat        = int32(P{g}.eNodMat);
    H(g).eleModulus     = double(mesh.eleModulus(double(P{g}.elementIds(:))));
    H(g).sharedNodesLocal = cell(1, numGPUs);
end
H(1).sharedNodesLocal{2} = int32(sharedLocal1(:));
H(2).sharedNodesLocal{1} = int32(sharedLocal2(:));
assert(isequal(P{1}.globalNodeIds(sharedLocal1), ...
               P{2}.globalNodeIds(sharedLocal2)), ...
       'Interface node ordering is inconsistent.');

% One global displacement/load, duplicated into each partition's local DOFs.
% Three consecutive DOFs belong to each node: [ux uy uz].
nGlobalDOFs = 3 * double(mesh.numNodes);
idxGlobal = (1:nGlobalDOFs)';
uGlobal = sin(idxGlobal * 0.013);
bGlobal = cos(idxGlobal * 0.017);
clear idxGlobal
localDOF = cell(1, numGPUs);
for g = 1:numGPUs
    nodeIDs = double(P{g}.globalNodeIds(:));
    localDOF{g} = reshape(3*nodeIDs.' + [-2;-1;0], [], 1);
end
y0 = [uGlobal(localDOF{1}); uGlobal(localDOF{2})];
b  = [bGlobal(localDOF{1}); bGlobal(localDOF{2})];
clear uGlobal bGlobal

fprintf('\n========================================\n');
fprintf(' TESTING TWO-GPU MATRIX-VECTOR PRODUCT\n');
fprintf('========================================\n');
tGPU = tic;
[Y, r] = Solving_KbyU_MatrixFree_MGPU( ...
    H(1), b, y0, KeTest, numGPUs);
fprintf('MEX elapsed time (includes allocation/transfers): %.3f s\n', toc(tGPU));
assert(isequal(size(Y), size(y0)) && isequal(size(r), size(b)), ...
       'MEX output dimensions are incorrect.');
assert(all(isfinite(Y)) && all(isfinite(r)), ...
       'MEX returned NaN or Inf.');
residualError = max(abs(r - (b - Y)));
assert(residualError < 1e-12 * max(1,max(abs(b))), ...
       'Residual r is not b - Y.');
fprintf('[PASS] Output dimensions, finiteness and residual.\n');

% The MEX returns one local vector per partition, with shared-node sums
% copied into BOTH local vectors. Check all interface DOFs.
offset2 = 3 * double(P{1}.numNodes);
interfaceDOF1 = reshape(3*double(sharedLocal1(:)).' + [-2;-1;0], [], 1);
interfaceDOF2 = offset2 + reshape(3*double(sharedLocal2(:)).' + [-2;-1;0], [], 1);
interfaceError = max(abs(Y(interfaceDOF1) - Y(interfaceDOF2)));
assert(interfaceError < 1e-12 * max(1,max(abs(Y(interfaceDOF1)))), ...
       'The two copies of interface K*U do not agree.');
fprintf('[PASS] %d interface nodes agree across both GPUs.\n', numel(sharedGlobal));

% Independent CPU reference for sampled GLOBAL nodes. We evaluate the
% local element contributions directly using eNodMat, Ke and eleModulus;
% where a node belongs to both partitions, add BOTH partial contributions.
% Sampling avoids assembling/storing a huge global sparse stiffness matrix.
rng(42);
sampleIDs = unique([double(sharedGlobal(1:min(32,end))); ...
                    randi(double(mesh.numNodes), min(64,double(mesh.numNodes)),1)]);
globalToLocal = zeros(double(mesh.numNodes), numGPUs, 'int32');
for g = 1:numGPUs
    globalToLocal(double(P{g}.globalNodeIds(:)),g) = int32(1:P{g}.numNodes);
end
maxAbsError = 0;
maxRelError = 0;
for k = 1:numel(sampleIDs)
    globalNode = sampleIDs(k);
    yReference = zeros(3,1);
    yGPU = [];
    for g = 1:numGPUs
        localNode = double(globalToLocal(globalNode,g));
        if localNode == 0, continue; end
        % CPU reference: same mathematical Ke * Ue but independent of CUDA.
        incident = double(H(g).nodeToElements(localNode,:));
        incident = incident(incident > 0);
        for ee = incident
            elemNodes = double(H(g).eNodMat(ee,:));
            localRow = find(elemNodes == localNode, 1);
            assert(~isempty(localRow), 'Broken nodeToElements in CPU reference.');
            elementDOFs = reshape(3*elemNodes + [-2;-1;0], [], 1);
            Ue = y0(elementDOFs + (g==2)*offset2);
            yReference = yReference + H(g).eleModulus(ee) * ...
                (KeTest(3*localRow-2:3*localRow,:) * Ue);
        end
        base = (g==2)*offset2 + 3*(localNode-1);
        yGPU = Y(base+(1:3));
    end
    assert(~isempty(yGPU), 'Sample node not assigned to either GPU.');
    absError = max(abs(yGPU - yReference));
    relError = absError / max(1, max(abs(yReference)));
    maxAbsError = max(maxAbsError, absError);
    maxRelError = max(maxRelError, relError);
end
fprintf('Sampled nodes: %d; max abs error: %.3e; max scaled error: %.3e\n', ...
        numel(sampleIDs), maxAbsError, maxRelError);
assert(maxRelError < 1e-9, 'GPU K*U differs from sampled CPU reference.');
fprintf('[PASS] Sampled CPU reference matches the two-GPU result.\n');
fprintf(' ALL MULTI-GPU K*U TESTS PASSED\n\n');
