function [bLocal, y0Local] = Split_GPU_Vectors(b, y0, H)
%SPLIT_GPU_VECTORS Convert global finest-level DOF vectors to GPU-local cells.
%
%   [bLocal, y0Local] = Split_GPU_Vectors(b, y0, H)
%
% The multigpu_initialization.inl parser requires b and y0 to be cell arrays
% with one 3*numNodes(1) double vector per GPU, in P.globalNodeIds{1} order.
% A node may occur in multiple partitions (shared/interface nodes); its
% three DOFs are copied to every partition containing the node.
%
% NOTE: This only distributes input vectors. The current multi-GPU MEX
% gateway must separately be updated to merge/output the GPU-local result.

    assert(isstruct(H) && isscalar(H) && isfield(H, 'partitions') && ...
           iscell(H.partitions) && ~isempty(H.partitions), ...
           'H must be the scalar structure returned by Build_GPU_Hierarchy.');
    assert(isa(b, 'double') && isreal(b) && isa(y0, 'double') && ...
           isreal(y0) && isvector(b) && isvector(y0) && numel(b)==numel(y0), ...
           'b and y0 must be real double global vectors of identical length.');
    ids = cellfun(@(p) double(p.globalNodeIds{1}(:)), H.partitions, ...
                  'UniformOutput', false);
    n = max(cellfun(@max, ids));
    assert(numel(b) == 3*n, ...
           'Global finest vector length must equal 3 * global active node count.');
    bNodes = reshape(b(:), 3, []);
    yNodes = reshape(y0(:), 3, []);
    numGPUs = numel(H.partitions);
    bLocal = cell(1, numGPUs);
    y0Local = cell(1, numGPUs);
    for g = 1:numGPUs
        localIds = ids{g};
        bLocal{g} = reshape(bNodes(:, localIds), [], 1);
        y0Local{g} = reshape(yNodes(:, localIds), [], 1);
    end
end
