function maxAbsError = Validate_RestrictResidual_MGPU(H, numGPUs, fineResidual, level, numSamples)
% Compare a sampled set of nodes with the unpartitioned CPU stencil.
% Use a small test hierarchy first; large GPU runs should be checked separately.
if nargin < 4 || isempty(level), level = 1; end
if nargin < 5 || isempty(numSamples), numSamples = 64; end

fineResidual = fineResidual(:);
computed = Solving_RestrictResidual_MGPU(H, numGPUs, fineResidual, level);
fine = H(level);
coarse = H(level+1);
spanWidth = double(coarse.spanWidth);
radius = spanWidth - 1;

if coarse.numNodes == 0
    maxAbsError = 0;
    return;
end
samples = unique(round(linspace(1, double(coarse.numNodes), ...
                                min(numSamples, double(coarse.numNodes)))));
maxAbsError = 0;
for coarseNode = samples
    coarseGrid = double(coarse.nodGridId(coarseNode)) - 1;
    y = mod(coarseGrid, double(coarse.resY)+1);
    x = mod(floor(coarseGrid / (double(coarse.resY)+1)), double(coarse.resX)+1);
    z = floor(coarseGrid / ((double(coarse.resY)+1)*(double(coarse.resX)+1)));
    center = spanWidth * [y,x,z];
    expected = zeros(3,1);
    for dz = -radius:radius
        for dx = -radius:radius
            for dy = -radius:radius
                f = center + [dy,dx,dz];
                if any(f < 0) || f(1) > double(fine.resY) || ...
                        f(2) > double(fine.resX) || f(3) > double(fine.resZ)
                    continue;
                end
                fineGridId = f(1) + f(2)*(double(fine.resY)+1) + ...
                    f(3)*(double(fine.resY)+1)*(double(fine.resX)+1) + 1;
                fineNode = double(fine.nodMapForward(fineGridId));
                if fineNode == 0, continue; end
                weight = (1-abs(dy)/spanWidth) * ...
                         (1-abs(dx)/spanWidth) * ...
                         (1-abs(dz)/spanWidth);
                expected = expected + weight * ...
                    fineResidual((3*fineNode-2):(3*fineNode));
            end
        end
    end
    actual = computed((3*coarseNode-2):(3*coarseNode));
    maxAbsError = max(maxAbsError, max(abs(actual-expected)));
end
fprintf('Maximum sampled restriction error: %.3e (%d nodes)\n', ...
    maxAbsError, numel(samples));
end
