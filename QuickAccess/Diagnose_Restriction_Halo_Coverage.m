function Diagnose_Restriction_Halo_Coverage(H, numGPUs, level)
% Check whether each coarse node's full active fine stencil exists in its
% owning GPU's normal+halo map. No GPU, no kernel changes required.
if nargin < 3, level = 1; end
fine = H(level); coarse = H(level+1);
s = double(coarse.spanWidth);
r = s-1;
fy = double(fine.resY)+1; fx = double(fine.resX)+1; fz = double(fine.resZ)+1;
cy = double(coarse.resY)+1; cx = double(coarse.resX)+1;
assert(isscalar(s) && s >= 1 && s == floor(s));
fGlobal = fine.nodMapForward(:);
for g=1:numGPUs
    P = coarse.partitions{g}; F = fine.partitions{g};
    ids = double(P.nodGridId(:)) - 1;
    y = mod(ids, cy); x = mod(floor(ids/cy), cx); z = floor(ids/(cy*cx));
    localMap = F.nodMapForwardWithHalo(:);
    bad = false(numel(ids),1);
    nMiss = 0;
    first = zeros(0,5);
    for dz=-r:r
        zz=s*z+dz;
        for dx=-r:r
            xx=s*x+dx;
            for dy=-r:r
                yy=s*y+dy;
                valid = yy>=0 & yy<fy & xx>=0 & xx<fx & zz>=0 & zz<fz;
                idx = find(valid);
                if isempty(idx), continue; end
                fullId = 1+yy(idx)+fy*xx(idx)+fy*fx*zz(idx);
                missing = fGlobal(fullId)~=0 & localMap(fullId)==0;
                if any(missing)
                    hits = idx(missing);
                    nMiss = nMiss + numel(hits);
                    bad(hits) = true;
                    if size(first,1)<10
                        take = hits(1:min(10-size(first,1),numel(hits)));
                        first = [first; [double(P.globalNodeIds(take)), yy(take), xx(take), zz(take), repmat(g,numel(take),1)]]; %#ok<AGROW>
                    end
                end
            end
        end
    end
    fprintf('L%d -> L%d, GPU%d: missing stencil entries=%d; affected coarse nodes=%d/%d; haloWidth=%d\n', ...
        level, level+1, g-1, nMiss, nnz(bad), numel(ids), double(H(level).haloWidth));
    if ~isempty(first)
        fprintf('First missing entries [global coarse node, fine Y, X, Z, GPU(1-based)]:\n');
        disp(first);
    end
end
end
