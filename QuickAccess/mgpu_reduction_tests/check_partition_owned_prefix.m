% Run AFTER Solving_BuildingMeshHierarchy_MGPU() has populated meshHierarchy_.
% Verifies the assumption needed by d_owned = d_all + 3*numLeftSharedNodes.
global meshHierarchy_
for lev=1:numel(meshHierarchy_)
    Ps=meshHierarchy_(lev).partitions;
    nGlobal=double(meshHierarchy_(lev).numNodes);
    ownerCounts=zeros(nGlobal,1);
    fprintf('\nLevel %d: %d global nodes, %d partitions\n',lev,nGlobal,numel(Ps));
    for g=1:numel(Ps)
        local=double(Ps{g}.globalNodeIds(:));
        if g>1
            left=intersect(local,double(Ps{g-1}.globalNodeIds(:)));
        else
            left=zeros(0,1);
        end
        if g<numel(Ps)
            right=intersect(local,double(Ps{g+1}.globalNodeIds(:)));
        else
            right=zeros(0,1);
        end
        assert(isempty(intersect(left,right)), 'Level %d GPU %d: triple-shared node / overlapping interfaces',lev,g);
        assert(isequal(local(1:numel(left)),left(:)), ...
            'Level %d GPU %d: left shared nodes are NOT a contiguous prefix',lev,g);
        owned=local(numel(left)+1:end);
        ownerCounts(owned)=ownerCounts(owned)+1;
        fprintf(' GPU %d: local=%d leftExcluded=%d rightShared=%d owned=%d\n', ...
            g-1,numel(local),numel(left),numel(right),numel(owned));
    end
    assert(all(ownerCounts==1),'Level %d: global DOFs are not uniquely owned',lev);
    fprintf(' Level %d: PASS: ownership covers each global node exactly once\n',lev);
end
