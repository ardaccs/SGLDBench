function results = Validate_VCycle_Raw_MGPU(H, numGPUs, tolerance, cpuApply, splitAxis)
%VALIDATE_VCYCLE_RAW_MGPU Exact full-domain raw GPU vs original CPU V-cycle.
%
% After original MATLAB FEA setup and compiling Solving_VCycle_Test_MGPU:
%   H = Build_VCycle_Test_H();
%   results = Validate_VCycle_Raw_MGPU(H, 2, 1e-9);
%
% If Solving_Vcycle is a private/local MATLAB function, pass a callable
% handle from a scope that can see it:
%   results = Validate_VCycle_Raw_MGPU(H,2,1e-9,@Solving_Vcycle);
%
% No GPU numerical function is changed. The GPU harness implements only
% initialization, partitioning, coarse cuDSS setup, and MATLAB MEX glue.
if nargin < 3 || isempty(tolerance), tolerance=1e-9; end
if nargin < 4 || isempty(cpuApply), cpuApply=@Solving_Vcycle; end
if nargin < 5 || isempty(splitAxis), splitAxis=1; end
assert(tolerance>0 && isfinite(tolerance) && isscalar(tolerance));
assert(isa(cpuApply,'function_handle'));
assert(exist('Solving_VCycle_Test_MGPU','file')==3, 'Compile the test MEX first.');
assert(numGPUs>=1 && fix(numGPUs)==numGPUs);
assert(ismember(splitAxis,1:3), 'splitAxis must be 1, 2 or 3.');
N=double(H.numNodes(1));
n=3*N;
fixed=double(H.fixedDOFIds(:));

% Split boundaries match the test MEX: coarse split cuts multiplied by all
% span factors on the way to the finest grid.
if splitAxis==1, res=double(H.resX(end));
elseif splitAxis==2, res=double(H.resY(end));
else, res=double(H.resZ(end)); end
factor=prod(double(H.spanWidth(:)));
borders=floor(res*(1:numGPUs-1)/numGPUs)*factor;
gid=double(H.nodGridId{1}(:))-1;
ny=double(H.resY(1))+1; nx=double(H.resX(1))+1;
if splitAxis==1, coords=mod(floor(gid/ny),nx);
elseif splitAxis==2, coords=mod(gid,ny);
else, coords=floor(gid/(nx*ny)); end
if isempty(borders), boundaryNode=ceil(N/2); nearNodes=[];
else
    [~,boundaryNode]=min(abs(coords-borders(1)));
    nearNodes=find(any(abs(coords-borders)<=max(1,factor),2));
end
nearDOFs=reshape(3*nearNodes(:)' + (-2:0)',[],1);

rng(42,'twister');
results=struct('test',{},'maxAbs',{},'relL2',{},'nearBoundaryAbs',{}, ...
    'maxSharedNodeDifference',{},'gpuSeconds',{},'cpuSeconds',{},'pass',{});
fprintf('\n=== RAW V-CYCLE: Original CPU vs multi-GPU CUDA ===\n');
for t=1:4
    r=zeros(n,1);
    switch t
        case 1, name='random'; r=randn(n,1);
        case 2, name='constant'; r(:)=1;
        case 3
            name='boundary_impulse';
            r(3*boundaryNode-2:3*boundaryNode)=[1;-2;3];
        case 4
            name='sparse_random';
            idx=randperm(n,min(2000,n));r(idx)=randn(numel(idx),1);
    end
    r(fixed)=0;
    assert(any(r), 'Input is zero after fixed-DOF masking.');
    tGpu=tic;
    [zGPU,info]=Solving_VCycle_Test_MGPU(r,H,numGPUs,splitAxis);
    gpuSeconds=toc(tGpu);
    tCPU=tic;
    zCPU=cpuApply(r);
    cpuSeconds=toc(tCPU);
    zGPU=double(zGPU(:));zCPU=double(zCPU(:));
    assert(numel(zCPU)==n && numel(zGPU)==n, 'V-cycle output length mismatch');
    assert(all(isfinite(zCPU)) && all(isfinite(zGPU)), 'Nonfinite V-cycle output');
    err=abs(zGPU-zCPU);
    maxAbs=max(err);
    relL2=norm(zGPU-zCPU)/max(norm(zCPU),eps);
    near=0;
    if ~isempty(nearDOFs),near=max(err(nearDOFs));end
    scale=max(1,max(abs(zCPU)));
    passed=maxAbs <= tolerance*scale && relL2 <= tolerance && ...
        info.maxSharedNodeDifference <= tolerance*scale;
    results(end+1)=struct('test',name,'maxAbs',maxAbs,'relL2',relL2, ...
        'nearBoundaryAbs',near,'maxSharedNodeDifference',info.maxSharedNodeDifference, ...
        'gpuSeconds',gpuSeconds,'cpuSeconds',cpuSeconds,'pass',passed); %#ok<AGROW>
    if passed, state='PASS'; else, state='FAIL'; end
    fprintf('  %-18s maxAbs %.3e relL2 %.3e near %.3e shared %.3e [%s]\n',...
        name,maxAbs,relL2,near,info.maxSharedNodeDifference,state);
end
assert(all([results.pass]), 'Raw V-cycle validation FAILED; see diagnostics above.');
fprintf('PASS: all full-domain raw V-cycle outputs agree within tolerance.\n');
end
