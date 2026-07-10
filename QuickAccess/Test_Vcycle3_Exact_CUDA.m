%% Test exact-transfer CUDA V-cycle against MATLAB Solving_Vcycle
global meshHierarchy_;
global weightFactorJacobi_;

mh1 = meshHierarchy_(1);
mh2 = meshHierarchy_(2);
mh3 = meshHierarchy_(3);

rng(1);
r1 = randn(mh1.numDOFs,1);
r1(mh1.fixedDOFs) = 0;

fixedMask1 = false(mh1.numDOFs,1);
fixedMask1(mh1.fixedDOFs) = true;

fixedMask2 = false(mh2.numDOFs,1);
fixedMask2(mh2.fixedDOFs) = true;

fixedMask3 = false(mh3.numDOFs,1);
fixedMask3(mh3.fixedDOFs) = true;

fprintf('Computing MATLAB restriction vectors...\n');
r2_ref = Solving_RestrictResidual(r1,2);
r3_ref = Solving_RestrictResidual(r2_ref,3);

fprintf('Computing CUDA V-cycle...\n');

[z_cuda,r2_cuda,r3_cuda] = sgld_vcycle3_exact_cuda_mex_fixed( ...
    full(double(r1(:))), ...
    mh2, ...
    mh3, ...
    full(double(mh1.diagK(:))), ...
    fixedMask1, ...
    full(double(mh2.diagK(:))), ...
    fixedMask2, ...
    fixedMask3, ...
    weightFactorJacobi_, ...
    @Debug_CoarseSolve);

fprintf('\nRestriction checks\n');
fprintf('r2 absErr: %.12e\n',norm(r2_cuda-r2_ref));
fprintf('r2 relErr: %.12e\n',norm(r2_cuda-r2_ref)/max(norm(r2_ref),1e-30));
fprintf('r3 absErr: %.12e\n',norm(r3_cuda-r3_ref));
fprintf('r3 relErr: %.12e\n',norm(r3_cuda-r3_ref)/max(norm(r3_ref),1e-30));

fprintf('\nFull V-cycle check\n');
z_ref = Solving_Vcycle(r1);
fprintf('z absErr: %.12e\n',norm(z_cuda-z_ref));
fprintf('z relErr: %.12e\n',norm(z_cuda-z_ref)/max(norm(z_ref),1e-30));
fprintf('r''z CUDA: %.12e\n',r1'*z_cuda);
fprintf('r''z MATLAB: %.12e\n',r1'*z_ref);

[~,idx]=max(abs(z_cuda-z_ref));
fprintf('max diff DOF %d: CUDA %.12e, MATLAB %.12e, diff %.12e\n', ...
    idx,z_cuda(idx),z_ref(idx),z_cuda(idx)-z_ref(idx));
