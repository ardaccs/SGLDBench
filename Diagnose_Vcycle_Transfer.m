%% Diagnose CUDA V-cycle transfer, determinism, and symmetry
%
% Run after:
%   FEA_ApplyBoundaryCondition();
%   FEA_SetupVoxelBased();
%   Solving_AssembleFEAstencil();
%
% Uses:
%   sgld_vcycle3_exact_cuda_mex_fixed
%   Debug_CoarseSolve

global meshHierarchy_;
global weightFactorJacobi_;

mh1 = meshHierarchy_(1);
mh2 = meshHierarchy_(2);
mh3 = meshHierarchy_(3);

fixedMask1 = logical(mh1.fixedDOFs(:));
fixedMask2 = logical(mh2.fixedDOFs(:));
fixedMask3 = logical(mh3.fixedDOFs(:));

applyCuda = @(r) sgld_vcycle3_exact_cuda_mex_fixed( ...
    full(double(r(:))), ...
    mh2, ...
    mh3, ...
    full(double(mh1.diagK(:))), ...
    fixedMask1, ...
    full(double(mh2.diagK(:))), ...
    fixedMask2, ...
    fixedMask3, ...
    weightFactorJacobi_, ...
    @Debug_CoarseSolve);

rng(1);
r1 = randn(mh1.numDOFs,1);
r1(fixedMask1) = 0;

%% 1. Restriction comparison with the SAME boundary treatment
fprintf('\n=== Restriction validation ===\n');

r2_ref_raw = Solving_RestrictResidual(r1,2);
r2_ref = r2_ref_raw;
r2_ref(fixedMask2) = 0;

r3_ref_raw = Solving_RestrictResidual(r2_ref,3);
r3_ref = r3_ref_raw;
r3_ref(fixedMask3) = 0;

[z_cuda,r2_cuda,r3_cuda] = applyCuda(r1);

fprintf('Raw MATLAB vs constrained CUDA:\n');
fprintf('  r2 relErr raw: %.12e\n', ...
    norm(r2_cuda-r2_ref_raw)/max(norm(r2_ref_raw),1e-30));
fprintf('  r3 relErr raw: %.12e\n', ...
    norm(r3_cuda-r3_ref_raw)/max(norm(r3_ref_raw),1e-30));

fprintf('Same fixed-DOF treatment:\n');
fprintf('  r2 absErr: %.12e\n',norm(r2_cuda-r2_ref));
fprintf('  r2 relErr: %.12e\n', ...
    norm(r2_cuda-r2_ref)/max(norm(r2_ref),1e-30));
fprintf('  r3 absErr: %.12e\n',norm(r3_cuda-r3_ref));
fprintf('  r3 relErr: %.12e\n', ...
    norm(r3_cuda-r3_ref)/max(norm(r3_ref),1e-30));

fprintf('Difference concentrated at fixed DOFs:\n');
fprintf('  norm raw r2 on fixed DOFs: %.12e\n',norm(r2_ref_raw(fixedMask2)));
fprintf('  norm raw r3 on fixed DOFs: %.12e\n',norm(r3_ref_raw(fixedMask3)));

%% 2. Full V-cycle comparison
fprintf('\n=== Full V-cycle validation ===\n');

z_ref = Solving_Vcycle(r1);

fprintf('z absErr: %.12e\n',norm(z_cuda-z_ref));
fprintf('z relErr: %.12e\n', ...
    norm(z_cuda-z_ref)/max(norm(z_ref),1e-30));
fprintf('r''z CUDA:   %.12e\n',r1'*z_cuda);
fprintf('r''z MATLAB: %.12e\n',r1'*z_ref);

%% 3. Repeated-call determinism
fprintf('\n=== Repeated-call determinism ===\n');

z_cuda_2 = applyCuda(r1);
z_cuda_3 = applyCuda(r1);

fprintf('call 1 vs 2 absErr: %.12e\n',norm(z_cuda-z_cuda_2));
fprintf('call 1 vs 2 relErr: %.12e\n', ...
    norm(z_cuda-z_cuda_2)/max(norm(z_cuda),1e-30));

fprintf('call 2 vs 3 absErr: %.12e\n',norm(z_cuda_2-z_cuda_3));
fprintf('call 2 vs 3 relErr: %.12e\n', ...
    norm(z_cuda_2-z_cuda_3)/max(norm(z_cuda_2),1e-30));

%% 4. Symmetry test: x' M y should equal y' M x
fprintf('\n=== Preconditioner symmetry test ===\n');

rng(2);
x = randn(mh1.numDOFs,1);
y = randn(mh1.numDOFs,1);
x(fixedMask1) = 0;
y(fixedMask1) = 0;

Mx_cuda = applyCuda(x);
My_cuda = applyCuda(y);

lhs_cuda = x' * My_cuda;
rhs_cuda = y' * Mx_cuda;
sym_cuda = abs(lhs_cuda-rhs_cuda) / ...
    max([abs(lhs_cuda),abs(rhs_cuda),1e-30]);

Mx_matlab = Solving_Vcycle(x);
My_matlab = Solving_Vcycle(y);

lhs_matlab = x' * My_matlab;
rhs_matlab = y' * Mx_matlab;
sym_matlab = abs(lhs_matlab-rhs_matlab) / ...
    max([abs(lhs_matlab),abs(rhs_matlab),1e-30]);

fprintf('CUDA x''My: %.12e\n',lhs_cuda);
fprintf('CUDA y''Mx: %.12e\n',rhs_cuda);
fprintf('CUDA symmetry rel diff: %.12e\n',sym_cuda);

fprintf('MATLAB x''My: %.12e\n',lhs_matlab);
fprintf('MATLAB y''Mx: %.12e\n',rhs_matlab);
fprintf('MATLAB symmetry rel diff: %.12e\n',sym_matlab);

fprintf('\nInterpretation:\n');
fprintf('  constrained r2/r3 relErr near 1e-12 => transfer is already exact.\n');
fprintf('  repeated-call relErr above ~1e-13 => atomic accumulation is variable.\n');
fprintf('  CUDA symmetry error much larger than MATLAB => standard PCG loses conjugacy.\n');
