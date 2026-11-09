function build_VCycle_Test_MGPU(cudssRoot, ncclRoot, cudaRoot)
%BUILD_VCYCLE_TEST_MGPU Build a separate MEX for raw V-cycle validation.
%
% On Linux with MATLAB, CUDA Toolkit, cuDSS and NCCL installed:
%   build_VCycle_Test_MGPU('/path/to/cudss', '/path/to/nccl');
%   build_VCycle_Test_MGPU('/path/to/cudss', '/path/to/nccl', '/usr/local/cuda');
%
% This DOES NOT modify the original CUDA source, kernels, or functions.
% VCycle_Test_MGPU.cu.in includes exact copies of the preserved functions.
% The generated .cu differs only by a THREE-PUNCTUATION syntax repair to the
% broken sendOneSideHalo call in applyVcycle; it does not alter computations.
%
% Compiler must support OpenMP (GCC host compiler on Linux).
% CUDA, NCCL, and cuDSS runtime libraries must be discoverable at runtime.

if nargin < 3 || isempty(cudaRoot), cudaRoot = '/usr/local/cuda'; end
if nargin < 2 || isempty(ncclRoot), error('Specify your NCCL installation root.'); end
if nargin < 1 || isempty(cudssRoot), error('Specify your cuDSS installation root.'); end
if ~isunix || ismac
    error('This harness build is intended for Linux (NCCL and OpenMP).');
end

root = fileparts(mfilename('fullpath'));
inFile = fullfile(root, 'VCycle_Test_MGPU.cu.in');
outFile = fullfile(root, 'Solving_VCycle_Test_MGPU.cu');
src = fileread(inFile);

% The original applyVcycle has a missing comma and an extra trailing comma
% in one coarse-halo call. The unmodified source is retained in .cu.in.
% No kernel, numerical operation, or communication strategy is changed.
before = sprintf(['                streams.rightStream\n', ...
    '                gpuData[neighbor].levels[fineIndex + 1].d_rhs,\n', ...
    '            );']);
after = sprintf(['                streams.rightStream,\n', ...
    '                gpuData[neighbor].levels[fineIndex + 1].d_rhs\n', ...
    '            );']);

% Normalize the source line endings only for reliable pattern matching.
src = strrep(src, sprintf('\r\n'), sprintf('\n'));
assert(count(src, before) == 1, ...
    'Expected exactly one malformed right-halo call; source may have changed.');
src = strrep(src, before, after);
% Second original coarse-halo call has one extra trailing comma.
beforeLeft = sprintf(['                streams.leftStream,\n', ...
    '                gpuData[neighbor].levels[fineIndex + 1].d_rhs,\n', ...
    '            );']);
afterLeft = sprintf(['                streams.leftStream,\n', ...
    '                gpuData[neighbor].levels[fineIndex + 1].d_rhs\n', ...
    '            );']);
assert(count(src, beforeLeft) == 1, 'Expected exactly one malformed left-halo call.');
src = strrep(src, beforeLeft, afterLeft);
fid = fopen(outFile, 'w');
if fid < 0, error('Cannot create generated CUDA file: %s',outFile); end
cleanObj = onCleanup(@()fclose(fid));
fwrite(fid, src, 'char');
clear cleanObj;

incCuda = fullfile(cudaRoot,'include');
incCudss = fullfile(cudssRoot,'include');
incNccl = fullfile(ncclRoot,'include');
libCuda = fullfile(cudaRoot,'lib64');
libCudss = localLibraryDir(cudssRoot);
libNccl = localLibraryDir(ncclRoot);

fprintf('Building standalone V-cycle MEX:\n  %s\n',outFile);
old = pwd;
restoreDir = onCleanup(@()cd(old));
cd(root);
mexcuda('-v', '-largeArrayDims', ...
    'NVCCFLAGS=$NVCCFLAGS -Xcompiler -fopenmp', ...
    'LDFLAGS=$LDFLAGS -fopenmp', ...
    ['-I' incCuda], ['-I' incCudss], ['-I' incNccl], ...
    ['-L' libCuda], ['-L' libCudss], ['-L' libNccl], ...
    '-lcudss', '-lnccl', ...
    '-output', 'Solving_VCycle_Test_MGPU', outFile);

fprintf('Compiled: %s\n', which('Solving_VCycle_Test_MGPU'));
end

function folder = localLibraryDir(root)
choices = {fullfile(root,'lib'), fullfile(root,'lib64'), ...
    fullfile(root,'lib','x86_64-linux-gnu')};
folder = '';
for i = 1:numel(choices)
    if isfolder(choices{i})
        folder = choices{i};
        break;
    end
end
if isempty(folder), error('No library directory found under %s',root); end
end
