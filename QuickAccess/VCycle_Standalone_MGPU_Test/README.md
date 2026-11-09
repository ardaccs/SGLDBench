# Standalone multi-GPU V-cycle test harness

This package calls the supplied `applyVcycle` directly and returns **its raw finest-grid correction** to MATLAB. It does not require the incomplete PCG solver or its initialization. No existing project file is modified.

## Files

- `VCycle_Test_MGPU.cu.in`: assembly of the preserved computational functions and original `applyVcycle` **verbatim** from the uploaded text, plus new standalone data structures, setup, adapters, and MEX entry point. This is the source of truth.
- `Solving_VCycle_Test_MGPU.cu`: generated compilable copy of the above with the **only unavoidable source punctuation corrections** described below.
- `build_VCycle_Test_MGPU.m`: regenerates the compiled `.cu` from `.cu.in`, applies exactly these punctuation corrections and calls `mexcuda`.
- `Build_VCycle_Test_H.m`: packs an **already initialized** original MATLAB CPU hierarchy and coarsest Cholesky factors into the required scalar input `H`.
- `Validate_VCycle_Raw_MGPU.m`: compares the raw GPU output directly against your *existing* `Solving_Vcycle(r)` on random, constant, interface impulse and sparse inputs.

## Protected functions

The existing computational bodies were copied directly without numerical or algorithmic edits:

`zeroSelectedDOFsKernel`, `gatherSelectedDOFsKernel`, `scatterSelectedDOFsKernel`, `dampedJacobiSmootherKernelFine`, `dampedJacobiSmootherKernelCoarse`, `addDampedJacobiKernel`, `addVectorsInPlaceKernel`, `interpolateResidualKernel`, `restrictResidualKernel`, `gatherGlobalCoarseResidualKernel`, `packHaloKernel`, `sendOneSideHalo`, `assembleGlobalCoarseResidual`, `solveCoarsest`, `broadcastCoarseSolution`, `applyVcycle`.

Other kernels from the full PCG source that the V-cycle never calls are not included. Extra code supplies data-layout structures, peer halo partitions, GPU buffers, cuDSS setup, NCCL communicators, cleanup, a missing coarse-scatter kernel, and wrappers for original signature mismatches. In particular, `singleGPUData::operator[]` supports the original `G[lastLevel]` expression, and overloads bridge `sendOneSideHalo(G, ...)` and `broadcastCoarseSolution(G, gpuData, lastLevel)` to their unchanged definitions.

### The one unavoidable source-level exception

The **original** `applyVcycle` contains two unparseable coarse-halo calls:

```cpp
streams.rightStream
    gpuData[neighbor].levels[fineIndex + 1].d_rhs,
```

and

```cpp
streams.leftStream,
    gpuData[neighbor].levels[fineIndex + 1].d_rhs,
```

These cannot be compiled literally. The build script only inserts a missing comma after `streams.rightStream` and removes the final trailing commas from the two calls (3 punctuation edits in total). The `.cu.in` keeps the exact original, including these mistakes; no change to its V-cycle maths, kernels, halo-copy function, coarse solver, interpolation, or restriction is applied. The compile-ready `.cu` necessarily has these 3 syntax edits; it is not strictly byte-identical to the original `applyVcycle`. All edits are visible in `build_VCycle_Test_MGPU.m`.

## Prerequisites

Linux (preferably your CUDA HPC machine), compatible MATLAB with `mexcuda`, NVCC, a supported GCC host compiler with OpenMP, NVIDIA cuDSS and NCCL headers / shared libraries, and *multiple visible NVIDIA GPUs* for MGPU tests. Runtime library search paths must include cuDSS / NCCL `.so` directories. **This has not been nvcc-compiled or GPU-executed in the authoring environment**, because CUDA/NVCC, cuDSS, NCCL and MATLAB are not installed here.

## Compile

In MATLAB, from the extracted folder:

```matlab
mex -setup C++
mexcuda -setup
build_VCycle_Test_MGPU('/path/to/cudss', '/path/to/nccl', '/usr/local/cuda');
which Solving_VCycle_Test_MGPU
```

The first two roots must contain both `include/` and `lib/` or `lib64/`; CUDA's root defaults to `/usr/local/cuda`.

## Validate

Run your original CPU FEA / hierarchy / Cholesky setup **first**, and make the existing `Solving_Vcycle` callable from MATLAB. If it is a local subfunction, pass a function handle from its owning scope or expose it in its own unmodified `.m` file.

```matlab
H = Build_VCycle_Test_H();
r = randn(double(H.numDOFs(1)),1);
r(double(H.fixedDOFIds)) = 0;
[zGPU,info] = Solving_VCycle_Test_MGPU(r,H,2,1);  % splitAxis=1 is x
zCPU = Solving_Vcycle(r);
fprintf('relL2 = %.3e, maxAbs = %.3e\n', ...
    norm(zGPU-zCPU)/max(norm(zCPU),eps), max(abs(zGPU-zCPU)));
results = Validate_VCycle_Raw_MGPU(H,2,1e-9);
```

You can also pass a ready-made packed `H` containing `numLevels,resX,resY,resZ,numNodes,spanWidth,nodGridId,nodMapForward,diagK,coarseFreeDOFIds,coarseKFree,fixedDOFIds,jacobiOmega` instead of calling `Build_VCycle_Test_H`.

Call signature:

```matlab
[z, info] = Solving_VCycle_Test_MGPU(r, H, numGPUs, splitAxis);
```

`z` is assembled in full-domain/global DOF ordering; shared partition normals are reconciled into one global vector, and `info.maxSharedNodeDifference` exposes disagreement at duplicate interface nodes. `splitAxis` is `1=x`, `2=y`, `3=z`; the harness defaults to `1`.

### Important test scope / limits

- The harness **generates its own contiguous slab partitions and halo mappings** from the global active-node geometry, aligned to the coarsest grid. It does **not** use your application's existing MATLAB `.partitions` data. This tests the protected V-cycle with newly built test geometry; it does not establish correctness of an independently implemented production partition builder.
- It assumes exact nested coarsening (fine resolution = `spanWidth` times coarse resolution on every axis), active-node/global-node consistency, and at least `numGPUs` coarsest elements along the chosen split axis. If too few active nodes occur in a slab, the setup reports an error.
- The original CUDA helpers use `NCCL_CHECK(... exit(EXIT_FAILURE))` and can throw from an OpenMP region. Those behaviors are **unchanged** and can terminate MATLAB on a fault. Running a test in a separate MATLAB process is safer.
- The test correctly compares **raw V-cycle scale**, unlike comparison through a first PCG iterate. It checks all full-domain DOFs plus shared-node discrepancies.
- Errors from `mexcuda` or numerical mismatches should be reported as-is; this harness does not silently correct any V-cycle calculation.
