/*
 * sgld_pcg_vcycle_gpucoarse_cuda_mex.cu
 *
 * PCG MEX with:
 *   - CUDA matrix-free KbyU on level 1
 *   - CUDA 3-level V-cycle preconditioner using SGLDBench transfer structures
 *   - fully GPU-resident level-3 sparse Cholesky solve using cuSPARSE
 *   - MATLAB cholFac_ CSC storage is interpreted as CSR storage of L^T
 *
 * This is the "PCG owns the loop" version:
 *
 *   y = 0 or U0
 *   r = b - A*y
 *   z = CUDA_Vcycle(r)
 *   p = z
 *   for its = 1:maxIT
 *       Ap = A*p
 *       alpha = (z'r) / (p'Ap)
 *       y = y + alpha*p
 *       r = r - alpha*Ap
 *       z = CUDA_Vcycle(r)
 *       beta = (z_new'r) / (z_old'r_old)
 *       p = z + beta*p
 *   end
 *
 * MATLAB call:
 *
 *   [U, its, relres] = sgld_pcg_vcycle_gpucoarse_cuda_mex( ...
 *       F, ...
 *       U0, ...
 *       int32(mh1.nodeToElements), ...
 *       int32(mh1.eNodMat), ...
 *       mh1.eleModulus(:), ...
 *       mh1.Ks, ...
 *       mh2, ...
 *       mh3, ...
 *       mh1.diagK, ...
 *       fixedMask1, ...
 *       mh2.diagK, ...
 *       fixedMask2, ...
 *       fixedMask3, ...
 *       cholFac_, ...
 *       coarsePerm, ...
 *       weightFactorJacobi_, ...
 *       tol_, ...
 *       maxIT_, ...
 *       'printP_ON');
 *
 * U0 may be [].
 *
 * Compile:
 *
 *   clear mex
 *   mexcuda -R2018a src/MEXfuncs/sgld_pcg_vcycle_gpucoarse_cuda_mex.cu -lcusparse
 *
 * or your local CUDA config:
 *
 *   clear mex
 *   mex -R2018a -v -f local_cuda.xml src/MEXfuncs/sgld_pcg_vcycle_gpucoarse_cuda_mex.cu -lcusparse
 */

#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

#include <cstdint>
#include <climits>
#include <cmath>
#include <cstring>


#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:cuda",                   \
                "CUDA error at %s:%d: %s",                                   \
                __FILE__, __LINE__, cudaGetErrorString(err__));               \
        }                                                                    \
    } while (0)


#define CUSPARSE_CHECK(call)                                                 \
    do {                                                                     \
        cusparseStatus_t status__ = (call);                                   \
        if (status__ != CUSPARSE_STATUS_SUCCESS) {                            \
            mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:cusparse",          \
                "cuSPARSE error at %s:%d: status %d",                        \
                __FILE__, __LINE__, static_cast<int>(status__));              \
        }                                                                    \
    } while (0)


#define MEX_PRINT(...)                                                       \
    do {                                                                     \
        mexPrintf(__VA_ARGS__);                                               \
        mexPrintf("\n");                                                     \
        mexEvalString("drawnow;");                                           \
    } while (0)


#if __CUDA_ARCH__ < 600
__device__ double atomicAddD(double* address, double val)
{
    unsigned long long int* addressAsULL =
        reinterpret_cast<unsigned long long int*>(address);

    unsigned long long int old = *addressAsULL;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(
            addressAsULL,
            assumed,
            __double_as_longlong(
                val + __longlong_as_double(assumed)));
    }
    while (assumed != old);

    return __longlong_as_double(old);
}
#else
__device__ double atomicAddD(double* address, double val)
{
    return atomicAdd(address, val);
}
#endif


// -----------------------------------------------------------------------------
// Basic PCG / vector kernels
// -----------------------------------------------------------------------------

__global__ void zero_kernel(double* __restrict__ x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0;
}


__global__ void zero_fixed_kernel(
    double* __restrict__ x,
    const unsigned char* __restrict__ fixed,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && fixed[i]) x[i] = 0.0;
}


__global__ void residual_kernel(
    const double* __restrict__ b,
    const double* __restrict__ Ax,
    double* __restrict__ r,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - Ax[i];
}


__global__ void update_y_r_kernel(
    double* __restrict__ y,
    double* __restrict__ r,
    const double* __restrict__ p,
    const double* __restrict__ Ap,
    double alpha,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        y[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
    }
}


__global__ void update_p_kernel(
    double* __restrict__ p,
    const double* __restrict__ z,
    double beta,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        p[i] = z[i] + beta * p[i];
}



// -----------------------------------------------------------------------------
// Coarsest-level gather / permutation / scatter kernels
// -----------------------------------------------------------------------------

// MATLAB uses rhsPerm = P' * rhsFree.  coarsePerm is defined by
// P*v = v(coarsePerm), so P'*rhs is rhs(invPerm).
__global__ void gather_permuted_coarse_rhs_kernel(
    const double* __restrict__ r3,
    double* __restrict__ rhsPerm,
    const int32_t* __restrict__ freeIndices,
    const int32_t* __restrict__ inversePerm,
    int nFree)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < nFree)
    {
        int freePosition = inversePerm[j];
        int globalDOF = freeIndices[freePosition];
        rhsPerm[j] = r3[globalDOF];
    }
}


// MATLAB uses xFree = P*xPerm, hence xFree(i) = xPerm(coarsePerm(i)).
__global__ void scatter_permuted_coarse_solution_kernel(
    const double* __restrict__ xPerm,
    double* __restrict__ z3,
    const int32_t* __restrict__ freeIndices,
    const int32_t* __restrict__ coarsePerm,
    int nFree)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nFree)
    {
        int globalDOF = freeIndices[i];
        int source = coarsePerm[i];
        z3[globalDOF] = xPerm[source];
    }
}


__global__ void jacobi_assign_kernel(
    const double* __restrict__ r,
    const double* __restrict__ diagK,
    const unsigned char* __restrict__ fixed,
    double* __restrict__ z,
    double omega,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        if (fixed[i])
        {
            z[i] = 0.0;
        }
        else
        {
            double d = diagK[i];
            z[i] = (d != 0.0) ? omega * r[i] / d : 0.0;
        }
    }
}


__global__ void jacobi_add_kernel(
    const double* __restrict__ r,
    const double* __restrict__ diagK,
    const unsigned char* __restrict__ fixed,
    double* __restrict__ z,
    double omega,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && !fixed[i])
    {
        double d = diagK[i];

        if (d != 0.0)
            z[i] += omega * r[i] / d;
    }
}


// Unmasked level-2 Jacobi operations used to reproduce Solving_Vcycle.
// Coarse fixed entries are kept through restriction and level-2 smoothing;
// the exact coarsest callback applies mh3.freeDOFs itself.
__global__ void jacobi_assign_all_kernel(
    const double* __restrict__ r,
    const double* __restrict__ diagK,
    double* __restrict__ z,
    double omega,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        double d = diagK[i];
        z[i] = (d != 0.0) ? omega * r[i] / d : 0.0;
    }
}


__global__ void jacobi_add_all_kernel(
    const double* __restrict__ r,
    const double* __restrict__ diagK,
    double* __restrict__ z,
    double omega,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        double d = diagK[i];

        if (d != 0.0)
            z[i] += omega * r[i] / d;
    }
}


// -----------------------------------------------------------------------------
// Level-1 KbyU kernel
// -----------------------------------------------------------------------------

__global__ void kbyu_kernel(
    const double* __restrict__ U,
    double* __restrict__ Y,

    const int32_t* __restrict__ nodeToElements,
    const int32_t* __restrict__ eNodMat,

    const double* __restrict__ E,
    const double* __restrict__ Ke,

    int numNodes,
    int numElements)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= numNodes)
        return;

    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    int elemNodes[8];
    double Ue[24];

    #pragma unroll
    for (int a = 0; a < 8; ++a)
    {
        int elem = nodeToElements[node + a * numNodes] - 1;

        if (elem < 0 || elem >= numElements)
            continue;

        int localNode = -1;

        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int n = eNodMat[elem + j * numElements] - 1;
            elemNodes[j] = n;

            if (n == node)
                localNode = j;
        }

        if (localNode < 0)
            continue;

        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int n = elemNodes[j];

            if (n < 0 || n >= numNodes)
            {
                Ue[3*j + 0] = 0.0;
                Ue[3*j + 1] = 0.0;
                Ue[3*j + 2] = 0.0;
                continue;
            }

            int base = 3 * n;

            Ue[3*j + 0] = U[base + 0];
            Ue[3*j + 1] = U[base + 1];
            Ue[3*j + 2] = U[base + 2];
        }

        int row0 = 3 * localNode;

        double y0 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;

        #pragma unroll
        for (int c = 0; c < 24; ++c)
        {
            double u = Ue[c];

            y0 += Ke[(row0 + 0) + c * 24] * u;
            y1 += Ke[(row0 + 1) + c * 24] * u;
            y2 += Ke[(row0 + 2) + c * 24] * u;
        }

        double Ee = E[elem];

        sum0 += Ee * y0;
        sum1 += Ee * y1;
        sum2 += Ee * y2;
    }

    int out = 3 * node;

    Y[out + 0] = sum0;
    Y[out + 1] = sum1;
    Y[out + 2] = sum2;
}


static void launch_kbyu(
    const double* d_U,
    double* d_Y,

    const int32_t* d_nodeToElements,
    const int32_t* d_eNodMat,

    const double* d_E,
    const double* d_Ke,

    const unsigned char* d_fixed,

    int numDOFs,
    int numNodes,
    int numElements,
    int threads)
{
    int nodeBlocks = (numNodes + threads - 1) / threads;
    int dofBlocks  = (numDOFs  + threads - 1) / threads;

    kbyu_kernel<<<nodeBlocks, threads>>>(
        d_U,
        d_Y,
        d_nodeToElements,
        d_eNodMat,
        d_E,
        d_Ke,
        numNodes,
        numElements);

    CUDA_CHECK(cudaGetLastError());

    // Match Solving_KbyU_MatrixFree:
    //
    //   productMV(meshHierarchy_(1).fixedDOFs,1) = 0;
    zero_fixed_kernel<<<dofBlocks, threads>>>(
        d_Y,
        d_fixed,
        numDOFs);

    CUDA_CHECK(cudaGetLastError());
}


static double device_dot(
    const double* d_a,
    const double* d_b,
    int n)
{
    thrust::device_ptr<const double> a =
        thrust::device_pointer_cast(d_a);

    thrust::device_ptr<const double> b =
        thrust::device_pointer_cast(d_b);

    return thrust::inner_product(a, a + n, b, 0.0);
}


// -----------------------------------------------------------------------------
// V-cycle transfer structures and kernels
// -----------------------------------------------------------------------------

struct TransferHost
{
    const int32_t* T = nullptr;
    const double* coeff = nullptr;
    const int32_t* solidMap = nullptr;
    const double* op = nullptr;
    const int32_t* eNod = nullptr;

    int nPatch = 0;
    int nElem = 0;
    int nInter = 0;
    int nFine = 0;
    int nCoarse = 0;
    int opPatchBy8 = 0;
};


struct TransferDevice
{
    int32_t* T = nullptr;
    double* coeff = nullptr;
    int32_t* solidMap = nullptr;
    double* op = nullptr;
    int32_t* eNod = nullptr;

    int nPatch = 0;
    int nElem = 0;
    int nInter = 0;
    int nFine = 0;
    int nCoarse = 0;
    int opPatchBy8 = 0;
};


static const mxArray* get_field(
    const mxArray* s,
    const char* name)
{
    if (!mxIsStruct(s) || mxGetNumberOfElements(s) != 1)
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:struct",
            "Level inputs mh2 and mh3 must be scalar meshHierarchy_ structs.");
    }

    const mxArray* a = mxGetField(s, 0, name);

    if (!a)
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:field",
            "Missing meshHierarchy_ field '%s'.",
            name);
    }

    return a;
}


static TransferHost parse_level(
    const mxArray* levelStruct,
    int nFineNodes,
    int nCoarseNodes)
{
    TransferHost h;

    const mxArray* T_mx =
        get_field(levelStruct, "transferMat");

    const mxArray* C_mx =
        get_field(levelStruct, "transferMatCoeffi");

    const mxArray* S_mx =
        get_field(levelStruct, "solidNodeMapCoarser2Finer");

    const mxArray* O_mx =
        get_field(levelStruct, "multiGridOperatorRIdense");

    const mxArray* E_mx =
        get_field(levelStruct, "eNodMat");

    const mxArray* N_mx =
        get_field(levelStruct, "intermediateNumNodes");

    if (!mxIsInt32(T_mx) || mxIsSparse(T_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:T", "transferMat must be full int32.");

    if (!mxIsDouble(C_mx) || mxIsComplex(C_mx) || mxIsSparse(C_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:C", "transferMatCoeffi must be full real double.");

    if (!mxIsInt32(S_mx) || mxIsSparse(S_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:S", "solidNodeMapCoarser2Finer must be full int32.");

    if (!mxIsDouble(O_mx) || mxIsComplex(O_mx) || mxIsSparse(O_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:O", "multiGridOperatorRIdense must be full real double.");

    if (!mxIsInt32(E_mx) || mxIsSparse(E_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:E", "coarse eNodMat must be full int32.");

    h.nPatch = static_cast<int>(mxGetM(T_mx));
    h.nElem = static_cast<int>(mxGetN(T_mx));
    h.nInter = static_cast<int>(mxGetScalar(N_mx));
    h.nFine = nFineNodes;
    h.nCoarse = nCoarseNodes;

    if (mxGetNumberOfElements(C_mx) != static_cast<size_t>(h.nInter))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:Csize", "Bad transferMatCoeffi length.");

    if (mxGetNumberOfElements(S_mx) != static_cast<size_t>(nFineNodes))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:Ssize", "solidNodeMapCoarser2Finer must have nFine entries.");

    if (static_cast<int>(mxGetM(E_mx)) != h.nElem || mxGetN(E_mx) < 8)
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:Esize", "eNodMat must be [nElem x 8].");

    if (static_cast<int>(mxGetM(O_mx)) == h.nPatch && mxGetN(O_mx) == 8)
        h.opPatchBy8 = 1;
    else if (mxGetM(O_mx) == 8 && static_cast<int>(mxGetN(O_mx)) == h.nPatch)
        h.opPatchBy8 = 0;
    else
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:Osize", "multiGridOperatorRIdense must be [nPatch x 8] or [8 x nPatch].");

    h.T = static_cast<const int32_t*>(mxGetData(T_mx));
    h.coeff = mxGetDoubles(C_mx);
    h.solidMap = static_cast<const int32_t*>(mxGetData(S_mx));
    h.op = mxGetDoubles(O_mx);
    h.eNod = static_cast<const int32_t*>(mxGetData(E_mx));

    return h;
}


static unsigned char* make_fixed_u8(
    const mxArray* fixed_mx,
    int n)
{
    if (!mxIsLogical(fixed_mx) ||
        mxIsSparse(fixed_mx) ||
        mxGetNumberOfElements(fixed_mx) != static_cast<size_t>(n))
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:fixed",
            "fixedDOFs/fixedMask inputs must be full logical vectors with matching length.");
    }

    const mxLogical* p = mxGetLogicals(fixed_mx);

    unsigned char* out =
        static_cast<unsigned char*>(
            mxMalloc(static_cast<mwSize>(n)));

    for (int i = 0; i < n; ++i)
        out[i] = p[i] ? 1 : 0;

    return out;
}


static void upload_transfer(
    const TransferHost& h,
    TransferDevice& d)
{
    d.nPatch = h.nPatch;
    d.nElem = h.nElem;
    d.nInter = h.nInter;
    d.nFine = h.nFine;
    d.nCoarse = h.nCoarse;
    d.opPatchBy8 = h.opPatchBy8;

    size_t bT =
        sizeof(int32_t) *
        static_cast<size_t>(h.nPatch) *
        static_cast<size_t>(h.nElem);

    size_t bC =
        sizeof(double) *
        static_cast<size_t>(h.nInter);

    size_t bS =
        sizeof(int32_t) *
        static_cast<size_t>(h.nFine);

    size_t bO =
        sizeof(double) *
        static_cast<size_t>(h.nPatch) * 8;

    size_t bE =
        sizeof(int32_t) *
        static_cast<size_t>(h.nElem) * 8;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.T), bT));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.coeff), bC));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.solidMap), bS));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.op), bO));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.eNod), bE));

    CUDA_CHECK(cudaMemcpy(d.T, h.T, bT, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.coeff, h.coeff, bC, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.solidMap, h.solidMap, bS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.op, h.op, bO, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.eNod, h.eNod, bE, cudaMemcpyHostToDevice));
}


static void free_transfer(TransferDevice& d)
{
    if (d.T) cudaFree(d.T);
    if (d.coeff) cudaFree(d.coeff);
    if (d.solidMap) cudaFree(d.solidMap);
    if (d.op) cudaFree(d.op);
    if (d.eNod) cudaFree(d.eNod);

    std::memset(&d, 0, sizeof(d));
}


__device__ __forceinline__
double transfer_weight(
    const double* __restrict__ op,
    int nPatch,
    int patch,
    int localCoarseNode,
    int patchBy8)
{
    // MATLAB column-major.
    if (patchBy8)
        return op[patch + localCoarseNode * nPatch];

    return op[localCoarseNode + patch * 8];
}


__global__ void scatter_fine_to_intermediate_kernel(
    const double* __restrict__ fine,
    double* __restrict__ intermediate,
    const int32_t* __restrict__ solidMap,
    int nFineNodes)
{
    int fineNode = blockIdx.x * blockDim.x + threadIdx.x;

    if (fineNode >= nFineNodes)
        return;

    int intermediateNode = solidMap[fineNode] - 1;

    if (intermediateNode < 0)
        return;

    int f = 3 * fineNode;
    int q = 3 * intermediateNode;

    intermediate[q + 0] = fine[f + 0];
    intermediate[q + 1] = fine[f + 1];
    intermediate[q + 2] = fine[f + 2];
}


__global__ void restrict_transpose_kernel(
    const double* __restrict__ intermediate,
    double* __restrict__ coarse,

    const int32_t* __restrict__ T,
    const double* __restrict__ coeff,
    const double* __restrict__ op,
    const int32_t* __restrict__ eNod,

    int nPatch,
    int nElem,
    int nInter,
    int nCoarse,
    int patchBy8)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nElem * 8 * 3;

    if (id >= total)
        return;

    int comp = id % 3;
    int q = id / 3;
    int localCoarseNode = q % 8;
    int elem = q / 8;

    int coarseNode =
        eNod[elem + localCoarseNode * nElem] - 1;

    if (coarseNode < 0 || coarseNode >= nCoarse)
        return;

    double sum = 0.0;

    for (int p = 0; p < nPatch; ++p)
    {
        int intermediateNode = T[p + elem * nPatch] - 1;

        if (intermediateNode < 0 || intermediateNode >= nInter)
            continue;

        double c = coeff[intermediateNode];

        if (c == 0.0)
            continue;

        double w =
            transfer_weight(
                op,
                nPatch,
                p,
                localCoarseNode,
                patchBy8);

        sum += w * intermediate[3 * intermediateNode + comp] / c;
    }

    atomicAddD(&coarse[3 * coarseNode + comp], sum);
}


__global__ void interpolate_element_kernel(
    const double* __restrict__ coarse,
    double* __restrict__ intermediate,

    const int32_t* __restrict__ T,
    const double* __restrict__ op,
    const int32_t* __restrict__ eNod,

    int nPatch,
    int nElem,
    int nInter,
    int nCoarse,
    int patchBy8)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nElem * nPatch * 3;

    if (id >= total)
        return;

    int comp = id % 3;
    int q = id / 3;
    int patch = q % nPatch;
    int elem = q / nPatch;

    int intermediateNode = T[patch + elem * nPatch] - 1;

    if (intermediateNode < 0 || intermediateNode >= nInter)
        return;

    double value = 0.0;

    #pragma unroll
    for (int a = 0; a < 8; ++a)
    {
        int coarseNode = eNod[elem + a * nElem] - 1;

        if (coarseNode < 0 || coarseNode >= nCoarse)
            continue;

        double w =
            transfer_weight(
                op,
                nPatch,
                patch,
                a,
                patchBy8);

        value += w * coarse[3 * coarseNode + comp];
    }

    atomicAddD(&intermediate[3 * intermediateNode + comp], value);
}


__global__ void normalize_intermediate_kernel(
    double* __restrict__ intermediate,
    const double* __restrict__ coeff,
    int nInter)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 3 * nInter;

    if (id >= total)
        return;

    int j = id / 3;
    double c = coeff[j];

    intermediate[id] = (c != 0.0) ? intermediate[id] / c : 0.0;
}


__global__ void gather_intermediate_add_kernel(
    const double* __restrict__ intermediate,
    double* __restrict__ fine,
    const int32_t* __restrict__ solidMap,
    int nFineNodes)
{
    int fineNode = blockIdx.x * blockDim.x + threadIdx.x;

    if (fineNode >= nFineNodes)
        return;

    int intermediateNode = solidMap[fineNode] - 1;

    if (intermediateNode < 0)
        return;

    int f = 3 * fineNode;
    int q = 3 * intermediateNode;

    fine[f + 0] += intermediate[q + 0];
    fine[f + 1] += intermediate[q + 1];
    fine[f + 2] += intermediate[q + 2];
}


static void restrict_level(
    const double* d_fine,
    double* d_coarse,
    double* d_intermediate,
    const TransferDevice& t,
    int threads)
{
    int intermediateDOFs = 3 * t.nInter;
    int coarseDOFs = 3 * t.nCoarse;

    int intermediateBlocks =
        (intermediateDOFs + threads - 1) / threads;

    int fineNodeBlocks =
        (t.nFine + threads - 1) / threads;

    int coarseBlocks =
        (coarseDOFs + threads - 1) / threads;

    int totalRestrictThreads =
        t.nElem * 8 * 3;

    int restrictBlocks =
        (totalRestrictThreads + threads - 1) / threads;

    zero_kernel<<<intermediateBlocks, threads>>>(
        d_intermediate,
        intermediateDOFs);

    CUDA_CHECK(cudaGetLastError());

    scatter_fine_to_intermediate_kernel<<<fineNodeBlocks, threads>>>(
        d_fine,
        d_intermediate,
        t.solidMap,
        t.nFine);

    CUDA_CHECK(cudaGetLastError());

    zero_kernel<<<coarseBlocks, threads>>>(
        d_coarse,
        coarseDOFs);

    CUDA_CHECK(cudaGetLastError());

    restrict_transpose_kernel<<<restrictBlocks, threads>>>(
        d_intermediate,
        d_coarse,
        t.T,
        t.coeff,
        t.op,
        t.eNod,
        t.nPatch,
        t.nElem,
        t.nInter,
        t.nCoarse,
        t.opPatchBy8);

    CUDA_CHECK(cudaGetLastError());
}


static void interpolate_add_level(
    const double* d_coarse,
    double* d_fine,
    double* d_intermediate,
    const TransferDevice& t,
    int threads)
{
    int intermediateDOFs =
        3 * t.nInter;

    int intermediateBlocks =
        (intermediateDOFs + threads - 1) / threads;

    int totalInterpThreads =
        t.nElem * t.nPatch * 3;

    int interpBlocks =
        (totalInterpThreads + threads - 1) / threads;

    int fineNodeBlocks =
        (t.nFine + threads - 1) / threads;

    zero_kernel<<<intermediateBlocks, threads>>>(
        d_intermediate,
        intermediateDOFs);

    CUDA_CHECK(cudaGetLastError());

    interpolate_element_kernel<<<interpBlocks, threads>>>(
        d_coarse,
        d_intermediate,
        t.T,
        t.op,
        t.eNod,
        t.nPatch,
        t.nElem,
        t.nInter,
        t.nCoarse,
        t.opPatchBy8);

    CUDA_CHECK(cudaGetLastError());

    normalize_intermediate_kernel<<<intermediateBlocks, threads>>>(
        d_intermediate,
        t.coeff,
        t.nInter);

    CUDA_CHECK(cudaGetLastError());

    gather_intermediate_add_kernel<<<fineNodeBlocks, threads>>>(
        d_intermediate,
        d_fine,
        t.solidMap,
        t.nFine);

    CUDA_CHECK(cudaGetLastError());
}


struct CoarseGpuSolver
{
    int n = 0;
    int nnz = 0;

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matLt = nullptr;
    cusparseDnVecDescr_t vecRhs = nullptr;
    cusparseDnVecDescr_t vecQ = nullptr;
    cusparseDnVecDescr_t vecX = nullptr;
    cusparseSpSVDescr_t solveLDescr = nullptr;
    cusparseSpSVDescr_t solveLtDescr = nullptr;

    int32_t* d_rowPtr = nullptr;
    int32_t* d_colInd = nullptr;
    double* d_values = nullptr;

    int32_t* d_freeIndices = nullptr;
    int32_t* d_perm = nullptr;
    int32_t* d_inversePerm = nullptr;

    double* d_rhsPerm = nullptr;
    double* d_q = nullptr;
    double* d_xPerm = nullptr;

    void* d_bufferL = nullptr;
    void* d_bufferLt = nullptr;
    size_t bufferSizeL = 0;
    size_t bufferSizeLt = 0;
};


static void destroy_coarse_gpu_solver(CoarseGpuSolver& solver)
{
    if (solver.solveLDescr)
        cusparseSpSV_destroyDescr(solver.solveLDescr);

    if (solver.solveLtDescr)
        cusparseSpSV_destroyDescr(solver.solveLtDescr);

    if (solver.vecRhs)
        cusparseDestroyDnVec(solver.vecRhs);

    if (solver.vecQ)
        cusparseDestroyDnVec(solver.vecQ);

    if (solver.vecX)
        cusparseDestroyDnVec(solver.vecX);

    if (solver.matLt)
        cusparseDestroySpMat(solver.matLt);

    if (solver.handle)
        cusparseDestroy(solver.handle);

    if (solver.d_bufferL) cudaFree(solver.d_bufferL);
    if (solver.d_bufferLt) cudaFree(solver.d_bufferLt);

    if (solver.d_rowPtr) cudaFree(solver.d_rowPtr);
    if (solver.d_colInd) cudaFree(solver.d_colInd);
    if (solver.d_values) cudaFree(solver.d_values);

    if (solver.d_freeIndices) cudaFree(solver.d_freeIndices);
    if (solver.d_perm) cudaFree(solver.d_perm);
    if (solver.d_inversePerm) cudaFree(solver.d_inversePerm);

    if (solver.d_rhsPerm) cudaFree(solver.d_rhsPerm);
    if (solver.d_q) cudaFree(solver.d_q);
    if (solver.d_xPerm) cudaFree(solver.d_xPerm);

    std::memset(&solver, 0, sizeof(solver));
}


static void initialize_coarse_gpu_solver(
    const mxArray* cholFac_mx,
    const mxArray* coarsePerm_mx,
    const mxArray* fixed3_mx,
    CoarseGpuSolver& solver)
{
    if (!mxIsSparse(cholFac_mx) ||
        !mxIsDouble(cholFac_mx) ||
        mxIsComplex(cholFac_mx) ||
        mxGetM(cholFac_mx) != mxGetN(cholFac_mx))
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:cholFac",
            "cholFac must be a square real sparse double matrix.");
    }

    if (!mxIsInt32(coarsePerm_mx) || mxIsSparse(coarsePerm_mx))
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:perm",
            "coarsePerm must be a full int32 vector.");
    }

    if (!mxIsLogical(fixed3_mx) || mxIsSparse(fixed3_mx))
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:fixed3",
            "fixedMask3 must be a full logical vector.");
    }

    const size_t n_sz = mxGetM(cholFac_mx);

    if (n_sz > static_cast<size_t>(INT_MAX))
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:coarseSize",
            "The coarse factor dimension exceeds int32 range.");
    }

    solver.n = static_cast<int>(n_sz);

    if (mxGetNumberOfElements(coarsePerm_mx) != n_sz)
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:permSize",
            "coarsePerm length must equal size(cholFac,1).");
    }

    const mwIndex* jc = mxGetJc(cholFac_mx);
    const mwIndex* ir = mxGetIr(cholFac_mx);
    const double* values = mxGetDoubles(cholFac_mx);

    const mwIndex nnz_mw = jc[n_sz];

    if (nnz_mw > static_cast<mwIndex>(INT_MAX))
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:nnz",
            "cholFac nnz exceeds int32 range.");
    }

    solver.nnz = static_cast<int>(nnz_mw);

    const mxLogical* fixed3 = mxGetLogicals(fixed3_mx);
    const size_t nDOFs3 = mxGetNumberOfElements(fixed3_mx);

    int freeCount = 0;

    for (size_t i = 0; i < nDOFs3; ++i)
        if (!fixed3[i]) ++freeCount;

    if (freeCount != solver.n)
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:freeCount",
            "Number of non-fixed level-3 DOFs (%d) does not match cholFac dimension (%d).",
            freeCount,
            solver.n);
    }

    int32_t* h_rowPtr = static_cast<int32_t*>(
        mxMalloc(sizeof(int32_t) * (static_cast<size_t>(solver.n) + 1)));

    int32_t* h_colInd = static_cast<int32_t*>(
        mxMalloc(sizeof(int32_t) * static_cast<size_t>(solver.nnz)));

    int32_t* h_freeIndices = static_cast<int32_t*>(
        mxMalloc(sizeof(int32_t) * static_cast<size_t>(solver.n)));

    int32_t* h_perm = static_cast<int32_t*>(
        mxMalloc(sizeof(int32_t) * static_cast<size_t>(solver.n)));

    int32_t* h_inversePerm = static_cast<int32_t*>(
        mxMalloc(sizeof(int32_t) * static_cast<size_t>(solver.n)));

    unsigned char* seen = static_cast<unsigned char*>(
        mxCalloc(static_cast<mwSize>(solver.n), sizeof(unsigned char)));

    // MATLAB sparse CSC(L) has exactly the same arrays as CSR(L^T).
    // MATLAB ir/jc are already zero-based internally.
    for (int i = 0; i <= solver.n; ++i)
        h_rowPtr[i] = static_cast<int32_t>(jc[i]);

    for (int k = 0; k < solver.nnz; ++k)
        h_colInd[k] = static_cast<int32_t>(ir[k]);

    int freePosition = 0;

    for (size_t globalDOF = 0; globalDOF < nDOFs3; ++globalDOF)
    {
        if (!fixed3[globalDOF])
            h_freeIndices[freePosition++] = static_cast<int32_t>(globalDOF);
    }

    const int32_t* permInput = static_cast<const int32_t*>(
        mxGetData(coarsePerm_mx));

    for (int i = 0; i < solver.n; ++i)
    {
        int p = static_cast<int>(permInput[i]) - 1;

        if (p < 0 || p >= solver.n || seen[p])
        {
            mxFree(h_rowPtr);
            mxFree(h_colInd);
            mxFree(h_freeIndices);
            mxFree(h_perm);
            mxFree(h_inversePerm);
            mxFree(seen);

            mexErrMsgIdAndTxt(
                "sgld_pcg_vcycle_gpucoarse:permInvalid",
                "coarsePerm must contain every integer from 1 to n exactly once.");
        }

        seen[p] = 1;
        h_perm[i] = static_cast<int32_t>(p);
        h_inversePerm[p] = static_cast<int32_t>(i);
    }

    mxFree(seen);

    const size_t rowPtrBytes =
        sizeof(int32_t) * (static_cast<size_t>(solver.n) + 1);

    const size_t colIndBytes =
        sizeof(int32_t) * static_cast<size_t>(solver.nnz);

    const size_t valueBytes =
        sizeof(double) * static_cast<size_t>(solver.nnz);

    const size_t indexBytes =
        sizeof(int32_t) * static_cast<size_t>(solver.n);

    const size_t vectorBytes =
        sizeof(double) * static_cast<size_t>(solver.n);

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&solver.d_rowPtr),
        rowPtrBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&solver.d_colInd),
        colIndBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&solver.d_values),
        valueBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&solver.d_freeIndices),
        indexBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&solver.d_perm),
        indexBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&solver.d_inversePerm),
        indexBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&solver.d_rhsPerm),
        vectorBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&solver.d_q),
        vectorBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&solver.d_xPerm),
        vectorBytes));

    CUDA_CHECK(cudaMemcpy(
        solver.d_rowPtr,
        h_rowPtr,
        rowPtrBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        solver.d_colInd,
        h_colInd,
        colIndBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        solver.d_values,
        values,
        valueBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        solver.d_freeIndices,
        h_freeIndices,
        indexBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        solver.d_perm,
        h_perm,
        indexBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        solver.d_inversePerm,
        h_inversePerm,
        indexBytes,
        cudaMemcpyHostToDevice));

    mxFree(h_rowPtr);
    mxFree(h_colInd);
    mxFree(h_freeIndices);
    mxFree(h_perm);
    mxFree(h_inversePerm);

    CUSPARSE_CHECK(cusparseCreate(&solver.handle));

    // Reinterpret CSC(L) as CSR(L^T).  The resulting matrix descriptor is
    // upper triangular.  op(A)=A^T performs the forward solve with L.
    CUSPARSE_CHECK(cusparseCreateCsr(
        &solver.matLt,
        static_cast<int64_t>(solver.n),
        static_cast<int64_t>(solver.n),
        static_cast<int64_t>(solver.nnz),
        solver.d_rowPtr,
        solver.d_colInd,
        solver.d_values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;

    CUSPARSE_CHECK(cusparseSpMatSetAttribute(
        solver.matLt,
        CUSPARSE_SPMAT_FILL_MODE,
        &fillMode,
        sizeof(fillMode)));

    CUSPARSE_CHECK(cusparseSpMatSetAttribute(
        solver.matLt,
        CUSPARSE_SPMAT_DIAG_TYPE,
        &diagType,
        sizeof(diagType)));

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &solver.vecRhs,
        static_cast<int64_t>(solver.n),
        solver.d_rhsPerm,
        CUDA_R_64F));

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &solver.vecQ,
        static_cast<int64_t>(solver.n),
        solver.d_q,
        CUDA_R_64F));

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &solver.vecX,
        static_cast<int64_t>(solver.n),
        solver.d_xPerm,
        CUDA_R_64F));

    CUSPARSE_CHECK(cusparseSpSV_createDescr(
        &solver.solveLDescr));

    CUSPARSE_CHECK(cusparseSpSV_createDescr(
        &solver.solveLtDescr));

    const double one = 1.0;

    // Forward solve: L*q = rhsPerm, where A = L^T, hence op(A)=A^T=L.
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        solver.handle,
        CUSPARSE_OPERATION_TRANSPOSE,
        &one,
        solver.matLt,
        solver.vecRhs,
        solver.vecQ,
        CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT,
        solver.solveLDescr,
        &solver.bufferSizeL));

    // Backward solve: L^T*xPerm = q, directly using A=L^T.
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        solver.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one,
        solver.matLt,
        solver.vecQ,
        solver.vecX,
        CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT,
        solver.solveLtDescr,
        &solver.bufferSizeLt));

    CUDA_CHECK(cudaMalloc(
        &solver.d_bufferL,
        solver.bufferSizeL));

    CUDA_CHECK(cudaMalloc(
        &solver.d_bufferLt,
        solver.bufferSizeLt));

    CUSPARSE_CHECK(cusparseSpSV_analysis(
        solver.handle,
        CUSPARSE_OPERATION_TRANSPOSE,
        &one,
        solver.matLt,
        solver.vecRhs,
        solver.vecQ,
        CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT,
        solver.solveLDescr,
        solver.d_bufferL));

    CUSPARSE_CHECK(cusparseSpSV_analysis(
        solver.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one,
        solver.matLt,
        solver.vecQ,
        solver.vecX,
        CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT,
        solver.solveLtDescr,
        solver.d_bufferLt));

    CUDA_CHECK(cudaDeviceSynchronize());
}


static void coarse_solve_gpu(
    const CoarseGpuSolver& solver,
    const double* d_r3,
    double* d_z3,
    int nDOFs3,
    int threads)
{
    int freeBlocks =
        (solver.n + threads - 1) / threads;

    int fullBlocks =
        (nDOFs3 + threads - 1) / threads;

    gather_permuted_coarse_rhs_kernel<<<freeBlocks, threads>>>(
        d_r3,
        solver.d_rhsPerm,
        solver.d_freeIndices,
        solver.d_inversePerm,
        solver.n);

    CUDA_CHECK(cudaGetLastError());

    const double one = 1.0;

    CUSPARSE_CHECK(cusparseSpSV_solve(
        solver.handle,
        CUSPARSE_OPERATION_TRANSPOSE,
        &one,
        solver.matLt,
        solver.vecRhs,
        solver.vecQ,
        CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT,
        solver.solveLDescr));

    CUSPARSE_CHECK(cusparseSpSV_solve(
        solver.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one,
        solver.matLt,
        solver.vecQ,
        solver.vecX,
        CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT,
        solver.solveLtDescr));

    zero_kernel<<<fullBlocks, threads>>>(
        d_z3,
        nDOFs3);

    CUDA_CHECK(cudaGetLastError());

    scatter_permuted_coarse_solution_kernel<<<freeBlocks, threads>>>(
        solver.d_xPerm,
        d_z3,
        solver.d_freeIndices,
        solver.d_perm,
        solver.n);

    CUDA_CHECK(cudaGetLastError());
}


static void apply_vcycle(
    const CoarseGpuSolver& coarseSolver,

    const double* d_r1,
    double* d_z1,

    double* d_r2,
    double* d_z2,

    double* d_r3,
    double* d_z3,

    double* d_intermediate12,
    double* d_intermediate23,

    const double* d_diag1,
    const double* d_diag2,

    const unsigned char* d_fixed1,
    const unsigned char* d_fixed2,
    const unsigned char* d_fixed3,

    const TransferDevice& t12,
    const TransferDevice& t23,

    int nDOFs1,
    int nDOFs2,
    int nDOFs3,

    double omega,
    int threads)
{
    int blocks1 = (nDOFs1 + threads - 1) / threads;
    int blocks2 = (nDOFs2 + threads - 1) / threads;
    int blocks3 = (nDOFs3 + threads - 1) / threads;
    (void)blocks3;
    (void)d_fixed3;

    // z1 = omega * r1 ./ diagK1
    jacobi_assign_kernel<<<blocks1, threads>>>(
        d_r1,
        d_diag1,
        d_fixed1,
        d_z1,
        omega,
        nDOFs1);

    CUDA_CHECK(cudaGetLastError());

    // r2 = R12 * r1. Keep the raw restricted values, including entries
    // marked fixed on level 2, matching Solving_Vcycle.
    restrict_level(
        d_r1,
        d_r2,
        d_intermediate12,
        t12,
        threads);

    // z2 = omega * r2 ./ diagK2 on all level-2 DOFs.
    jacobi_assign_all_kernel<<<blocks2, threads>>>(
        d_r2,
        d_diag2,
        d_z2,
        omega,
        nDOFs2);

    CUDA_CHECK(cudaGetLastError());

    // r3 = R23 * r2. Keep raw entries; the exact coarse callback extracts
    // mh3.freeDOFs internally.
    restrict_level(
        d_r2,
        d_r3,
        d_intermediate23,
        t23,
        threads);

    CUDA_CHECK(cudaDeviceSynchronize());

    // z3 = exact GPU sparse Cholesky solve on the free level-3 DOFs.
    coarse_solve_gpu(
        coarseSolver,
        d_r3,
        d_z3,
        nDOFs3,
        threads);

    // z2 += P23 * z3. Do not project level-2 entries.
    interpolate_add_level(
        d_z3,
        d_z2,
        d_intermediate23,
        t23,
        threads);

    // post-smooth level 2 on all level-2 DOFs.
    jacobi_add_all_kernel<<<blocks2, threads>>>(
        d_r2,
        d_diag2,
        d_z2,
        omega,
        nDOFs2);

    CUDA_CHECK(cudaGetLastError());

    // z1 += P12 * z2
    interpolate_add_level(
        d_z2,
        d_z1,
        d_intermediate12,
        t12,
        threads);

    zero_fixed_kernel<<<blocks1, threads>>>(
        d_z1,
        d_fixed1,
        nDOFs1);

    CUDA_CHECK(cudaGetLastError());

    // post-smooth level 1
    jacobi_add_kernel<<<blocks1, threads>>>(
        d_r1,
        d_diag1,
        d_fixed1,
        d_z1,
        omega,
        nDOFs1);

    CUDA_CHECK(cudaGetLastError());

    zero_fixed_kernel<<<blocks1, threads>>>(
        d_z1,
        d_fixed1,
        nDOFs1);

    CUDA_CHECK(cudaGetLastError());
}


// -----------------------------------------------------------------------------
// Misc helpers
// -----------------------------------------------------------------------------

static int parse_print_flag(const mxArray* print_mx)
{
    if (mxIsChar(print_mx))
    {
        char* s = mxArrayToString(print_mx);

        if (s == nullptr)
            return 0;

        int enabled = (std::strcmp(s, "printP_ON") == 0);

        mxFree(s);

        return enabled;
    }

    if (mxIsNumeric(print_mx))
        return static_cast<int>(mxGetScalar(print_mx)) != 0;

    return 0;
}


// -----------------------------------------------------------------------------
// MEX entry point
// -----------------------------------------------------------------------------

void mexFunction(
    int nlhs,
    mxArray* plhs[],
    int nrhs,
    const mxArray* prhs[])
{
    /*
     * [U, its, relres] = sgld_pcg_vcycle_gpucoarse_cuda_mex(
     *   F, U0,
     *   nodeToElements1, eNodMat1, E1, Ks1,
     *   mh2, mh3,
     *   diagK1, fixedMask1,
     *   diagK2, fixedMask2,
     *   fixedMask3,
     *   cholFac,
     *   coarsePerm,
     *   omega,
     *   tol, maxIT,
     *   printP)
     */

    if (nrhs != 19)
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:nrhs",
            "Expected 19 inputs. See the file header for the call signature.");
    }

    if (nlhs > 3)
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:nlhs",
            "Outputs are [U, its, relres].");
    }

    const mxArray* F_mx = prhs[0];
    const mxArray* U0_mx = prhs[1];

    const mxArray* nodeToElements_mx = prhs[2];
    const mxArray* eNodMat_mx = prhs[3];
    const mxArray* E_mx = prhs[4];
    const mxArray* Ke_mx = prhs[5];

    const mxArray* mh2_mx = prhs[6];
    const mxArray* mh3_mx = prhs[7];

    const mxArray* diag1_mx = prhs[8];
    const mxArray* fixed1_mx = prhs[9];

    const mxArray* diag2_mx = prhs[10];
    const mxArray* fixed2_mx = prhs[11];

    const mxArray* fixed3_mx = prhs[12];
    const mxArray* cholFac_mx = prhs[13];
    const mxArray* coarsePerm_mx = prhs[14];

    double omega = mxGetScalar(prhs[15]);
    double tol = mxGetScalar(prhs[16]);
    int maxIT = static_cast<int>(mxGetScalar(prhs[17]));

    int printEnabled = parse_print_flag(prhs[18]);

    if (!mxIsDouble(F_mx) || mxIsComplex(F_mx) || mxIsSparse(F_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:F", "F must be full real double.");

    if (!mxIsInt32(nodeToElements_mx) || mxIsSparse(nodeToElements_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:nodeToElements", "nodeToElements must be full int32.");

    if (!mxIsInt32(eNodMat_mx) || mxIsSparse(eNodMat_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:eNodMat", "eNodMat must be full int32.");

    if (!mxIsDouble(E_mx) || mxIsComplex(E_mx) || mxIsSparse(E_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:E", "E must be full real double.");

    if (!mxIsDouble(Ke_mx) || mxIsComplex(Ke_mx) || mxIsSparse(Ke_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:Ks", "Ks must be full real double [24 x 24].");

    if (!mxIsDouble(diag1_mx) || mxIsComplex(diag1_mx) || mxIsSparse(diag1_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:diag1", "diagK1 must be full real double.");

    if (!mxIsDouble(diag2_mx) || mxIsComplex(diag2_mx) || mxIsSparse(diag2_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:diag2", "diagK2 must be full real double.");

    if (!mxIsLogical(fixed1_mx) || mxIsSparse(fixed1_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:fixed1", "fixedMask1 must be full logical.");

    if (!mxIsLogical(fixed2_mx) || mxIsSparse(fixed2_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:fixed2", "fixedMask2 must be full logical.");

    if (!mxIsLogical(fixed3_mx) || mxIsSparse(fixed3_mx))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:fixed3", "fixedMask3 must be full logical.");

    if (mxGetM(Ke_mx) != 24 || mxGetN(Ke_mx) != 24)
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:KsSize", "Ks must be [24 x 24].");

    size_t nDOFs1_sz =
        mxGetNumberOfElements(F_mx);

    if (nDOFs1_sz % 3 != 0 || nDOFs1_sz > INT_MAX)
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:FSize",
            "length(F) must be divisible by 3 and fit in int.");
    }

    int nDOFs1 = static_cast<int>(nDOFs1_sz);
    int nNodes1 = nDOFs1 / 3;

    size_t nElems1_sz =
        mxGetNumberOfElements(E_mx);

    if (nElems1_sz > INT_MAX)
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:Esize",
            "numElements exceeds int range.");
    }

    int nElems1 =
        static_cast<int>(nElems1_sz);

    if (mxGetNumberOfElements(diag1_mx) != static_cast<size_t>(nDOFs1))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:diag1Size", "diagK1 length must match F.");

    if (mxGetNumberOfElements(fixed1_mx) != static_cast<size_t>(nDOFs1))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:fixed1Size", "fixedMask1 length must match F.");

    if (static_cast<int>(mxGetM(nodeToElements_mx)) != nNodes1 || mxGetN(nodeToElements_mx) < 8)
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:nodeToElementsSize", "nodeToElements must be [numNodes1 x 8].");

    if (static_cast<int>(mxGetM(eNodMat_mx)) != nElems1 || mxGetN(eNodMat_mx) < 8)
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:eNodMatSize", "eNodMat must be [numElements1 x 8].");

    int nDOFs2 =
        static_cast<int>(mxGetNumberOfElements(diag2_mx));

    if (nDOFs2 % 3 != 0)
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:diag2Size", "diagK2 length must be divisible by 3.");

    if (mxGetNumberOfElements(fixed2_mx) != static_cast<size_t>(nDOFs2))
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:fixed2Size", "fixedMask2 length must match diagK2.");

    int nDOFs3 =
        static_cast<int>(mxGetNumberOfElements(fixed3_mx));

    if (nDOFs3 % 3 != 0)
        mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:fixed3Size", "fixedMask3 length must be divisible by 3.");

    int nNodes2 = nDOFs2 / 3;
    int nNodes3 = nDOFs3 / 3;

    bool hasU0 = !mxIsEmpty(U0_mx);

    if (hasU0)
    {
        if (!mxIsDouble(U0_mx) || mxIsComplex(U0_mx) || mxIsSparse(U0_mx))
            mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:U0", "U0 must be [] or full real double.");

        if (mxGetNumberOfElements(U0_mx) != static_cast<size_t>(nDOFs1))
            mexErrMsgIdAndTxt("sgld_pcg_vcycle_gpucoarse:U0Size", "U0 length must match F.");
    }

    TransferHost h12 =
        parse_level(
            mh2_mx,
            nNodes1,
            nNodes2);

    TransferHost h23 =
        parse_level(
            mh3_mx,
            nNodes2,
            nNodes3);

    unsigned char* h_fixed1 =
        make_fixed_u8(
            fixed1_mx,
            nDOFs1);

    unsigned char* h_fixed2 =
        make_fixed_u8(
            fixed2_mx,
            nDOFs2);

    unsigned char* h_fixed3 =
        make_fixed_u8(
            fixed3_mx,
            nDOFs3);

    const int32_t* h_nodeToElements =
        static_cast<const int32_t*>(
            mxGetData(nodeToElements_mx));

    const int32_t* h_eNodMat =
        static_cast<const int32_t*>(
            mxGetData(eNodMat_mx));

    const double* h_F =
        mxGetDoubles(F_mx);

    const double* h_U0 =
        hasU0 ? mxGetDoubles(U0_mx) : nullptr;

    const double* h_E =
        mxGetDoubles(E_mx);

    const double* h_Ke =
        mxGetDoubles(Ke_mx);

    const double* h_diag1 =
        mxGetDoubles(diag1_mx);

    const double* h_diag2 =
        mxGetDoubles(diag2_mx);

    size_t bytesDOF1 =
        sizeof(double) * static_cast<size_t>(nDOFs1);

    size_t bytesDOF2 =
        sizeof(double) * static_cast<size_t>(nDOFs2);

    size_t bytesDOF3 =
        sizeof(double) * static_cast<size_t>(nDOFs3);

    size_t bytesFixed1 =
        static_cast<size_t>(nDOFs1);

    size_t bytesFixed2 =
        static_cast<size_t>(nDOFs2);

    size_t bytesFixed3 =
        static_cast<size_t>(nDOFs3);

    size_t bytesNodeToElements =
        sizeof(int32_t) * static_cast<size_t>(nNodes1) * 8;

    size_t bytesENodMat =
        sizeof(int32_t) * static_cast<size_t>(nElems1) * 8;

    size_t bytesE =
        sizeof(double) * static_cast<size_t>(nElems1);

    size_t bytesKe =
        sizeof(double) * 24 * 24;

    double* d_b = nullptr;
    double* d_y = nullptr;
    double* d_r = nullptr;
    double* d_z = nullptr;
    double* d_p = nullptr;
    double* d_Ap = nullptr;

    double* d_r2 = nullptr;
    double* d_z2 = nullptr;
    double* d_r3 = nullptr;
    double* d_z3 = nullptr;

    double* d_intermediate12 = nullptr;
    double* d_intermediate23 = nullptr;

    double* d_diag1 = nullptr;
    double* d_diag2 = nullptr;

    unsigned char* d_fixed1 = nullptr;
    unsigned char* d_fixed2 = nullptr;
    unsigned char* d_fixed3 = nullptr;

    int32_t* d_nodeToElements = nullptr;
    int32_t* d_eNodMat = nullptr;

    double* d_E = nullptr;
    double* d_Ke = nullptr;

    TransferDevice t12;
    TransferDevice t23;
    CoarseGpuSolver coarseSolver;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytesDOF1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y), bytesDOF1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_r), bytesDOF1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_z), bytesDOF1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p), bytesDOF1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Ap), bytesDOF1));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_r2), bytesDOF2));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_z2), bytesDOF2));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_r3), bytesDOF3));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_z3), bytesDOF3));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_intermediate12),
        sizeof(double) * static_cast<size_t>(3 * h12.nInter)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_intermediate23),
        sizeof(double) * static_cast<size_t>(3 * h23.nInter)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_diag1), bytesDOF1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_diag2), bytesDOF2));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_fixed1), bytesFixed1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_fixed2), bytesFixed2));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_fixed3), bytesFixed3));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nodeToElements), bytesNodeToElements));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_eNodMat), bytesENodMat));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_E), bytesE));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Ke), bytesKe));

    upload_transfer(h12, t12);
    upload_transfer(h23, t23);

    CUDA_CHECK(cudaMemcpy(d_b, h_F, bytesDOF1, cudaMemcpyHostToDevice));

    if (hasU0)
        CUDA_CHECK(cudaMemcpy(d_y, h_U0, bytesDOF1, cudaMemcpyHostToDevice));
    else
        CUDA_CHECK(cudaMemset(d_y, 0, bytesDOF1));

    CUDA_CHECK(cudaMemcpy(d_diag1, h_diag1, bytesDOF1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_diag2, h_diag2, bytesDOF2, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_fixed1, h_fixed1, bytesFixed1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fixed2, h_fixed2, bytesFixed2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fixed3, h_fixed3, bytesFixed3, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_nodeToElements, h_nodeToElements, bytesNodeToElements, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eNodMat, h_eNodMat, bytesENodMat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E, bytesE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ke, h_Ke, bytesKe, cudaMemcpyHostToDevice));

    mxFree(h_fixed1);
    mxFree(h_fixed2);
    mxFree(h_fixed3);

    MEX_PRINT(
        "Preparing GPU coarse Cholesky: n=%d, nnz=%d",
        static_cast<int>(mxGetM(cholFac_mx)),
        static_cast<int>(mxGetJc(cholFac_mx)[mxGetN(cholFac_mx)]));

    initialize_coarse_gpu_solver(
        cholFac_mx,
        coarsePerm_mx,
        fixed3_mx,
        coarseSolver);

    const int threads = 256;
    const int blocks1 = (nDOFs1 + threads - 1) / threads;

    int its = 0;
    double resnorm = 0.0;

    double normB =
        std::sqrt(
            device_dot(
                d_b,
                d_b,
                nDOFs1));

    if (normB == 0.0)
    {
        plhs[0] =
            mxCreateDoubleMatrix(
                static_cast<mwSize>(nDOFs1),
                1,
                mxREAL);

        double* out =
            mxGetDoubles(plhs[0]);

        std::memset(out, 0, bytesDOF1);

        if (nlhs >= 2)
            plhs[1] = mxCreateDoubleScalar(0.0);

        if (nlhs >= 3)
            plhs[2] = mxCreateDoubleScalar(0.0);

        goto cleanup;
    }

    // r = b - A*y
    launch_kbyu(
        d_y,
        d_Ap,
        d_nodeToElements,
        d_eNodMat,
        d_E,
        d_Ke,
        d_fixed1,
        nDOFs1,
        nNodes1,
        nElems1,
        threads);

    residual_kernel<<<blocks1, threads>>>(
        d_b,
        d_Ap,
        d_r,
        nDOFs1);

    CUDA_CHECK(cudaGetLastError());

    zero_fixed_kernel<<<blocks1, threads>>>(
        d_r,
        d_fixed1,
        nDOFs1);

    CUDA_CHECK(cudaGetLastError());

    // z = Vcycle(r)
    apply_vcycle(
        coarseSolver,
        d_r,
        d_z,
        d_r2,
        d_z2,
        d_r3,
        d_z3,
        d_intermediate12,
        d_intermediate23,
        d_diag1,
        d_diag2,
        d_fixed1,
        d_fixed2,
        d_fixed3,
        t12,
        t23,
        nDOFs1,
        nDOFs2,
        nDOFs3,
        omega,
        threads);

    // p = z
    CUDA_CHECK(cudaMemcpy(
        d_p,
        d_z,
        bytesDOF1,
        cudaMemcpyDeviceToDevice));

    double x1Val =
        device_dot(
            d_z,
            d_r,
            nDOFs1);

    if (!std::isfinite(x1Val))
    {
        mexErrMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:x1Val",
            "Initial z' * r is not finite.");
    }

    for (its = 1; its <= maxIT; ++its)
    {
        // Ap = A*p
        launch_kbyu(
            d_p,
            d_Ap,
            d_nodeToElements,
            d_eNodMat,
            d_E,
            d_Ke,
            d_fixed1,
            nDOFs1,
            nNodes1,
            nElems1,
            threads);

        double pAp =
            device_dot(
                d_p,
                d_Ap,
                nDOFs1);

        if (!std::isfinite(pAp) || pAp == 0.0)
        {
            mexErrMsgIdAndTxt(
                "sgld_pcg_vcycle_gpucoarse:pAp",
                "Invalid p' * Ap = %.12e at iteration %d.",
                pAp,
                its);
        }

        double alpha = x1Val / pAp;

        update_y_r_kernel<<<blocks1, threads>>>(
            d_y,
            d_r,
            d_p,
            d_Ap,
            alpha,
            nDOFs1);

        CUDA_CHECK(cudaGetLastError());

        zero_fixed_kernel<<<blocks1, threads>>>(
            d_y,
            d_fixed1,
            nDOFs1);

        CUDA_CHECK(cudaGetLastError());

        zero_fixed_kernel<<<blocks1, threads>>>(
            d_r,
            d_fixed1,
            nDOFs1);

        CUDA_CHECK(cudaGetLastError());

        resnorm =
            std::sqrt(
                device_dot(
                    d_r,
                    d_r,
                    nDOFs1)) / normB;

        if (printEnabled)
        {
            MEX_PRINT(
                " It.: %4d Res.: %16.6e",
                its,
                resnorm);
        }

        if (resnorm < tol)
        {
            MEX_PRINT(
                "CG solver converged at iteration%5d to a solution with relative residual%16.6e",
                its,
                resnorm);

            break;
        }

        // z = Vcycle(r)
        apply_vcycle(
            coarseSolver,
            d_r,
            d_z,
            d_r2,
            d_z2,
            d_r3,
            d_z3,
            d_intermediate12,
            d_intermediate23,
            d_diag1,
            d_diag2,
            d_fixed1,
            d_fixed2,
            d_fixed3,
            t12,
            t23,
            nDOFs1,
            nDOFs2,
            nDOFs3,
            omega,
            threads);

        double x2Val =
            device_dot(
                d_z,
                d_r,
                nDOFs1);

        if (!std::isfinite(x2Val))
        {
            mexErrMsgIdAndTxt(
                "sgld_pcg_vcycle_gpucoarse:x2Val",
                "Invalid z' * r = %.12e at iteration %d.",
                x2Val,
                its);
        }

        double beta = x2Val / x1Val;

        update_p_kernel<<<blocks1, threads>>>(
            d_p,
            d_z,
            beta,
            nDOFs1);

        CUDA_CHECK(cudaGetLastError());

        zero_fixed_kernel<<<blocks1, threads>>>(
            d_p,
            d_fixed1,
            nDOFs1);

        CUDA_CHECK(cudaGetLastError());

        x1Val = x2Val;
    }

    if (its > maxIT)
    {
        its = maxIT;

        mexWarnMsgIdAndTxt(
            "sgld_pcg_vcycle_gpucoarse:maxIT",
            "Exceed the maximum iterate numbers.");

        MEX_PRINT(
            "The iterative process stops at residual = %10.4f",
            resnorm);
    }

    plhs[0] =
        mxCreateDoubleMatrix(
            static_cast<mwSize>(nDOFs1),
            1,
            mxREAL);

    CUDA_CHECK(cudaMemcpy(
        mxGetDoubles(plhs[0]),
        d_y,
        bytesDOF1,
        cudaMemcpyDeviceToHost));

    if (nlhs >= 2)
        plhs[1] =
            mxCreateDoubleScalar(
                static_cast<double>(its));

    if (nlhs >= 3)
        plhs[2] =
            mxCreateDoubleScalar(resnorm);

cleanup:

    if (d_b) CUDA_CHECK(cudaFree(d_b));
    if (d_y) CUDA_CHECK(cudaFree(d_y));
    if (d_r) CUDA_CHECK(cudaFree(d_r));
    if (d_z) CUDA_CHECK(cudaFree(d_z));
    if (d_p) CUDA_CHECK(cudaFree(d_p));
    if (d_Ap) CUDA_CHECK(cudaFree(d_Ap));

    if (d_r2) CUDA_CHECK(cudaFree(d_r2));
    if (d_z2) CUDA_CHECK(cudaFree(d_z2));
    if (d_r3) CUDA_CHECK(cudaFree(d_r3));
    if (d_z3) CUDA_CHECK(cudaFree(d_z3));

    if (d_intermediate12) CUDA_CHECK(cudaFree(d_intermediate12));
    if (d_intermediate23) CUDA_CHECK(cudaFree(d_intermediate23));

    if (d_diag1) CUDA_CHECK(cudaFree(d_diag1));
    if (d_diag2) CUDA_CHECK(cudaFree(d_diag2));

    if (d_fixed1) CUDA_CHECK(cudaFree(d_fixed1));
    if (d_fixed2) CUDA_CHECK(cudaFree(d_fixed2));
    if (d_fixed3) CUDA_CHECK(cudaFree(d_fixed3));

    if (d_nodeToElements) CUDA_CHECK(cudaFree(d_nodeToElements));
    if (d_eNodMat) CUDA_CHECK(cudaFree(d_eNodMat));
    if (d_E) CUDA_CHECK(cudaFree(d_E));
    if (d_Ke) CUDA_CHECK(cudaFree(d_Ke));

    destroy_coarse_gpu_solver(coarseSolver);
    free_transfer(t12);
    free_transfer(t23);
}
