/*
 * sgld_kbyu_cuda_mex.cu
 *
 * Vertex-based matrix-free K*u CUDA MEX for SGLDBench.
 *
 * MATLAB call:
 *
 *   Y = sgld_kbyu_cuda_mex( ...
 *       U, ...
 *       int32(meshHierarchy_(1).nodeToElements), ...
 *       int32(meshHierarchy_(1).eNodMat), ...
 *       meshHierarchy_(1).eleModulus, ...
 *       meshHierarchy_(1).Ke, ...
 *       meshHierarchy_(1).resX, ...
 *       meshHierarchy_(1).resY, ...
 *       meshHierarchy_(1).resZ ...
 *   );
 *
 * Compile:
 *
 *   clear mex
 *   mexcuda -R2018a src/MEXfuncs/sgld_kbyu_cuda_mex.cu
 *
 * Assumptions:
 *
 *   U              : [3*numNodes x 1] double
 *   Y              : [3*numNodes x 1] double
 *   nodeToElements : [numNodes x 8] int32, MATLAB 1-based, 0 means invalid
 *   eNodMat        : [numElements x 8] int32, MATLAB 1-based
 *   E              : [numElements x 1] double
 *   Ke             : [24 x 24] double, MATLAB column-major
 *
 * One CUDA thread owns one global node and computes its 3 output DOFs.
 */

#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <climits>
#include <cstdio>


#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            mexErrMsgIdAndTxt("sgld_kbyu_cuda:cuda",                         \
                "CUDA error at %s:%d: %s",                                   \
                __FILE__, __LINE__, cudaGetErrorString(err__));               \
        }                                                                    \
    } while (0)


#define MEX_PRINT(...)                                                       \
    do {                                                                     \
        mexPrintf(__VA_ARGS__);                                               \
        mexPrintf("\n");                                                     \
        mexEvalString("drawnow;");                                           \
    } while (0)

__global__
void jacobi_kernel(
    const double* r,
    const double* invDiag,
    double* z,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        z[i] = invDiag[i] * r[i];
}

__global__ void kbyu_kernel(
    const double* __restrict__ U,               // [3*numNodes]
    double* __restrict__ Y,                     // [3*numNodes]

    const int32_t* __restrict__ nodeToElements, // [numNodes x 8], MATLAB column-major
    const int32_t* __restrict__ eNodMat,        // [numElements x 8], MATLAB column-major

    const double* __restrict__ E,               // [numElements]
    const double* __restrict__ Ke,              // [24 x 24], MATLAB column-major

    int numNodes,
    int numElements,
    int nx, int ny, int nz)
{
    // One thread owns one node.
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= numNodes) return;

    // Currently unused. Keep these in the signature for compatibility/debugging.
    (void)nx;
    (void)ny;
    (void)nz;

    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    int elemNodes[8];
    double Ue[24];

    // Each structured hex node belongs to up to 8 surrounding elements.
    #pragma unroll
    for (int a = 0; a < 8; ++a)
    {
        // nodeToElements(node, a) in MATLAB column-major:
        //
        //   nodeToElements[node + a*numNodes]
        //
        // MATLAB stores 1-based element ids. 0 means invalid/no element.
        int elem = nodeToElements[node + a * numNodes] - 1;

        if (elem < 0 || elem >= numElements)
            continue;

        int localNode = -1;

        // eNodMat(elem, j) in MATLAB column-major:
        //
        //   eNodMat[elem + j*numElements]
        //
        // MATLAB stores 1-based node ids.

        // Gather the 8 global nodes of this element from eNodMat(elem, :).
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int n = eNodMat[elem + j * numElements] - 1;
            elemNodes[j] = n;

            if (n == node)
                localNode = j;
        }

        // If this happens, nodeToElements/eNodMat are inconsistent.
        if (localNode < 0)
            continue;

        // Build element displacement vector Ue with 24 DOFs:
        //
        //   [ux0 uy0 uz0 ux1 uy1 uz1 ... ux7 uy7 uz7]
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int n = elemNodes[j];
            /*
            if (n < 0 || n >= numNodes)
            {
                Ue[3*j + 0] = 0.0;
                Ue[3*j + 1] = 0.0;
                Ue[3*j + 2] = 0.0;
                continue;
            }
            */
            int base = 3 * n;

            Ue[3*j + 0] = U[base + 0];
            Ue[3*j + 1] = U[base + 1];
            Ue[3*j + 2] = U[base + 2];
        }

        // Current node corresponds to local rows:
        //
        //   row0 + 0
        //   row0 + 1
        //   row0 + 2
        //
        // in the 24x24 element stiffness matrix.
        int row0 = 3 * localNode;

        double y0 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;

        // Compute only the 3 rows needed for this node:
        //
        //   [y0 y1 y2]^T = Ke(row0:row0+2, :) * Ue
        //
        // Ke(row, col) in MATLAB column-major:
        //
        //   Ke[row + col*24]
        #pragma unroll
        for (int c = 0; c < 24; ++c)
        {
            double u = Ue[c];

            y0 += Ke[(row0 + 0) + c * 24] * u;
            y1 += Ke[(row0 + 1) + c * 24] * u;
            y2 += Ke[(row0 + 2) + c * 24] * u;
        }

        // Scale this element's contribution by its material stiffness/modulus.
        double Ee = E[elem];

        sum0 += Ee * y0;
        sum1 += Ee * y1;
        sum2 += Ee * y2;
    }

    // One thread owns this node, so no atomics are needed.
    int out = 3 * node;

    Y[out + 0] = sum0;
    Y[out + 1] = sum1;
    Y[out + 2] = sum2;
}


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*
        MATLAB signature:

        Y = sgld_kbyu_cuda_mex(U, nodeToElements, eNodMat, E, Ke, nx, ny, nz)
    */

    //MEX_PRINT("MEX 1: entered sgld_kbyu_cuda_mex");
    //MEX_PRINT("MEX 1.1: nrhs = %d, nlhs = %d", nrhs, nlhs);

    if (nrhs != 8) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:nrhs",
            "Expected 8 inputs: U, nodeToElements, eNodMat, E, Ke, nx, ny, nz."
        );
    }

    if (nlhs > 1) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:nlhs",
            "Expected one output: Y."
        );
    }

    const mxArray* U_mx              = prhs[0];
    const mxArray* nodeToElements_mx = prhs[1];
    const mxArray* eNodMat_mx        = prhs[2];
    const mxArray* E_mx              = prhs[3];
    const mxArray* Ke_mx             = prhs[4];

    int nx = (int)mxGetScalar(prhs[5]);
    int ny = (int)mxGetScalar(prhs[6]);
    int nz = (int)mxGetScalar(prhs[7]);

    //MEX_PRINT("MEX 2: read inputs");
    //MEX_PRINT("MEX 2.1: nx=%d ny=%d nz=%d", nx, ny, nz);

    // -----------------------------
    // Validate types
    // -----------------------------

    if (!mxIsDouble(U_mx) || mxIsComplex(U_mx)) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:U",
            "U must be a real double array."
        );
    }
    if (mxIsSparse(U_mx)) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:USparse",
            "U must be a full dense double vector, not sparse. Use full(U) before calling the CUDA MEX."
        );
    }
    if (!mxIsInt32(nodeToElements_mx)) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:nodeToElements",
            "nodeToElements must be int32. Use int32(meshHierarchy_(1).nodeToElements)."
        );
    }

    if (!mxIsInt32(eNodMat_mx)) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:eNodMat",
            "eNodMat must be int32. Use int32(meshHierarchy_(1).eNodMat)."
        );
    }

    if (!mxIsDouble(E_mx) || mxIsComplex(E_mx)) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:E",
            "E / eleModulus must be a real double array."
        );
    }

    if (!mxIsDouble(Ke_mx) || mxIsComplex(Ke_mx)) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:Ke",
            "Ke must be a real double array."
        );
    }
    if (mxIsSparse(nodeToElements_mx)) {
        mexErrMsgIdAndTxt("sgld_kbyu_cuda:nodeToElementsSparse",
            "nodeToElements must be full int32, not sparse.");
    }

    if (mxIsSparse(eNodMat_mx)) {
        mexErrMsgIdAndTxt("sgld_kbyu_cuda:eNodMatSparse",
            "eNodMat must be full int32, not sparse.");
    }

    if (mxIsSparse(E_mx)) {
        mexErrMsgIdAndTxt("sgld_kbyu_cuda:ESparse",
            "E / eleModulus must be full double, not sparse.");
    }

    if (mxIsSparse(Ke_mx)) {
        mexErrMsgIdAndTxt("sgld_kbyu_cuda:KeSparse",
            "Ke must be full double, not sparse.");
    }
    // -----------------------------
    // Infer sizes
    // -----------------------------

    size_t numU = mxGetNumberOfElements(U_mx);

    if (numU % 3 != 0) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:USize",
            "U length must be divisible by 3."
        );
    }

    size_t numNodes_sz = numU / 3;
    size_t numElements_sz = mxGetNumberOfElements(E_mx);

    if (numNodes_sz > INT_MAX || numElements_sz > INT_MAX) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:size",
            "numNodes or numElements exceeds int range."
        );
    }

    int numNodes    = (int)numNodes_sz;
    int numElements = (int)numElements_sz;

    //MEX_PRINT("MEX 3: inferred sizes");
    //MEX_PRINT("MEX 3.1: numNodes = %d", numNodes);
    //MEX_PRINT("MEX 3.2: numElements = %d", numElements);
    //MEX_PRINT("MEX 3.3: numDOFs = %d", 3 * numNodes);

    /*MEX_PRINT("MEX 3.4: size(nodeToElements) = [%llu x %llu]",
        (unsigned long long)mxGetM(nodeToElements_mx),
        (unsigned long long)mxGetN(nodeToElements_mx));

    MEX_PRINT("MEX 3.5: size(eNodMat) = [%llu x %llu]",
        (unsigned long long)mxGetM(eNodMat_mx),
        (unsigned long long)mxGetN(eNodMat_mx));

    MEX_PRINT("MEX 3.6: size(E) = [%llu x %llu]",
        (unsigned long long)mxGetM(E_mx),
        (unsigned long long)mxGetN(E_mx));

    MEX_PRINT("MEX 3.7: size(Ke) = [%llu x %llu]",
        (unsigned long long)mxGetM(Ke_mx),
        (unsigned long long)mxGetN(Ke_mx));
    */
    // nodeToElements should be [numNodes x 8].
    if ((int)mxGetM(nodeToElements_mx) != numNodes || mxGetN(nodeToElements_mx) < 8) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:nodeToElementsSize",
            "nodeToElements must have size [numNodes x 8]. If yours is [8 x numNodes], the kernel indexing must be changed."
        );
    }

    // eNodMat should be [numElements x 8].
    if ((int)mxGetM(eNodMat_mx) != numElements || mxGetN(eNodMat_mx) < 8) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:eNodMatSize",
            "eNodMat must have size [numElements x 8]."
        );
    }

    // Ke should be [24 x 24].
    if (mxGetM(Ke_mx) != 24 || mxGetN(Ke_mx) != 24) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:KeSize",
            "Ke must have size [24 x 24]."
        );
    }

    // -----------------------------
    // Host pointers
    // -----------------------------

    const double*  h_U              = mxGetDoubles(U_mx);
    const int32_t* h_nodeToElements = static_cast<const int32_t*>(mxGetData(nodeToElements_mx));
    const int32_t* h_eNodMat        = static_cast<const int32_t*>(mxGetData(eNodMat_mx));
    const double*  h_E              = mxGetDoubles(E_mx);
    const double*  h_Ke             = mxGetDoubles(Ke_mx);

    // Output Y.
    plhs[0] = mxCreateDoubleMatrix((mwSize)(3 * numNodes), 1, mxREAL);
    double* h_Y = mxGetDoubles(plhs[0]);

    // -----------------------------
    // Allocate device memory
    // -----------------------------

    double* d_U = nullptr;
    double* d_Y = nullptr;
    int32_t* d_nodeToElements = nullptr;
    int32_t* d_eNodMat = nullptr;
    double* d_E = nullptr;
    double* d_Ke = nullptr;

    size_t bytesU              = sizeof(double)  * (size_t)3 * (size_t)numNodes;
    size_t bytesY              = sizeof(double)  * (size_t)3 * (size_t)numNodes;
    size_t bytesNodeToElements = sizeof(int32_t) * (size_t)numNodes * (size_t)8;
    size_t bytesENodMat        = sizeof(int32_t) * (size_t)numElements * (size_t)8;
    size_t bytesE              = sizeof(double)  * (size_t)numElements;
    size_t bytesKe             = sizeof(double)  * (size_t)24 * (size_t)24;

    /*
    MEX_PRINT("MEX 4: memory sizes");
    MEX_PRINT("MEX 4.1: U              = %.3f GB", bytesU / 1e9);
    MEX_PRINT("MEX 4.2: Y              = %.3f GB", bytesY / 1e9);
    MEX_PRINT("MEX 4.3: nodeToElements = %.3f GB", bytesNodeToElements / 1e9);
    MEX_PRINT("MEX 4.4: eNodMat        = %.3f GB", bytesENodMat / 1e9);
    MEX_PRINT("MEX 4.5: E              = %.3f GB", bytesE / 1e9);
    MEX_PRINT("MEX 4.6: Ke             = %.6f GB", bytesKe / 1e9);
    */
    size_t freeMem = 0;
    size_t totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    /*
    MEX_PRINT("MEX 4.7: GPU free memory  = %.3f GB", freeMem / 1e9);
    MEX_PRINT("MEX 4.8: GPU total memory = %.3f GB", totalMem / 1e9);

    MEX_PRINT("MEX 5: starting cudaMallocs");
    */
    CUDA_CHECK(cudaMalloc((void**)&d_U, bytesU));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, bytesY));
    CUDA_CHECK(cudaMalloc((void**)&d_nodeToElements, bytesNodeToElements));
    CUDA_CHECK(cudaMalloc((void**)&d_eNodMat, bytesENodMat));
    CUDA_CHECK(cudaMalloc((void**)&d_E, bytesE));
    CUDA_CHECK(cudaMalloc((void**)&d_Ke, bytesKe));

    //MEX_PRINT("MEX 5.1: finished cudaMallocs");

    // -----------------------------
    // Copy host -> device
    // -----------------------------

    //MEX_PRINT("MEX 6: copying U to GPU");
    CUDA_CHECK(cudaMemcpy(d_U, h_U, bytesU, cudaMemcpyHostToDevice));

    //MEX_PRINT("MEX 6.1: copying nodeToElements to GPU");
    CUDA_CHECK(cudaMemcpy(d_nodeToElements, h_nodeToElements, bytesNodeToElements, cudaMemcpyHostToDevice));

    //MEX_PRINT("MEX 6.2: copying eNodMat to GPU");
    CUDA_CHECK(cudaMemcpy(d_eNodMat, h_eNodMat, bytesENodMat, cudaMemcpyHostToDevice));

    //MEX_PRINT("MEX 6.3: copying E to GPU");
    CUDA_CHECK(cudaMemcpy(d_E, h_E, bytesE, cudaMemcpyHostToDevice));

    //MEX_PRINT("MEX 6.4: copying Ke to GPU");
    CUDA_CHECK(cudaMemcpy(d_Ke, h_Ke, bytesKe, cudaMemcpyHostToDevice));

    //MEX_PRINT("MEX 6.5: memset Y");
    CUDA_CHECK(cudaMemset(d_Y, 0, bytesY));

    //MEX_PRINT("MEX 6.6: finished host-to-device copies");

    // -----------------------------
    // Launch kernel
    // -----------------------------

    int threads = 256;
    int blocks = (numNodes + threads - 1) / threads;

    //MEX_PRINT("MEX 7: launching kernel");
    //MEX_PRINT("MEX 7.1: blocks = %d, threads = %d", blocks, threads);

    kbyu_kernel<<<blocks, threads>>>(
        d_U,
        d_Y,
        d_nodeToElements,
        d_eNodMat,
        d_E,
        d_Ke,
        numNodes,
        numElements,
        nx, ny, nz
    );

    //MEX_PRINT("MEX 7.2: kernel launched, checking launch error");
    CUDA_CHECK(cudaGetLastError());

    //MEX_PRINT("MEX 7.3: synchronizing kernel");
    CUDA_CHECK(cudaDeviceSynchronize());

    //MEX_PRINT("MEX 7.4: kernel finished");

    // -----------------------------
    // Copy device -> host
    // -----------------------------

    //MEX_PRINT("MEX 8: copying Y back to MATLAB");
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, bytesY, cudaMemcpyDeviceToHost));
    //MEX_PRINT("MEX 8.1: finished copying Y back");

    // -----------------------------
    // Free device memory
    // -----------------------------

    //MEX_PRINT("MEX 9: freeing GPU memory");

    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_nodeToElements));
    CUDA_CHECK(cudaFree(d_eNodMat));
    CUDA_CHECK(cudaFree(d_E));
    CUDA_CHECK(cudaFree(d_Ke));

    //MEX_PRINT("MEX 9.1: finished freeing GPU memory");
    //MEX_PRINT("MEX 10: exiting sgld_kbyu_cuda_mex");
}

