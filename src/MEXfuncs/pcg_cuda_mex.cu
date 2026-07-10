/*
 * sgld_pcg_cuda_mex.cu
 *
 * First-draft CUDA MEX solver for SGLDBench:
 *   - compact eNodMat-based matrix-free K*u
 *   - single GPU
 *   - Jacobi preconditioned CG
 *   - vectors stay on GPU during iterations
 *
 * MATLAB call:
 *   [U, relres, its] = sgld_pcg_cuda_mex(Ke, F, U0, eNodMat, E, diagK, fixedDOFs, tol, maxIT, printEvery)
 *
 * Compile idea:
 *   mexcuda -R2018a sgld_pcg_cuda_mex.cu
 */

#include "mex.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            mexErrMsgIdAndTxt("sgld_kbyu_cuda:cuda",                         \
                "CUDA error: %s", cudaGetErrorString(err__));                \
        }                                                                    \
    } while (0)
/*
__global__ void kbyu_kernel(
    // TODO: Add stuff to the shared memory
    const double* __restrict__ U,
    double* __restrict__ Y,
    const int32_t* __restrict__ nodeToElements,
    const int32_t* __restrict__ eNodMat,
    const double* __restrict__ E,
    int numNodes,
    int nx, int ny, int nz)
{
    //TODO: If fasten node skip

    // Assuming the active nodes are in a 1D array
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= numNodes) return;

    // Declarimg variables
    // TODO.Does putting these choke memory?
    int elements[8];
    double Ue[24];
    double Ye[24];
    float sum[3] = {0, 0, 0};

    // MATLAB stores eNodMat column-major: e + j*numElements
    // Assuming that we store nodeToVoxels in such fashion that 128 byte access contains 32 column information
    // So 32 nodes/rows are accessed
    // Lets assume that for this logic is like finding eth row and jth column
    //TODO.FIX

    // Getting anchor nodes per element
    #pragma unroll
    for (int j = 1; j < 9; ++j) voxels[j] = nodeToElements[e*8+j] - 1;

    #pragma unroll
    // Per voxel computations
    for (int j = 0; j < 8; ++j) {
        auto voxel = voxels[j];

        if(voxel == 0){
            continue;
        }

        // Gather other nodes of this element
        // Knowing that the elem is the anchor node for the voxel, the other nodes are

        // Gather displacement for the voxels
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int base = 3 * voxel[j];
            Ue[3*j + 0] = U[base + 0];
            Ue[3*j + 1] = U[base + 1];
            Ue[3*j + 2] = U[base + 2];
        }
    }

}
*/
__global__ void kbyu_kernel(
    const double* __restrict__ U,              // [3*numNodes]
    double* __restrict__ Y,                    // [3*numNodes]

    const int32_t* __restrict__ nodeToElements, // [numNodes x 8], MATLAB column-major
    const int32_t* __restrict__ eNodMat,        // [numElements x 8], MATLAB column-major

    const double* __restrict__ E,              // [numElements]
    const double* __restrict__ Ke,             // [24 x 24], MATLAB column-major

    int numNodes,
    int numElements,
    int nx, int ny, int nz)
{
    // One thread owns one node.
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= numNodes) return;

    // nx, ny, nz currently unused.
    // They are only needed if you derive connectivity analytically.
    (void)nx;
    (void)ny;
    (void)nz;

    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    // Local arrays.
    // These usually live in registers unless register pressure gets too high.
    int elemNodes[8];
    double Ue[24];

    // Each node can belong to up to 8 voxel elements in a structured hex mesh.
    #pragma unroll
    for (int a = 0; a < 8; ++a)
    {
        // nodeToElements is assumed MATLAB-style:
        //
        // nodeToElements(node, a)
        //
        // MATLAB column-major indexing:
        //
        // node + a*numNodes
        //
        // Values are assumed 1-based from MATLAB.
        int elem = nodeToElements[node + a * numNodes] - 1;

        // MATLAB may store 0 for invalid/no element.
        // After subtracting 1, invalid becomes -1.
        if (elem < 0 || elem >= numElements)
            continue;

        int localNode = -1;

        // Gather the 8 global nodes of this element.
        //
        // eNodMat(elem, j)
        //
        // MATLAB column-major:
        //
        // elem + j*numElements
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int n = eNodMat[elem + j * numElements] - 1;
            elemNodes[j] = n;

            if (n == node)
                localNode = j;
        }

        // This should not happen if nodeToElements and eNodMat are consistent.
        if (localNode < 0)
            continue;

        // Build Ue = displacement vector of the element.
        //
        // Ue =
        // [
        //   ux(node0), uy(node0), uz(node0),
        //   ux(node1), uy(node1), uz(node1),
        //   ...
        // ]
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

        // Current node's local DOF rows inside Ke.
        //
        // localNode = 0 -> rows 0,1,2
        // localNode = 1 -> rows 3,4,5
        // ...
        int row0 = 3 * localNode;

        double Ee = E[elem];

        double y0 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;

        // Compute only the 3 rows of Ke needed for this node:
        //
        // [y0 y1 y2]^T = Ee * Ke(row0:row0+2, :) * Ue
        //
        // Ke is assumed MATLAB column-major [24 x 24]:
        //
        // Ke(row, col) -> Ke[row + col*24]
        #pragma unroll
        for (int c = 0; c < 24; ++c)
        {
            double u = Ue[c];

            y0 += Ke[(row0 + 0) + c * 24] * u;
            y1 += Ke[(row0 + 1) + c * 24] * u;
            y2 += Ke[(row0 + 2) + c * 24] * u;
        }

        sum0 += Ee * y0;
        sum1 += Ee * y1;
        sum2 += Ee * y2;
    }

    // One thread owns this node, so no atomicAdd needed.
    int out = 3 * node;

    Y[out + 0] = sum0;
    Y[out + 1] = sum1;
    Y[out + 2] = sum2;
}
/*
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
if (nrhs < 9 || nrhs > 10) {
    mexErrMsgIdAndTxt("sgld_pcg_cuda:nrhs", "Expected 9 or 10 inputs: Ke,F,U0,eNodMat,E,diagK,fixedDOFs,tol,maxIT,printP");
}
if (nlhs > 3) mexErrMsgIdAndTxt("sgld_pcg_cuda:nlhs", "Outputs: [U, relres, its]");

const mxArray* Ke_mx      = prhs[0];
const mxArray* F_mx       = prhs[1];
const mxArray* U0_mx      = prhs[2];
const mxArray* eNodMat_mx = prhs[3];
const mxArray* E_mx       = prhs[4];
const mxArray* diagK_mx   = prhs[5];
const mxArray* fixed_mx   = prhs[6];

double tol = mxGetScalar(prhs[7]);
int maxIT = (int)mxGetScalar(prhs[8]);
int printEvery = (nrhs == 10) ? (int)mxGetScalar(prhs[9]) : 0;
*/

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*
        MATLAB signature:

        Y = sgld_kbyu_cuda_mex(U, nodeToElements, eNodMat, E, Ke, nx, ny, nz)

        U              : [3*numNodes x 1] double
        nodeToElements : [numNodes x 8] int32, MATLAB 1-based, 0 means invalid
        eNodMat        : [numElements x 8] int32, MATLAB 1-based
        E              : [numElements x 1] double
        Ke             : [24 x 24] double
        nx, ny, nz     : scalar grid resolution values
    */

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

    // -----------------------------
    // Validate types
    // -----------------------------

    if (!mxIsDouble(U_mx) || mxIsComplex(U_mx)) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:U",
            "U must be a real double array."
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

    // nodeToElements should be [numNodes x 8]
    if ((int)mxGetM(nodeToElements_mx) != numNodes || mxGetN(nodeToElements_mx) < 8) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:nodeToElementsSize",
            "nodeToElements must have size [numNodes x 8]."
        );
    }

    // eNodMat should be [numElements x 8]
    if ((int)mxGetM(eNodMat_mx) != numElements || mxGetN(eNodMat_mx) < 8) {
        mexErrMsgIdAndTxt(
            "sgld_kbyu_cuda:eNodMatSize",
            "eNodMat must have size [numElements x 8]."
        );
    }

    // Ke should be [24 x 24]
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

    // Output Y
    plhs[0] = mxCreateDoubleMatrix(3 * numNodes, 1, mxREAL);
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

    size_t bytesU              = sizeof(double)  * 3 * numNodes;
    size_t bytesY              = sizeof(double)  * 3 * numNodes;
    size_t bytesNodeToElements = sizeof(int32_t) * numNodes * 8;
    size_t bytesENodMat        = sizeof(int32_t) * numElements * 8;
    size_t bytesE              = sizeof(double)  * numElements;
    size_t bytesKe             = sizeof(double)  * 24 * 24;

    CUDA_CHECK(cudaMalloc((void**)&d_U, bytesU));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, bytesY));
    CUDA_CHECK(cudaMalloc((void**)&d_nodeToElements, bytesNodeToElements));
    CUDA_CHECK(cudaMalloc((void**)&d_eNodMat, bytesENodMat));
    CUDA_CHECK(cudaMalloc((void**)&d_E, bytesE));
    CUDA_CHECK(cudaMalloc((void**)&d_Ke, bytesKe));

    // -----------------------------
    // Copy host -> device
    // -----------------------------

    CUDA_CHECK(cudaMemcpy(d_U, h_U, bytesU, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodeToElements, h_nodeToElements, bytesNodeToElements, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eNodMat, h_eNodMat, bytesENodMat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E, bytesE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ke, h_Ke, bytesKe, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_Y, 0, bytesY));

    // -----------------------------
    // Launch kernel
    // -----------------------------

    int threads = 256;
    int blocks = (numNodes + threads - 1) / threads;

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

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -----------------------------
    // Copy device -> host
    // -----------------------------

    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, bytesY, cudaMemcpyDeviceToHost));

    // -----------------------------
    // Free device memory
    // -----------------------------

    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_nodeToElements));
    CUDA_CHECK(cudaFree(d_eNodMat));
    CUDA_CHECK(cudaFree(d_E));
    CUDA_CHECK(cudaFree(d_Ke));
}