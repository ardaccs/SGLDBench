#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <climits>
#include <cstdio>

// Elementary stiffness matrix (4.6 kB)
__constant__ double c_Ke[24 * 24];

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            mexErrMsgIdAndTxt("kbyu_single_gpu:cuda",                         \
                "CUDA error at %s:%d: %s",                                   \
                __FILE__, __LINE__, cudaGetErrorString(err__));               \
        }                                                                    \
    } while (0)

// MEX PRINT function that prints to MATLAB, but can hidner performance
#define MEX_PRINT(...)                                                       \
    do {                                                                     \
        mexPrintf(__VA_ARGS__);                                               \
        mexPrintf("\n");                                                     \
        mexEvalString("drawnow;");                                           \
    } while (0)

__global__ void kbyu_kernel(
    const double* __restrict__ U,               // [3*numNodes]
    double* __restrict__ Y,                     // [3*numNodes]

    const int32_t* __restrict__ nodeToElements, // [numNodes x 8], MATLAB column-major
    const int32_t* __restrict__ eNodMat,        // [numElements x 8], MATLAB column-major

    const double* __restrict__ E,               // [numElements]

    int numNodes,
    int numElements)
{
    // This kernel performs a node based gather operative matrix free matrix-vector multiplication
    // Each thread owns an active node, thus there is numNodes threads
    
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= numNodes) return;

    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    int elemNodes[8];
    double Ue[24];

    // Get the elements touching the node
    #pragma unroll
    for (int a = 0; a < 8; ++a){

        // Get the elem id from nodeToElements
        int elem = nodeToElements[node + a * numNodes] - 1;

        if (elem < 0 || elem >= numElements)
            continue;

        //TODO.When you first load or build your nodeToElements and eNodMat arrays in MATLAB or C++, run a quick validation pass there once.
        int localNode = -1;

        // Gather the 8 global nodes of this element from eNodMat(elem, :).
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int n = eNodMat[elem + j * numElements] - 1;
            elemNodes[j] = n;
            if (n == node)
                localNode = j;
        }
        /*// If this happens, nodeToElements/eNodMat are inconsistent.
        if (localNode < 0)
            continue;*/

        // Build element displacement vector Ue
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int n = elemNodes[j];
            //TODO. Also validate in MATLAB OR C++
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
        // in the 24x24 element stiffness matrix.
        int row0 = 3 * localNode;

        double y0 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;

        double Ee = E[elem];
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
            double Ee_u = Ee * Ue[c];

            sum0 = __fma_rn(c_Ke[row0 + c * 24],       Ee_u, sum0);
            sum1 = __fma_rn(c_Ke[(row0 + 1) + c * 24], Ee_u, sum1);
            sum2 = __fma_rn(c_Ke[(row0 + 2) + c * 24], Ee_u, sum2);
        }
    }
    
    int out = 3 * node;
    Y[out + 0] = sum0;
    Y[out + 1] = sum1;
    Y[out + 2] = sum2;

}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // inputs:
    // U, Ke, nodMapBack, nodMapForward, eleMapForward, E, nx, ny, nz

    if (nrhs != 8) {
        mexErrMsgIdAndTxt("sgld:nrhs", "Need 8 inputs.");
    }

    const mxArray* U_mx = prhs[0];
    const mxArray* nodeToElements_mx = prhs[1];
    const mxArray* eNodMat_mx = prhs[2];
    const mxArray* Ke_mx = prhs[3];
    const mxArray* E_mx = prhs[4];

    int nx = (int)mxGetScalar(prhs[5]);
    int ny = (int)mxGetScalar(prhs[6]);
    int nz = (int)mxGetScalar(prhs[7]);

    int numDOFs = (int)mxGetNumberOfElements(U_mx);
    int numNodes = numDOFs / 3;
    int numElements = (int)mxGetNumberOfElements(E_mx);

    const double* h_U = mxGetDoubles(U_mx);
    const double* h_Ke = mxGetDoubles(Ke_mx);
    const int32_t* h_nodeToElements =
        static_cast<const int32_t*>(mxGetData(nodeToElements_mx));

    const int32_t* h_eNodMat = static_cast<const int32_t*>(mxGetData(eNodMat_mx));
    const double* h_E = mxGetDoubles(E_mx);

    plhs[0] = mxCreateDoubleMatrix(numDOFs, 1, mxREAL);
    double* h_Y = mxGetDoubles(plhs[0]);

    double *d_U, *d_Y, *d_E;
    int32_t *d_nodeToElements, *d_eNodMat;

    CUDA_CHECK(cudaMalloc(&d_U, numDOFs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Y, numDOFs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_E, numElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nodeToElements, numNodes * 8 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_eNodMat, numElements * 8 * sizeof(int32_t)));



    CUDA_CHECK(cudaMemcpy(d_U, h_U, numDOFs * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, numDOFs * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodeToElements, h_nodeToElements, numNodes * 8 * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eNodMat, h_eNodMat, numElements * 8 * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E, numElements * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(c_Ke, h_Ke, 24 * 24 * sizeof(double)));

    int block = 256;
    int grid = (numNodes + block - 1) / block;

    kbyu_kernel<<<grid, block>>>(
        d_U,
        d_Y,
        d_nodeToElements,
        d_eNodMat,
        d_E,
        numNodes,
        numElements
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, numDOFs * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_U);
    cudaFree(d_Y);
    cudaFree(d_E);
    cudaFree(d_nodeToElements);
    cudaFree(d_eNodMat);
}