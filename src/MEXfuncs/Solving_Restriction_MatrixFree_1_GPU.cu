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

__global__ void restrictResidualKernel(
    const int* __restrict__ coarseNodeGridId,
    const int* __restrict__ fineNodeMapForward,
    const double* __restrict__ fineResidual,
    double* __restrict__ coarseResidual,
    int numCoarseNodes,
    int coarseNx,
    int coarseNy,
    int coarseNz,
    int fineNx,
    int fineNy,
    int fineNz,
    int spanWidth)
{
    // Gather per active coarse node
    int coarseNode =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (coarseNode >= numCoarseNodes) {
        return;
    }

    int coarseNyNodes = coarseNy + 1;
    int coarseNxNodes = coarseNx + 1;

    int fineNyNodes = fineNy + 1;
    int fineNxNodes = fineNx + 1;
    int fineNzNodes = fineNz + 1;

    int coarseGridId = coarseNodeGridId[coarseNode] - 1;

    int coarseY =
        coarseGridId % coarseNyNodes;

    int coarseX =
        (coarseGridId / coarseNyNodes)
        % coarseNxNodes;

    int coarseZ =
        coarseGridId /
        (coarseNyNodes * coarseNxNodes);

    int fineCenterY = spanWidth * coarseY;
    int fineCenterX = spanWidth * coarseX;
    int fineCenterZ = spanWidth * coarseZ;

    int radius = spanWidth - 1;

    double resultX = 0.0;
    double resultY = 0.0;
    double resultZ = 0.0;

    #pragma unroll
    for (int dz = -radius; dz <= radius; ++dz)
    {
        int fineZ = fineCenterZ + dz;

        if (fineZ < 0 || fineZ >= fineNzNodes)
            continue;

        double wz = 1.0 - double(abs(dz)) / double(spanWidth);
        #pragma unroll
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int fineX = fineCenterX + dx;

            if (fineX < 0 || fineX >= fineNxNodes)
                continue;

            double wx =
                1.0 - double(abs(dx)) / double(spanWidth);
            #pragma unroll
            for (int dy = -radius; dy <= radius; ++dy)
            {
                int fineY = fineCenterY + dy;

                if (fineY < 0 || fineY >= fineNyNodes)
                    continue;

                double wy =
                    1.0 - double(abs(dy)) / double(spanWidth);

                int fineGridId =
                    fineY
                    + fineX * fineNyNodes
                    + fineZ * fineNyNodes * fineNxNodes;

                int activeFineNode = fineNodeMapForward[fineGridId];

                if (activeFineNode == 0)
                    continue;

                int fineNode = activeFineNode - 1;

                double weight = wx * wy * wz;

                int input = 3 * fineNode;

                resultX +=
                    weight * fineResidual[input + 0];

                resultY +=
                    weight * fineResidual[input + 1];

                resultZ +=
                    weight * fineResidual[input + 2];
            }
        }
    }

    int output = 3 * coarseNode;

    coarseResidual[output + 0] = resultX;
    coarseResidual[output + 1] = resultY;
    coarseResidual[output + 2] = resultZ;

}

void mexFunction(
    int nlhs,
    mxArray* plhs[],
    int nrhs,
    const mxArray* prhs[])
{
    /*
     * Inputs:
     *
     *  0: coarseNodeGridId   [numCoarseNodes x 1], int32
     *  1: fineNodeMapForward [fullFineGridNodes x 1], int32
     *  2: fineResidual       [3*numFineNodes x 1], double
     *  3: coarseNx
     *  4: coarseNy
     *  5: coarseNz
     *  6: fineNx
     *  7: fineNy
     *  8: fineNz
     *  9: spanWidth
     *
     * Output:
     *
     *  coarseResidual [3*numCoarseNodes x 1], double
     */

    if (nrhs != 10) {
        mexErrMsgIdAndTxt(
            "restrict_residual_gpu:nrhs",
            "Need 10 inputs.");
    }

    if (nlhs > 1) {
        mexErrMsgIdAndTxt(
            "restrict_residual_gpu:nlhs",
            "Only one output is supported.");
    }

    const mxArray* coarseNodeGridId_mx   = prhs[0];
    const mxArray* fineNodeMapForward_mx = prhs[1];
    const mxArray* fineResidual_mx       = prhs[2];

    int coarseNx = static_cast<int>(mxGetScalar(prhs[3]));
    int coarseNy = static_cast<int>(mxGetScalar(prhs[4]));
    int coarseNz = static_cast<int>(mxGetScalar(prhs[5]));

    int fineNx = static_cast<int>(mxGetScalar(prhs[6]));
    int fineNy = static_cast<int>(mxGetScalar(prhs[7]));
    int fineNz = static_cast<int>(mxGetScalar(prhs[8]));

    int spanWidth = static_cast<int>(mxGetScalar(prhs[9]));

    if (!mxIsInt32(coarseNodeGridId_mx)) {
        mexErrMsgIdAndTxt(
            "restrict_residual_gpu:type",
            "coarseNodeGridId must be int32.");
    }

    if (!mxIsInt32(fineNodeMapForward_mx)) {
        mexErrMsgIdAndTxt(
            "restrict_residual_gpu:type",
            "fineNodeMapForward must be int32.");
    }

    if (!mxIsDouble(fineResidual_mx) ||
        mxIsComplex(fineResidual_mx)) {
        mexErrMsgIdAndTxt(
            "restrict_residual_gpu:type",
            "fineResidual must be a real double array.");
    }

    if (spanWidth <= 0) {
        mexErrMsgIdAndTxt(
            "restrict_residual_gpu:spanWidth",
            "spanWidth must be positive.");
    }

    static_assert(
        sizeof(int) == sizeof(int32_t),
        "This MEX function requires a 32-bit int type.");

    int numCoarseNodes =
        static_cast<int>(
            mxGetNumberOfElements(coarseNodeGridId_mx));

    size_t numFineGridNodes =
        mxGetNumberOfElements(fineNodeMapForward_mx);

    size_t numFineDOFs =
        mxGetNumberOfElements(fineResidual_mx);

    size_t expectedFineGridNodes =
        static_cast<size_t>(fineNy + 1)
        * static_cast<size_t>(fineNx + 1)
        * static_cast<size_t>(fineNz + 1);

    if (numFineGridNodes < expectedFineGridNodes) {
        mexErrMsgIdAndTxt(
            "restrict_residual_gpu:fineMapSize",
            "fineNodeMapForward contains fewer entries than the full fine grid.");
    }

    if (numFineDOFs % 3 != 0) {
        mexErrMsgIdAndTxt(
            "restrict_residual_gpu:fineResidualSize",
            "The number of fineResidual entries must be divisible by 3.");
    }

    const int* h_coarseNodeGridId =
        static_cast<const int*>(
            mxGetData(coarseNodeGridId_mx));

    const int* h_fineNodeMapForward =
        static_cast<const int*>(
            mxGetData(fineNodeMapForward_mx));

    const double* h_fineResidual =
        mxGetDoubles(fineResidual_mx);

    size_t numCoarseDOFs =
        static_cast<size_t>(3)
        * static_cast<size_t>(numCoarseNodes);

    plhs[0] = mxCreateDoubleMatrix(
        numCoarseDOFs,
        1,
        mxREAL);

    double* h_coarseResidual =
        mxGetDoubles(plhs[0]);

    // Nothing needs to be launched for an empty coarse grid.
    if (numCoarseNodes == 0) {
        return;
    }

    int* d_coarseNodeGridId = nullptr;
    int* d_fineNodeMapForward = nullptr;

    double* d_fineResidual = nullptr;
    double* d_coarseResidual = nullptr;

    size_t coarseNodeGridIdBytes =
        static_cast<size_t>(numCoarseNodes) * sizeof(int);

    size_t fineNodeMapForwardBytes =
        numFineGridNodes * sizeof(int);

    size_t fineResidualBytes =
        numFineDOFs * sizeof(double);

    size_t coarseResidualBytes =
        numCoarseDOFs * sizeof(double);

    CUDA_CHECK(cudaMalloc(
        &d_coarseNodeGridId,
        coarseNodeGridIdBytes));

    CUDA_CHECK(cudaMalloc(
        &d_fineNodeMapForward,
        fineNodeMapForwardBytes));

    CUDA_CHECK(cudaMalloc(
        &d_fineResidual,
        fineResidualBytes));

    CUDA_CHECK(cudaMalloc(
        &d_coarseResidual,
        coarseResidualBytes));

    CUDA_CHECK(cudaMemcpy(
        d_coarseNodeGridId,
        h_coarseNodeGridId,
        coarseNodeGridIdBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_fineNodeMapForward,
        h_fineNodeMapForward,
        fineNodeMapForwardBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_fineResidual,
        h_fineResidual,
        fineResidualBytes,
        cudaMemcpyHostToDevice));

    int block = 256;

    int grid =
        (numCoarseNodes + block - 1) / block;

    restrictResidualKernel<<<grid, block>>>(
        d_coarseNodeGridId,
        d_fineNodeMapForward,
        d_fineResidual,
        d_coarseResidual,
        numCoarseNodes,
        coarseNx,
        coarseNy,
        coarseNz,
        fineNx,
        fineNy,
        fineNz,
        spanWidth);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_coarseResidual,
        d_coarseResidual,
        coarseResidualBytes,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_coarseNodeGridId));
    CUDA_CHECK(cudaFree(d_fineNodeMapForward));
    CUDA_CHECK(cudaFree(d_fineResidual));
    CUDA_CHECK(cudaFree(d_coarseResidual));
}