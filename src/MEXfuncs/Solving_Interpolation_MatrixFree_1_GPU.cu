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

__global__ void interpolateResidualKernel(
    const int* __restrict__ fineNodeGridId,
    const int* __restrict__ coarseNodeMapForward,
    const double* __restrict__ coarseResidual,
    double* __restrict__ fineResidual,
    int numFineNodes,
    int coarseNx,
    int coarseNy,
    int coarseNz,
    int fineNx,
    int fineNy,
    int fineNz,
    int spanWidth)
{
    // One thread per active fine node
    int fineNode =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (fineNode >= numFineNodes) {
        return;
    }

    const int fineNyNodes = fineNy + 1;
    const int fineNxNodes = fineNx + 1;
    const int fineNzNodes = fineNz + 1;

    const int coarseNyNodes = coarseNy + 1;
    const int coarseNxNodes = coarseNx + 1;
    const int coarseNzNodes = coarseNz + 1;

    /*
     * fineNodeGridId:
     *
     * active fine node index -> global fine-grid index
     *
     * MATLAB indices are assumed to be 1-based.
     */
    const int fineGridId =
        fineNodeGridId[fineNode] - 1;

    /*
     * MATLAB/column-major node ordering:
     *
     * gridId = y
     *        + x * numYNodes
     *        + z * numYNodes * numXNodes
     */
    const int fineY =
        fineGridId % fineNyNodes;

    const int fineX =
        (fineGridId / fineNyNodes)
        % fineNxNodes;

    const int fineZ =
        fineGridId /
        (fineNyNodes * fineNxNodes);

    /*
     * Lower coarse-grid node surrounding this fine node.
     */
    const int coarseBaseY = fineY / spanWidth;
    const int coarseBaseX = fineX / spanWidth;
    const int coarseBaseZ = fineZ / spanWidth;

    /*
     * Fine-node offset inside the coarse-grid cell.
     *
     * Example for spanWidth = 2:
     *
     * fine coordinate: 0 1 2
     * coarse nodes:    0   1
     *
     * offset:
     *   fine=0 -> 0
     *   fine=1 -> 1
     *   fine=2 -> 0 in the next coarse cell
     */
    const int offsetY =
        fineY - coarseBaseY * spanWidth;

    const int offsetX =
        fineX - coarseBaseX * spanWidth;

    const int offsetZ =
        fineZ - coarseBaseZ * spanWidth;

    const double invSpan =
        1.0 / static_cast<double>(spanWidth);

    /*
     * Interpolation coordinates within the coarse cell.
     */
    const double ty =
        static_cast<double>(offsetY) * invSpan;

    const double tx =
        static_cast<double>(offsetX) * invSpan;

    const double tz =
        static_cast<double>(offsetZ) * invSpan;

    /*
     * Candidate coarse coordinates and corresponding 1D weights.
     *
     * A fine node can interpolate from:
     *
     * base coarse node     weight = 1 - t
     * base + 1 coarse node weight = t
     */
    const int coarseYs[2] = {
        coarseBaseY,
        coarseBaseY + 1
    };

    const int coarseXs[2] = {
        coarseBaseX,
        coarseBaseX + 1
    };

    const int coarseZs[2] = {
        coarseBaseZ,
        coarseBaseZ + 1
    };

    const double wy[2] = {
        1.0 - ty,
        ty
    };

    const double wx[2] = {
        1.0 - tx,
        tx
    };

    const double wz[2] = {
        1.0 - tz,
        tz
    };

    double resultX = 0.0;
    double resultY = 0.0;
    double resultZ = 0.0;

    #pragma unroll
    for (int iz = 0; iz < 2; ++iz)
    {
        const int coarseZ = coarseZs[iz];

        if (coarseZ < 0 || coarseZ >= coarseNzNodes)
            continue;

        if (wz[iz] == 0.0)
            continue;

        #pragma unroll
        for (int ix = 0; ix < 2; ++ix)
        {
            const int coarseX = coarseXs[ix];

            if (coarseX < 0 || coarseX >= coarseNxNodes)
                continue;

            if (wx[ix] == 0.0)
                continue;

            #pragma unroll
            for (int iy = 0; iy < 2; ++iy)
            {
                const int coarseY = coarseYs[iy];

                if (coarseY < 0 || coarseY >= coarseNyNodes)
                    continue;

                if (wy[iy] == 0.0)
                    continue;

                const int coarseGridId =
                    coarseY
                    + coarseX * coarseNyNodes
                    + coarseZ *
                        coarseNyNodes *
                        coarseNxNodes;

                /*
                 * coarseNodeMapForward:
                 *
                 * global coarse-grid index -> active coarse node index
                 *
                 * 0 means inactive.
                 * Positive values are MATLAB-style 1-based indices.
                 */
                const int activeCoarseNode =
                    coarseNodeMapForward[coarseGridId];

                if (activeCoarseNode == 0)
                    continue;

                const int coarseNode =
                    activeCoarseNode - 1;

                const double weight =
                    wx[ix] * wy[iy] * wz[iz];

                const int input =
                    3 * coarseNode;

                resultX +=
                    weight * coarseResidual[input + 0];

                resultY +=
                    weight * coarseResidual[input + 1];

                resultZ +=
                    weight * coarseResidual[input + 2];
            }
        }
    }

    const int output = 3 * fineNode;

    fineResidual[output + 0] = resultX;
    fineResidual[output + 1] = resultY;
    fineResidual[output + 2] = resultZ;
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
     *  0: fineNodeGridId       [numFineNodes x 1], int32
     *  1: coarseNodeMapForward [fullCoarseGridNodes x 1], int32
     *  2: coarseResidual       [3*numCoarseNodes x 1], double
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
     *  fineResidual [3*numFineNodes x 1], double
     */

    if (nrhs != 10) {
        mexErrMsgIdAndTxt(
            "interpolate_residual_gpu:nrhs",
            "Need 10 inputs.");
    }

    if (nlhs > 1) {
        mexErrMsgIdAndTxt(
            "interpolate_residual_gpu:nlhs",
            "Only one output is supported.");
    }

    const mxArray* fineNodeGridId_mx =
        prhs[0];

    const mxArray* coarseNodeMapForward_mx =
        prhs[1];

    const mxArray* coarseResidual_mx =
        prhs[2];

    int coarseNx =
        static_cast<int>(mxGetScalar(prhs[3]));

    int coarseNy =
        static_cast<int>(mxGetScalar(prhs[4]));

    int coarseNz =
        static_cast<int>(mxGetScalar(prhs[5]));

    int fineNx =
        static_cast<int>(mxGetScalar(prhs[6]));

    int fineNy =
        static_cast<int>(mxGetScalar(prhs[7]));

    int fineNz =
        static_cast<int>(mxGetScalar(prhs[8]));

    int spanWidth =
        static_cast<int>(mxGetScalar(prhs[9]));

    if (!mxIsInt32(fineNodeGridId_mx)) {
        mexErrMsgIdAndTxt(
            "interpolate_residual_gpu:type",
            "fineNodeGridId must be int32.");
    }

    if (!mxIsInt32(coarseNodeMapForward_mx)) {
        mexErrMsgIdAndTxt(
            "interpolate_residual_gpu:type",
            "coarseNodeMapForward must be int32.");
    }

    if (!mxIsDouble(coarseResidual_mx) ||
        mxIsComplex(coarseResidual_mx)) {
        mexErrMsgIdAndTxt(
            "interpolate_residual_gpu:type",
            "coarseResidual must be a real double array.");
    }

    if (spanWidth <= 0) {
        mexErrMsgIdAndTxt(
            "interpolate_residual_gpu:spanWidth",
            "spanWidth must be positive.");
    }

    static_assert(
        sizeof(int) == sizeof(int32_t),
        "This MEX function requires a 32-bit int type.");

    size_t numFineNodesSize =
        mxGetNumberOfElements(fineNodeGridId_mx);

    if (numFineNodesSize >
        static_cast<size_t>(INT_MAX)) {
        mexErrMsgIdAndTxt(
            "interpolate_residual_gpu:size",
            "The number of active fine nodes exceeds INT_MAX.");
    }

    int numFineNodes =
        static_cast<int>(numFineNodesSize);

    size_t numCoarseGridNodes =
        mxGetNumberOfElements(
            coarseNodeMapForward_mx);

    size_t numCoarseDOFs =
        mxGetNumberOfElements(
            coarseResidual_mx);

    if (numCoarseDOFs % 3 != 0) {
        mexErrMsgIdAndTxt(
            "interpolate_residual_gpu:coarseResidualSize",
            "The number of coarseResidual entries must be divisible by 3.");
    }

    size_t expectedCoarseGridNodes =
        static_cast<size_t>(coarseNy + 1)
        * static_cast<size_t>(coarseNx + 1)
        * static_cast<size_t>(coarseNz + 1);

    if (numCoarseGridNodes <
        expectedCoarseGridNodes) {
        mexErrMsgIdAndTxt(
            "interpolate_residual_gpu:coarseMapSize",
            "coarseNodeMapForward contains fewer entries than the full coarse grid.");
    }

    const int* h_fineNodeGridId =
        static_cast<const int*>(
            mxGetData(fineNodeGridId_mx));

    const int* h_coarseNodeMapForward =
        static_cast<const int*>(
            mxGetData(coarseNodeMapForward_mx));

    const double* h_coarseResidual =
        mxGetDoubles(coarseResidual_mx);

    size_t numFineDOFs =
        static_cast<size_t>(3)
        * static_cast<size_t>(numFineNodes);

    plhs[0] = mxCreateDoubleMatrix(
        numFineDOFs,
        1,
        mxREAL);

    double* h_fineResidual =
        mxGetDoubles(plhs[0]);

    if (numFineNodes == 0) {
        return;
    }

    int* d_fineNodeGridId = nullptr;
    int* d_coarseNodeMapForward = nullptr;

    double* d_coarseResidual = nullptr;
    double* d_fineResidual = nullptr;

    size_t fineNodeGridIdBytes =
        static_cast<size_t>(numFineNodes)
        * sizeof(int);

    size_t coarseNodeMapForwardBytes =
        numCoarseGridNodes
        * sizeof(int);

    size_t coarseResidualBytes =
        numCoarseDOFs
        * sizeof(double);

    size_t fineResidualBytes =
        numFineDOFs
        * sizeof(double);

    CUDA_CHECK(cudaMalloc(
        &d_fineNodeGridId,
        fineNodeGridIdBytes));

    CUDA_CHECK(cudaMalloc(
        &d_coarseNodeMapForward,
        coarseNodeMapForwardBytes));

    CUDA_CHECK(cudaMalloc(
        &d_coarseResidual,
        coarseResidualBytes));

    CUDA_CHECK(cudaMalloc(
        &d_fineResidual,
        fineResidualBytes));

    CUDA_CHECK(cudaMemcpy(
        d_fineNodeGridId,
        h_fineNodeGridId,
        fineNodeGridIdBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_coarseNodeMapForward,
        h_coarseNodeMapForward,
        coarseNodeMapForwardBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_coarseResidual,
        h_coarseResidual,
        coarseResidualBytes,
        cudaMemcpyHostToDevice));

    int block = 256;

    int grid =
        (numFineNodes + block - 1) / block;

    interpolateResidualKernel<<<grid, block>>>(
        d_fineNodeGridId,
        d_coarseNodeMapForward,
        d_coarseResidual,
        d_fineResidual,
        numFineNodes,
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
        h_fineResidual,
        d_fineResidual,
        fineResidualBytes,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_fineNodeGridId));
    CUDA_CHECK(cudaFree(d_coarseNodeMapForward));
    CUDA_CHECK(cudaFree(d_coarseResidual));
    CUDA_CHECK(cudaFree(d_fineResidual));
}