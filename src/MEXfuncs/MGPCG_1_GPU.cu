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

// One LevelGPU represents one of these:LevelGPU 0 → meshHierarchy_(1), LevelGPU 1 → meshHierarchy_(2)
struct LevelGPU
{
    int nx = 0;
    int ny = 0;
    int nz = 0;

    int numNodes = 0;
    int numElements = 0;
    int numDOFs = 0;

    size_t numGridNodes = 0;

    /*
     * Static hierarchy data.
     */
    int32_t* d_nodeToElements = nullptr;
    int32_t* d_eNodMat = nullptr;
    int32_t* d_nodGridId = nullptr;
    int32_t* d_nodMapForward = nullptr;

    /*
     * Material values.
     */
    double* d_eleModulus = nullptr;

    /*
     * Multigrid working vectors.
     */
    double* d_rhs = nullptr;
    double* d_x = nullptr;
    double* d_residual = nullptr;
    double* d_temp = nullptr;
    double* d_dK = nullptr;
};

// This global object remains alive between MATLAB MEX calls and preserves data in the GPU.
// mgpcg_gpu_mex('init', H); creates the GPU data and later mgpcg_gpu_mex('solve', ...); uses the same data.
struct SolverContext
{
    bool initialized = false;

    int numLevels = 0;

    std::vector<LevelGPU> levels;
    std::vector<int> spanWidths;

    cublasHandle_t cublasHandle = nullptr;

    /*
     * Finest-level PCG vectors.
     */
    double* d_b = nullptr;
    double* d_y = nullptr;
    double* d_r = nullptr;
    double* d_z = nullptr;
    double* d_p = nullptr;
    double* d_Ap = nullptr;
};

__global__ void zeroSelectedDOFsKernel(
    double* vector,
    const int32_t* fixedDOFIds,
    int numFixedDOFs)
{
    const int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numFixedDOFs)
        return;

    const int dof =
        fixedDOFIds[index] - 1;

    vector[dof] = 0.0;
}

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
static void zeroFixedDOFs(
    SolverContext& solver,
    double* d_vector)
{
    if (solver.numFixedDOFs == 0)
        return;

    constexpr int blockSize = 256;

    const int gridSize =
        (solver.numFixedDOFs + blockSize - 1)
        / blockSize;

    zeroSelectedDOFsKernel<<<gridSize, blockSize>>>(
        d_vector,
        solver.d_fixedDOFIds,
        solver.numFixedDOFs);

    CUDA_CHECK(cudaGetLastError());
}
void mexFunction(
    int nlhs,
    mxArray* plhs[],
    int nrhs,
    const mxArray* prhs[])
{
    if (nrhs != 5) {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:nrhs",
            "Usage: [y,it,res] = Solving_MGPCG_GPU("
            "b, tolerance, maxIterations, y0, H)");
    }

    if (nlhs < 1 || nlhs > 3) {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:nlhs",
            "The function supports one to three outputs.");
    }

    const mxArray* bMx =
        prhs[0];

    const double tolerance =
        mxGetScalar(prhs[1]);

    const int maxIterations =
        static_cast<int>(
            mxGetScalar(prhs[2]));

    const mxArray* y0Mx =
        prhs[3];

    const mxArray* hierarchyMx =
        prhs[4];

    SolverContext solver;

    try {
        initializeGPUHierarchy(
            solver,
            hierarchyMx);

        allocateSolverWorkspace(
            solver);

        runMGPCG(
            solver,
            bMx,
            y0Mx,
            tolerance,
            maxIterations);

        createMATLABOutputs(
            solver,
            nlhs,
            plhs);

        destroySolver(
            solver);
    }
    catch (...) {
        /*
         * Make sure GPU memory is also released
         * when something fails.
         */
        destroySolver(
            solver);

        throw;
    }
}