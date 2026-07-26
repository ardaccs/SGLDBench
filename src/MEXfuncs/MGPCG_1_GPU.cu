#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <vector>

// TODO. Add checks especially ones in the SGLDBench mex files and mgpcg9

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

// One Levelrepresents one of these:Level 0 → meshHierarchy_(1), Level 1 → meshHierarchy_(2)
struct Level
{
    int nx = 0;
    int ny = 0;
    int nz = 0;

    int numNodes = 0;
    int numElements = 0;
    int numDOFs = 0;

    size_t numGridNodes = 0;

    // MATLAB/CPU source pointers.
    const int32_t* h_nodeToElements = nullptr;
    const int32_t* h_eNodMat = nullptr;
    const int32_t* h_nodGridId = nullptr;
    const int32_t* h_nodMapForward = nullptr;

    const double* h_eleModulus = nullptr;
    const double* h_dK = nullptr;

    // CUDA device pointers.
    int32_t* d_nodeToElements = nullptr;
    int32_t* d_eNodMat = nullptr;
    int32_t* d_nodGridId = nullptr;
    int32_t* d_nodMapForward = nullptr;

    double* d_eleModulus = nullptr;
    double* d_dK = nullptr;

    double* d_rhs = nullptr;
    double* d_x = nullptr;
    double* d_residual = nullptr;
    double* d_temp = nullptr;
    double* d_rTilde = nullptr;
};

// This global object remains alive between MATLAB MEX calls and preserves data in the GPU.
struct SolverContext
{
    bool initialized = false;

    int numLevels = 0;
    size_t size = 0;
    double tolerance = 0.0;
    int maxIterations = 0;

    std::vector<Level> levels;
    std::vector<int> spanWidths;

    cublasHandle_t cublasHandle = nullptr;

    int numFixedDOFs = 0;
    int32_t* h_fixedDOFIds = nullptr;

    int numCoarseFreeDOFs = 0;
    int32_t* h_coarseFreeDOFIds = nullptr;

    double* h_Ke = nullptr;
    double* d_Ke = nullptr;

    int32_t* h_fixedDOFIds = nullptr;
    int32_t* h_coarseFreeDOFIds = nullptr;

    int32_t* d_fixedDOFIds = nullptr;
    int32_t* d_coarseFreeDOFIds = nullptr;

    void* d_workspace = nullptr;

    double* h_b = nullptr;
    double* d_b = nullptr;

    double* h_y = nullptr;
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
__global__ void dampedJacobiSmootherKernelFine(
    const double* __restrict__ r,
    const double* __restrict__ diagK,
    double * __restrict__ x,
    double* __restrict__ rTilde,
    const double weightFactorJacobi,
    const int numDOFs)
{
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't read or write out of bounds
    if (idx < numDOFs) {
        // rTilde = weightFactorJacobi * r ./ diagK
        x[idx] = weightFactorJacobi * (r[idx] / diagK[idx]);
    }
    rTilde = x;
}
__global__ void dampedJacobiSmootherKernelCoarse(
    const double* __restrict__ r,
    const double* __restrict__ diagK,
    double* __restrict__ x,
    const double weightFactorJacobi,
    const int numDOFs)
{
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't read or write out of bounds
    if (idx < numDOFs) {
        // rTilde = weightFactorJacobi * r ./ diagK
        x[idx] = weightFactorJacobi * (r[idx] / diagK[idx]);
    }
}
__global__ void addVectorsInPlaceKernel(
    double* __restrict__ a,
    const double* __restrict__ b,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        a[i] += b[i];
    }
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
//TODO---->Will this need to be updated? And contain later added fields? Or is it just for the static hierarchy arrays?
static size_t calculateRequiredGPUBytes(
    const SolverContext& solver)
{
    size_t bytes = 0;

    for (int levelIndex = 0;
         levelIndex < solver.numLevels;
         ++levelIndex)
    {
        const Level& level =
            solver.levels[levelIndex];

        // Static integer hierarchy arrays.
        bytes +=
            static_cast<size_t>(level.numNodes) *
            8 *
            sizeof(int32_t);

        bytes +=
            static_cast<size_t>(level.numElements) *
            8 *
            sizeof(int32_t);

        bytes +=
            static_cast<size_t>(level.numNodes) *
            sizeof(int32_t);

        bytes +=
            level.numGridNodes *
            sizeof(int32_t);

        // eleModulus exists on the finest level.
        if (levelIndex == 0)
        {
            bytes +=
                static_cast<size_t>(level.numElements) *
                sizeof(double);
        }

        // dK exists on every level except the coarsest,
        // based on your current intended layout.
        if (levelIndex < solver.numLevels - 1)
        {
            bytes +=
                static_cast<size_t>(level.numDOFs) *
                sizeof(double);
        }

        // rhs, x, residual and temp.
        bytes +=
            4 *
            static_cast<size_t>(level.numDOFs) *
            sizeof(double);
    }

    bytes +=
        static_cast<size_t>(solver.numFixedDOFs) *
        sizeof(int32_t);

    bytes +=
        static_cast<size_t>(
            solver.numCoarseFreeDOFs) *
        sizeof(int32_t);

    if (!solver.levels.empty())
    {
        const size_t finestDOFs =
            static_cast<size_t>(
                solver.levels[0].numDOFs);

        // b, y, r, z, p and Ap.
        bytes +=
            6 *
            finestDOFs *
            sizeof(double);
    }

    return bytes;
}
static void initializeGPUHierarchy(
    SolverContext& solver,
    const mxArray* hierarchyMx,
    const mxArray* bMx,
    const mxArray* yMx,
    double tolerance,
    int maxIterations)
{
    if (hierarchyMx == nullptr ||!mxIsStruct(hierarchyMx) || mxGetNumberOfElements(hierarchyMx) != 1)
    {
        mexErrMsgIdAndTxt("mgpcg_gpu:hierarchy","H must be a scalar MATLAB struct.");
    };

    // get the number of levels in the hierarchy
    mxArray *numLevelsField = mxGetField(hierarchyMx, 0, "numLevels");
    int32_t numLevels = (int32_t)mxGetScalar(numLevelsField);
    // Retrieve the resX, resY, resZ, numNodes, numElements, numDOFs, spanWidth, vecB, vecY fields from the hierarchy struct
    
    mxArray *resXField = mxGetField(hierarchyMx, 0, "resX");
    int32_t *resX = static_cast<int32_t*>(mxGetData(resXField));

    mxArray *resYField = mxGetField(hierarchyMx, 0, "resY");
    int32_t *resY = static_cast<int32_t*>(mxGetData(resYField));

    mxArray *resZField = mxGetField(hierarchyMx, 0, "resZ");
    int32_t *resZ = static_cast<int32_t*>(mxGetData(resZField));

    mxArray *numNodesField = mxGetField(hierarchyMx, 0, "numNodes");
    int32_t *numNodes = static_cast<int32_t*>(mxGetData(numNodesField));

    mxArray *numElementsField = mxGetField(hierarchyMx, 0, "numElements");
    int32_t *numElements = static_cast<int32_t*>(mxGetData(numElementsField));

    mxArray *numDOFsField = mxGetField(hierarchyMx, 0, "numDOFs");
    int32_t *numDOFs = static_cast<int32_t*>(mxGetData(numDOFsField));

    mxArray *spanWidthField = mxGetField(hierarchyMx, 0, "spanWidth");
    int32_t *spanWidth = static_cast<int32_t*>(mxGetData(spanWidthField));

    mxArray *vecBField = mxGetField(hierarchyMx, 0, "vecB");
    double *vecB = static_cast<double*>(mxGetData(vecBField));

    mxArray *vecYField = mxGetField(hierarchyMx, 0, "vecY");
    double *vecY = static_cast<double*>(mxGetData(vecYField));

    // Retrieve the nodeToElements, eNodMat, nodGridId, nodMapForward, eleModulus, diagK fields from the hierarchy struct
    mxArray *nodeToElementsField = mxGetField(hierarchyMx, 0, "nodeToElements");
    mxArray *eNodMatField = mxGetField(hierarchyMx, 0, "eNodMat");
    mxArray *nodGridIdField = mxGetField(hierarchyMx, 0, "nodGridId");
    mxArray *nodMapForwardField = mxGetField(hierarchyMx, 0, "nodMapForward");
    mxArray *eleModulusField = mxGetField(hierarchyMx, 0, "eleModulus");
    mxArray *diagKField = mxGetField(hierarchyMx, 0, "diagK");

    solver.initialized = false;
    solver.numLevels = numLevels;

    solver.levels.clear();
    solver.levels.resize(numLevels);

    solver.spanWidths.clear();

    solver.tolerance = tolerance;
    solver.maxIterations = maxIterations;

    solver.h_b = vecB;
    solver.h_y = vecY;

    if (numLevels > 1)
        solver.spanWidths.resize(numLevels - 1);

    size_t staticHierarchyBytes = 0;
    size_t multigridWorkspaceBytes = 0;
    size_t finestPCGWorkspaceBytes = 0;

    for (int level = 0; level < numLevels; ++level)
    {

        Level& hlevel = solver.levels[level];

        hlevel.nx = static_cast<int32_t>(resX[level]);
        hlevel.ny = static_cast<int32_t>(resY[level]);
        hlevel.nz = static_cast<int32_t>(resZ[level]);
        hlevel.numNodes = static_cast<int32_t>(numNodes[level]);
        hlevel.numElements = static_cast<int32_t>(numElements[level]);
        hlevel.numDOFs = static_cast<int32_t>(numDOFs[level]);
        hlevel.numGridNodes =
            static_cast<size_t>(hlevel.nx + 1) *
            static_cast<size_t>(hlevel.ny + 1) *
            static_cast<size_t>(hlevel.nz + 1);

        /*
         * Explicitly keep all GPU pointers null.
         */
        hlevel.h_nodeToElements = static_cast<const int32_t*>(mxGetData((mxGetCell(nodeToElementsField, level))));
        hlevel.h_eNodMat = static_cast<const int32_t*>(mxGetData((mxGetCell(eNodMatField, level))));
        hlevel.h_nodGridId = static_cast<const int32_t*>(mxGetData((mxGetCell(nodGridIdField, level))));
        hlevel.h_nodMapForward = static_cast<const int32_t*>(mxGetData((mxGetCell(nodMapForwardField, level))));

        if (level < numLevels - 1)
        {
            hlevel.h_dK = static_cast<const double*>(mxGetData((mxGetCell(diagKField, level))));
        }
        else{
            hlevel.h_dK = nullptr;
        }

        if (level == 0){
            hlevel.h_eleModulus = static_cast<const double*>(mxGetData((mxGetCell(eleModulusField, level))));
        }
        else{
            hlevel.h_eleModulus = nullptr;
        }

        // TODO. Check whether we are using the numFixed etc
        if(level == 0){
            mxArray* fixedDOFIdsMx = mxGetField(hierarchyMx, 0, "fixedDOFIds");
            solver.h_fixedDOFIds = static_cast<const int32_t*>(mxGetData(fixedDOFIdsMx));
            solver.numFixedDOFs = static_cast<int>(mxGetNumberOfElements(fixedDOFIdsMx));
            solver.h_Ke = static_cast<const double*>(mxGetData(mxGetField(hierarchyMx, 0, "Ke")));

        }else if(level == numLevels - 1){
            mxArray* coarseFreeDOFIdsMx = mxGetField(hierarchyMx, 0, "coarseFreeDOFIds");
            solver.h_coarseFreeDOFIds = static_cast<const int32_t*>(mxGetData(coarseFreeDOFIdsMx));
            solver.numCoarseFreeDOFs = static_cast<int>(mxGetNumberOfElements(coarseFreeDOFIdsMx));
        }
    
        // add spanwidth
        if(level < numLevels - 2){
            solver.spanWidths[level] = static_cast<int32_t>(spanWidth[level]);
        }

        hlevel.d_rhs = nullptr;
        hlevel.d_x = nullptr;
        hlevel.d_residual = nullptr;
        hlevel.d_temp = nullptr;
    }
    solver.size = calculateRequiredGPUBytes(solver);
}

/*
static void initializeGPU(
    SolverContext& solver)
{       
        // Initialze GPU
        cudaDeviceReset();
        cudaSetDevice(0);
        cudaFree(0);
        size_t f, t;
        cudaMemGetInfo(&f, &t);
        //TODO.ADD--->CUDA_CHECK(cudaMemcpyToSymbol( Ae, Ae0, sizeof(double) * 24*24) );
        // Get the overall size of the SolverContext
        double size = solver.size;
        // Report also the GPU Memory, check whether it fits
        if (size>f)
            MEX_PRINT("\nMGCG - Cuda device has not %d free memory, %d is required.\n", f, size);
            plhs[0] = mxCreateDoubleScalar(0.0);
            plhs[1] = mxCreateDoubleScalar(-1.0);
        }
        else {
        // Allocate Memory In The GPU
        double *devW;
        CUDA_CHECK(cudaMalloc( (void**)&devW, size) );
        // Map the pointers to the Allocated Memory
        for (int level = 0; level < solver.numLevels; ++level)
        {
            
        }      
}
        
*/

static void initializeGPU(
    SolverContext& solver)
{
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    size_t freeMemory;
    size_t totalMemory;

    CUDA_CHECK(cudaMemGetInfo(&freeMemory, &totalMemory));

    if (solver.size > freeMemory)
    {
        mexErrMsgIdAndTxt("mgpcg_gpu:memory", "\nMGCG - Cuda device does not have %zu free memory, %zu is required.\n", freeMemory, solver.size);
    }

    CUDA_CHECK(cudaMalloc(&solver.d_workspace, solver.size));

    // Casting void* to char* lets us move through the allocated memory one byte at a time.
    char* dW = static_cast<char*>(solver.d_workspace);

    size_t offset = 0;

    // Adding doubles first to ensure proper alignment for double arrays
    solver.levels[0].d_eleModulus= reinterpret_cast<double*>(dW + offset);
    offset += static_cast<size_t>(solver.levels[0].numElements) * sizeof(double);

    solver.levels[solver.numLevels - 1].d_dK = reinterpret_cast<double*>(dW + offset);
    offset += static_cast<size_t>(solver.levels[solver.numLevels - 1].numDOFs) * sizeof(double);

    const size_t finestDOFs = static_cast<size_t>(solver.levels[0].numDOFs);
    solver.d_b = reinterpret_cast<double*>(dW + offset);
    offset += finestDOFs * sizeof(double);

    solver.d_y = reinterpret_cast<double*>(dW + offset);
    offset += finestDOFs * sizeof(double);

    solver.d_r = reinterpret_cast<double*>(dW + offset);
    offset += finestDOFs * sizeof(double);

    solver.d_z = reinterpret_cast<double*>(dW + offset);
    offset += finestDOFs * sizeof(double);

    solver.d_p = reinterpret_cast<double*>(dW + offset);
    offset += finestDOFs * sizeof(double);

    solver.d_Ap = reinterpret_cast<double*>(dW + offset);
    offset += finestDOFs * sizeof(double);

    solver.levels[0].d_rTilde = reinterpret_cast<double*>(dW + offset);
    offset += static_cast<size_t>(solver.levels[0].numDOFs) * sizeof(double);

    for (int levelIndex = 0;levelIndex < solver.numLevels; ++levelIndex)
    {
        Level& level = solver.levels[levelIndex];
        level.d_rhs = reinterpret_cast<double*>(dW + offset);
        offset += static_cast<size_t>(level.numDOFs) * sizeof(double);

        level.d_x = reinterpret_cast<double*>(dW + offset);
        offset +=static_cast<size_t>(level.numDOFs) * sizeof(double);

        level.d_residual = reinterpret_cast<double*>(dW + offset);
        offset += static_cast<size_t>(level.numDOFs) * sizeof(double);

        level.d_temp = reinterpret_cast<double*>(dW + offset);
        offset += static_cast<size_t>(level.numDOFs) * sizeof(double);

    }

    for (int levelIndex = 0; levelIndex < solver.numLevels; ++levelIndex)
    {
        Level& level = solver.levels[levelIndex];

        level.d_nodeToElements = reinterpret_cast<int32_t*>(dW + offset);
        offset += static_cast<size_t>(level.numNodes) * 8 * sizeof(int32_t);

        level.d_eNodMat = reinterpret_cast<int32_t*>(dW + offset);
        offset += static_cast<size_t>(level.numElements) * 8 * sizeof(int32_t);

        level.d_nodGridId = reinterpret_cast<int32_t*>(dW + offset);
        offset += static_cast<size_t>(level.numNodes) * sizeof(int32_t);

        level.d_nodMapForward = reinterpret_cast<int32_t*>(dW + offset);
        offset += level.numGridNodes * sizeof(int32_t);
    }
    // Also add numdofS, jacobiOmega
    solver.d_fixedDOFIds = reinterpret_cast<int32_t*>(dW + offset);
    offset += static_cast<size_t>(solver.numFixedDOFs) * sizeof(int32_t);

    solver.d_coarseFreeDOFIds = reinterpret_cast<int32_t*>(dW + offset);
    offset += static_cast<size_t>(solver.numCoarseFreeDOFs) * sizeof(int32_t);


    // -------------------------------------------------------------------------
    // Copy static hierarchy data to the GPU.
    // -------------------------------------------------------------------------

    for (int levelIndex = 0; levelIndex < solver.numLevels; ++levelIndex)
    {
        Level& level = solver.levels[levelIndex];

        CUDA_CHECK(cudaMemcpy(
            level.d_nodeToElements,
            level.h_nodeToElements,
            static_cast<size_t>(level.numNodes) *
                8 * sizeof(int32_t),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            level.d_eNodMat,
            level.h_eNodMat,
            static_cast<size_t>(level.numElements) *
                8 * sizeof(int32_t),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            level.d_nodGridId,
            level.h_nodGridId,
            static_cast<size_t>(level.numNodes) *
                sizeof(int32_t),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            level.d_nodMapForward,
            level.h_nodMapForward,
            level.numGridNodes *
                sizeof(int32_t),
            cudaMemcpyHostToDevice));

        if (levelIndex == 0)
        {
            CUDA_CHECK(cudaMemcpy(
                level.d_eleModulus,
                level.h_eleModulus,
                static_cast<size_t>(level.numElements) *
                    sizeof(double),
                cudaMemcpyHostToDevice));
        }

        if (levelIndex < solver.numLevels - 1)
        {
            CUDA_CHECK(cudaMemcpy(
                level.d_dK,
                level.h_dK,
                static_cast<size_t>(level.numDOFs) *
                    sizeof(double),
                cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMemset(
            level.d_rhs,
            0,
            static_cast<size_t>(level.numDOFs) *
                sizeof(double)));

        CUDA_CHECK(cudaMemset(
            level.d_x,
            0,
            static_cast<size_t>(level.numDOFs) *
                sizeof(double)));

        CUDA_CHECK(cudaMemset(
            level.d_residual,
            0,
            static_cast<size_t>(level.numDOFs) *
                sizeof(double)));

        CUDA_CHECK(cudaMemset(
            level.d_temp,
            0,
            static_cast<size_t>(level.numDOFs) *
                sizeof(double)));
    }

    CUDA_CHECK(cudaMemcpy(
        solver.d_fixedDOFIds,
        solver.h_fixedDOFIds,
        static_cast<size_t>(solver.numFixedDOFs) *
            sizeof(int32_t),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        solver.d_coarseFreeDOFIds,
        solver.h_coarseFreeDOFIds,
        static_cast<size_t>(solver.numCoarseFreeDOFs) *
            sizeof(int32_t),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(
        solver.d_b,
        solver.h_b,
        finestDOFs * sizeof(double)));

    CUDA_CHECK(cudaMemset(
        solver.d_y,
        solver.h_y,
        finestDOFs * sizeof(double)));

    CUDA_CHECK(cudaMemset(
        solver.d_r,
        0,
        finestDOFs * sizeof(double)));

    CUDA_CHECK(cudaMemset(
        solver.d_z,
        0,
        finestDOFs * sizeof(double)));

    CUDA_CHECK(cudaMemset(
        solver.d_p,
        0,
        finestDOFs * sizeof(double)));

    CUDA_CHECK(cudaMemset(
        solver.d_Ap,
        0,
        finestDOFs * sizeof(double)));

    CUDA_CHECK(cudaMemcpyToSymbol(
        c_Ke,
        solver.h_Ke,
        24 * 24 * sizeof(double)));

    if (cublasCreate(&solver.cublasHandle) != CUBLAS_STATUS_SUCCESS)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:cublas",
            "Could not create the cuBLAS handle.");
    }

    solver.initialized = true;
}

void destroySolver(
    SolverContext& solver)
{
    if (solver.initialized)
    {
        CUDA_CHECK(cudaFree(solver.d_workspace));
        solver.d_workspace = nullptr;

        if (solver.cublasHandle != nullptr)
        {
            cublasDestroy(solver.cublasHandle);
            solver.cublasHandle = nullptr;
        }

        solver.initialized = false;
    }
}
/*
static void applyVcycle(
    SolverContext& solver,
    const double* d_fineResidual,
    double* d_fineCorrection)
{
    const int lastLevel = solver.numLevels - 1;

    constexpr int blockSize = 256;

    // Store the finest RHS in the already allocated level workspace.
    CUDA_CHECK(cudaMemcpy(solver.levels[0].d_rhs,d_fineResidual, static_cast<size_t>(solver.levels[0].numDOFs) * sizeof(double), cudaMemcpyDeviceToDevice));

    // Start every level correction from zero.
    // TODO: Add this to initializeGPU to avoid this extra kernel launch.
    for (int levelIndex = 0; levelIndex < solver.numLevels; ++levelIndex)
    {
        Level& level = solver.levels[levelIndex];
        CUDA_CHECK(cudaMemset(level.d_x, 0,static_cast<size_t>(level.numDOFs) * sizeof(double)));
    }

    // Fine -> coarse:
    // x_l = omega * r_l ./ diagK_l
    // r_(l+1) = R_l * r_l
    for (int levelIndex = 1;levelIndex <= lastLevel;++levelIndex)
    {
        Level& level = solver.levels[levelIndex];
        Level& fineLevel = solver.levels[levelIndex - 1];

        // TODO. Check thread sizes etc
        if (levelIndex == 1){
            dampedJacobiSmootherKernelFine<<<level.numDOFs, blockSize, blockSize>>>(
                fineLevel.d_rhs,
                fineLevel.d_dK,
                fineLevel.d_residual,
                fineLevel.d_rTilde,
                solver.jacobiOmega,
                fineLevel.numDOFs);
             
            restrictResidualKernel<<<level.numDOFs, blockSize, blockSize>>>(
                level.d_nodGridId,
                fineLevel.d_nodMapForward,
                fineLevel.d_residual,
                level.d_rhs,
                level.numNodes,
                level.nx,
                level.ny,
                level.nz,
                solver.spanWidths[levelIndex]);
        }else{
            dampedJacobiSmootherKernelCoarse<<<level.numDOFs, blockSize, blockSize>>>(
                level.d_rhs,
                level.d_dK,
                level.d_x,
                solver.jacobiOmega,
                level.numDOFs);
                
            restrictResidualKernel<<<level.numDOFs, blockSize, blockSize>>>(
                level.d_nodGridId,
                fineLevel.d_nodMapForward,
                fineLevel.d_residual,
                level.d_rhs,
                level.numNodes,
                level.nx,
                level.ny,
                level.nz,
                solver.spanWidths[levelIndex]);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Implement this with cuDDS
    solveCoarsest(
        solver,
        solver.levels[lastLevel].d_rhs,
        solver.levels[lastLevel].d_x);

    // Coarse -> fine:
    // x_l += P_l * x_(l+1)
    // x_l += omega * r_l ./ diagK_l
    for (int levelIndex = lastLevel - 1; levelIndex >= 0; --levelIndex)
    {
        Level& level = solver.levels[levelIndex];
        Level& finelevel = solver.levels[levelIndex - 1];

        if(fineLevelIndex == 1){
            double* d_rTilde = finelevel.d_rTilde;
            interpolateResidualKernel<<<level.numDOFs, blockSize, blockSize>>>(
                level.d_nodGridId,
                level.d_nodMapForward,
                level.d_x,
                level.d_residual,
                level.numNodes,
                level.nx,
                level.ny,
                level.nz,
                solver.spanWidths[fineLevelIndex]);
            // Element wise addition of d_residual and d_rTilde
            addVectorsInPlaceKernel<<<level.numDOFs, blockSize, blockSize>>>(
                level.d_residual,
                d_rTilde,
                level.numDOFs);
            
            // Apply the damped Jacobi smoother on the fine level
            dampedJacobiSmootherKernelFine<<<level.numDOFs, blockSize, blockSize>>>(
                fineLevel.d_rhs,
                fineLevel.d_dK,
                fineLevel.d_residual,
                fineLevel.d_rTilde,
                solver.jacobiOmega,
                fineLevel.numDOFs);
            double rTilde2 = finelevel.d_rTilde;
            // Element wise addition of d_rTilde and d_rTilde2
            addVectorsInPlaceKernel<<<level.numDOFs, blockSize, blockSize>>>(
                rTilde2,
                d_rTilde,
                level.numDOFs);
        }else{
            interpolateResidualKernel<<<level.numDOFs, blockSize, blockSize>>>(
                finelevel.d_nodGridId,
                level.d_nodMapForward,
                level.d_x,
                finelevel.d_x,
                finelevel.numNodes,
                level.nx,
                level.ny,
                level.nz,
                solver.spanWidths[fineLevelIndex]);
            addVectorsInPlaceKernel<<<level.numDOFs, blockSize, blockSize>>>(
                level.d_x,
                finelevel.d_x,
                finelevel.numDOFs);
            double xTemp = finelevel.d_temp;
            // Apply the damped Jacobi smoother on the fine level
            dampedJacobiSmootherKernelCoarse<<<level.numDOFs, blockSize, blockSize>>>(
                fineLevel.d_rhs,
                fineLevel.d_dK,
                fineLevel.d_x,
                solver.jacobiOmega,
                fineLevel.numDOFs);
            addVectorsInPlaceKernel<<<level.numDOFs, blockSize, blockSize>>>(
                xTemp,
                finelevel.d_x,
                finelevel.numDOFs);
        }

    }

    // Zero out the fixed DOFs in the correction vector.
    zeroSelectedDOFsKernel<<<(solver.numFixedDOFs + blockSize - 1) / blockSize, blockSize>>>(
        solver.levels[0].d_x,
        solver.d_fixedDOFIds,
        solver.numFixedDOFs);

    CUDA_CHECK(cudaMemcpy(
        d_fineCorrection,
        solver.levels[0].d_x,
        static_cast<size_t>(solver.levels[0].numDOFs) *
            sizeof(double),
        cudaMemcpyDeviceToDevice));
}
        */
static void applyVcycle(
    SolverContext& solver,
    const double* d_fineResidual,
    double* d_fineCorrection)
{
    constexpr int blockSize = 256;

    const int lastLevel = solver.numLevels - 1;

    Level& finest = solver.levels[0];

    // MATLAB input r.
    CUDA_CHECK(cudaMemcpy(finest.d_rhs, d_fineResidual, static_cast<size_t>(finest.numDOFs) * sizeof(double), cudaMemcpyDeviceToDevice));

    // Reset x and temporary vectors.
    for (Level& level : solver.levels)
    {
        CUDA_CHECK(cudaMemset(level.d_x,0,static_cast<size_t>(level.numDOFs) * sizeof(double)));
        CUDA_CHECK(cudaMemset(level.d_temp,0,static_cast<size_t>(level.numDOFs) * sizeof(double)));
    }

    /*
     * Fine -> coarse.
     */
    for (int fineIndex = 0; fineIndex < lastLevel; ++fineIndex)
    {
        Level& fine = solver.levels[fineIndex];

        Level& coarse = solver.levels[fineIndex + 1];

        const int fineDOFBlocks = (fine.numDOFs + blockSize - 1) / blockSize;

        dampedJacobiSmootherKernelCoarse
            <<<fineDOFBlocks, blockSize>>>(
                fine.d_rhs,
                fine.d_dK,
                fine.d_x,
                solver.jacobiOmega,
                fine.numDOFs);

        CUDA_CHECK(cudaGetLastError());

        const int coarseNodeBlocks = (coarse.numNodes + blockSize - 1) / blockSize;

        restrictResidualKernel
            <<<coarseNodeBlocks, blockSize>>>(
                coarse.d_nodGridId,
                fine.d_nodMapForward,
                fine.d_rhs,
                coarse.d_rhs,
                coarse.numNodes,
                coarse.nx,
                coarse.ny,
                coarse.nz,
                fine.nx,
                fine.ny,
                fine.nz,
                solver.spanWidths[fineIndex]);

        CUDA_CHECK(cudaGetLastError());
    }

    /*
     * Direct coarse solve.
     */
    solveCoarsest(
        solver,
        solver.levels[lastLevel].d_rhs,
        solver.levels[lastLevel].d_x);

    /*
     * Coarse -> fine.
     */
    for (int fineIndex = lastLevel - 1; fineIndex >= 0; --fineIndex)
    {
        Level& fine = solver.levels[fineIndex];

        Level& coarse = solver.levels[fineIndex + 1];

        const int fineNodeBlocks = (fine.numNodes + blockSize - 1) / blockSize;

        interpolateResidualKernel
            <<<fineNodeBlocks, blockSize>>>(
                fine.d_nodGridId,
                coarse.d_nodMapForward,
                coarse.d_x,
                fine.d_temp,
                fine.numNodes,
                coarse.nx,
                coarse.ny,
                coarse.nz,
                fine.nx,
                fine.ny,
                fine.nz,
                solver.spanWidths[fineIndex]);

        CUDA_CHECK(cudaGetLastError());

        const int fineDOFBlocks = (fine.numDOFs + blockSize - 1) / blockSize;

        addVectorsInPlaceKernel
            <<<fineDOFBlocks, blockSize>>>(
                fine.d_x,
                fine.d_temp,
                fine.numDOFs);

        CUDA_CHECK(cudaGetLastError());

        // Reuse temp for the post-smoothing term.
        dampedJacobiSmootherKernelCoarse
            <<<fineDOFBlocks, blockSize>>>(
                fine.d_rhs,
                fine.d_dK,
                fine.d_temp,
                solver.jacobiOmega,
                fine.numDOFs);

        CUDA_CHECK(cudaGetLastError());

        addVectorsInPlaceKernel
            <<<fineDOFBlocks, blockSize>>>(
                fine.d_x,
                fine.d_temp,
                fine.numDOFs);

        CUDA_CHECK(cudaGetLastError());
    }

    zeroFixedDOFs(solver, finest.d_x);

    CUDA_CHECK(cudaMemcpy(d_fineCorrection, finest.d_x, static_cast<size_t>(finest.numDOFs) * sizeof(double), cudaMemcpyDeviceToDevice));
}
void runMGPCG(
    SolverContext* solver)
{

    // Everthing is already on the GPU, so to this function implement the MGPCG algorithm on the GPU.

    // Initialize the residual r = b - A*y
    



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
            "Wrong Number of Arguments for Solving_MGPCG_GPU. It should be("
            "b, tolerance, maxIterations, y0, H)");
    }

    if (nlhs < 1 || nlhs > 3) {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:nlhs",
            "The function supports one to three outputs.");
    }

    const mxArray* bMx = prhs[0];
    const double tolerance = mxGetScalar(prhs[1]);
    const int maxIterations = static_cast<int>(mxGetScalar(prhs[2]));
    const mxArray* y0Mx = prhs[3];
    const mxArray* hierarchyMx = prhs[4];

    SolverContext solver;

    try {
        // Unpack hierarchyMx to SolverContext
        initializeGPUHierarchy(
            &solver,
            hierarchyMx,
            bMx,
            y0Mx,
            tolerance,
            maxIterations
        );
        // Get the overall size of the SolverContext
        // Report also the GPU Memory, check whether it fits
        // Allocate Memory In The GPU
        // Map the pointers to the Allocated Memory
        
        // 
        initializeGPU(
            &solver, 
        );

        runMGPCG(
            &solver);

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