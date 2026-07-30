#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudss.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>

// Elementary stiffness matrix (24 x 24 doubles = 4608 bytes).
__constant__ double c_Ke[24 * 24];

static std::runtime_error makeCudaError(
    cudaError_t status,
    const char* expression,
    const char* file,
    int line)
{
    char message[1024];
    std::snprintf(
        message,
        sizeof(message),
        "CUDA call failed at %s:%d: %s -> %s",
        file,
        line,
        expression,
        cudaGetErrorString(status));
    return std::runtime_error(message);
}

static std::runtime_error makeCublasError(
    cublasStatus_t status,
    const char* expression,
    const char* file,
    int line)
{
    char message[1024];
    std::snprintf(
        message,
        sizeof(message),
        "cuBLAS call failed at %s:%d: %s (status %d)",
        file,
        line,
        expression,
        static_cast<int>(status));
    return std::runtime_error(message);
}

static std::runtime_error makeCudssError(
    cudssStatus_t status,
    const char* expression,
    const char* file,
    int line)
{
    char message[1024];
    std::snprintf(
        message,
        sizeof(message),
        "cuDSS call failed at %s:%d: %s (status %d)",
        file,
        line,
        expression,
        static_cast<int>(status));
    return std::runtime_error(message);
}

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        const cudaError_t status__ = (call);                                 \
        if (status__ != cudaSuccess)                                         \
            throw makeCudaError(status__, #call, __FILE__, __LINE__);        \
    } while (0)

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        const cublasStatus_t status__ = (call);                              \
        if (status__ != CUBLAS_STATUS_SUCCESS)                               \
            throw makeCublasError(status__, #call, __FILE__, __LINE__);      \
    } while (0)

#define CUDSS_CHECK(call)                                                    \
    do {                                                                     \
        const cudssStatus_t status__ = (call);                               \
        if (status__ != CUDSS_STATUS_SUCCESS)                                \
            throw makeCudssError(status__, #call, __FILE__, __LINE__);       \
    } while (0)

#define MEX_PRINT(...)                                                       \
    do {                                                                     \
        mexPrintf(__VA_ARGS__);                                              \
        mexPrintf("\n");                                                     \
        mexEvalString("drawnow;");                                           \
    } while (0)

// C++ level 0 corresponds to MATLAB meshHierarchy_(1).
struct Level
{
    int nx = 0;
    int ny = 0;
    int nz = 0;

    int numNodes = 0;
    int numElements = 0;
    int numDOFs = 0;

    size_t numGridNodes = 0;

    // MATLAB-owned host data. These pointers are used only during this MEX call.
    const int32_t* h_nodeToElements = nullptr;
    const int32_t* h_eNodMat = nullptr;
    const int32_t* h_nodGridId = nullptr;
    const int32_t* h_nodMapForward = nullptr;
    const double* h_eleModulus = nullptr;
    const double* h_dK = nullptr;

    // Device hierarchy data.
    int32_t* d_nodeToElements = nullptr;
    int32_t* d_eNodMat = nullptr;
    int32_t* d_nodGridId = nullptr;
    int32_t* d_nodMapForward = nullptr;
    double* d_eleModulus = nullptr;
    double* d_dK = nullptr;

    // V-cycle workspace.
    double* d_rhs = nullptr;
    double* d_x = nullptr;
    double* d_residual = nullptr;
    double* d_temp = nullptr;

    double* d_rTilde = nullptr;
};

struct SolverContext
{
    bool initialized = false;

    int numLevels = 0;
    size_t size = 0;
    double tolerance = 0.0;
    int maxIterations = 0;
    double jacobiOmega = 0.0;

    int iterations = 0;
    double relativeResidual = 0.0;
    bool converged = false;
    mwSize outputRows = 0;
    mwSize outputCols = 0;

    std::vector<Level> levels;
    std::vector<int> spanWidths;

    cublasHandle_t cublasHandle = nullptr;

    int numFixedDOFs = 0;
    const int32_t* h_fixedDOFIds = nullptr;
    int32_t* d_fixedDOFIds = nullptr;

    int numCoarseFreeDOFs = 0;
    const int32_t* h_coarseFreeDOFIds = nullptr;
    int32_t* d_coarseFreeDOFIds = nullptr;

    const double* h_Ke = nullptr;

    // MATLAB CSC is interpreted as CSR of A^T. This is equivalent because
    // coarseKFree is required to be symmetric.
    std::vector<int32_t> h_coarseRowOffsets;
    std::vector<int32_t> h_coarseColIndices;
    const double* h_coarseValues = nullptr;
    int64_t coarseNNZ = 0;

    int32_t* d_coarseRowOffsets = nullptr;
    int32_t* d_coarseColIndices = nullptr;
    double* d_coarseValues = nullptr;

    double* d_coarseRhsFree = nullptr;
    double* d_coarseXFree = nullptr;

    cudssHandle_t cudssHandle = nullptr;
    cudssConfig_t cudssConfig = nullptr;
    cudssData_t cudssData = nullptr;
    cudssMatrix_t cudssA = nullptr;
    cudssMatrix_t cudssB = nullptr;
    cudssMatrix_t cudssX = nullptr;

    void* d_workspace = nullptr;

    const double* h_b = nullptr;
    const double* h_y = nullptr;

    double* d_y = nullptr;
    double* d_r = nullptr;
    double* d_z = nullptr;
    double* d_p = nullptr;
    double* d_Ap = nullptr;

    // CUDA events for timing.
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    float totalVcycleTimeMs = 0.0f;
    float totalSpMVTimeMs = 0.0f;
};

__global__ void zeroSelectedDOFsKernel(
    double* vector,
    const int32_t* fixedDOFIds,
    int numFixedDOFs)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numFixedDOFs)
        return;

    const int dof = fixedDOFIds[index] - 1;

    vector[dof] = 0.0;
}

__global__ void gatherSelectedDOFsKernel(
    const double* __restrict__ fullVector,
    const int32_t* __restrict__ selectedDOFIds,
    double* __restrict__ selectedVector,
    int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        selectedVector[i] = fullVector[selectedDOFIds[i] - 1];
}

__global__ void scatterSelectedDOFsKernel(
    const double* __restrict__ selectedVector,
    const int32_t* __restrict__ selectedDOFIds,
    double* __restrict__ fullVector,
    int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        fullVector[selectedDOFIds[i] - 1] = selectedVector[i];
}

__global__ void dampedJacobiSmootherKernelFine(
    const double* __restrict__ r,
    const double* __restrict__ diagK,
    double * __restrict__ x,
    double* __restrict__ rTilde,
    const double weightFactorJacobi,
    const int numDOFs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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
    int fineNode = blockIdx.x * blockDim.x + threadIdx.x;

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
    const int fineGridId = fineNodeGridId[fineNode] - 1;

    /*
     * MATLAB/column-major node ordering:
     *
     * gridId = y
     *        + x * numYNodes
     *        + z * numYNodes * numXNodes
     */
    const int fineY = fineGridId % fineNyNodes;

    const int fineX = (fineGridId / fineNyNodes) % fineNxNodes;

    const int fineZ = fineGridId / (fineNyNodes * fineNxNodes);

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
    const int offsetY = fineY - coarseBaseY * spanWidth;

    const int offsetX = fineX - coarseBaseX * spanWidth;

    const int offsetZ = fineZ - coarseBaseZ * spanWidth;

    const double invSpan = 1.0 / static_cast<double>(spanWidth);

    /*
     * Interpolation coordinates within the coarse cell.
     */
    const double ty = static_cast<double>(offsetY) * invSpan;

    const double tx = static_cast<double>(offsetX) * invSpan;

    const double tz = static_cast<double>(offsetZ) * invSpan;

    /*
     * Candidate coarse coordinates and corresponding 1D weights.
     *
     * A fine node can interpolate from:
     *
     * base coarse node     weight = 1 - t
     * base + 1 coarse node weight = t
     */
    const int coarseYs[2] = { coarseBaseY, coarseBaseY + 1};

    const int coarseXs[2] = { coarseBaseX, coarseBaseX + 1};

    const int coarseZs[2] = {coarseBaseZ, coarseBaseZ + 1 };

    const double wy[2] = {1.0 - ty, ty};

    const double wx[2] = {1.0 - tx, tx };

    const double wz[2] = {1.0 - tz, tz };

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
                const int activeCoarseNode = coarseNodeMapForward[coarseGridId];

                if (activeCoarseNode == 0)
                    continue;

                const int coarseNode = activeCoarseNode - 1;

                const double weight = wx[ix] * wy[iy] * wz[iz];

                const int input = 3 * coarseNode;

                resultX += weight * coarseResidual[input + 0];

                resultY += weight * coarseResidual[input + 1];

                resultZ += weight * coarseResidual[input + 2];
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
    int coarseNode = blockIdx.x * blockDim.x + threadIdx.x;

    if (coarseNode >= numCoarseNodes) {
        return;
    }

    int coarseNyNodes = coarseNy + 1;
    int coarseNxNodes = coarseNx + 1;

    int fineNyNodes = fineNy + 1;
    int fineNxNodes = fineNx + 1;
    int fineNzNodes = fineNz + 1;

    int coarseGridId = coarseNodeGridId[coarseNode] - 1;

    int coarseY = coarseGridId % coarseNyNodes;

    int coarseX = (coarseGridId / coarseNyNodes) % coarseNxNodes;

    int coarseZ = coarseGridId / (coarseNyNodes * coarseNxNodes);

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

            double wx = 1.0 - double(abs(dx)) / double(spanWidth);
            #pragma unroll
            for (int dy = -radius; dy <= radius; ++dy)
            {
                int fineY = fineCenterY + dy;

                if (fineY < 0 || fineY >= fineNyNodes)
                    continue;

                double wy = 1.0 - double(abs(dy)) / double(spanWidth);

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

                resultX += weight * fineResidual[input + 0];

                resultY += weight * fineResidual[input + 1];

                resultZ += weight * fineResidual[input + 2];
            }
        }
    }

    int output = 3 * coarseNode;

    coarseResidual[output + 0] = resultX;
    coarseResidual[output + 1] = resultY;
    coarseResidual[output + 2] = resultZ;

}

static int gridSizeFor(
    int count,
    int blockSize)
{
    if (count <= 0)
        return 0;

    return (count + blockSize - 1) / blockSize;
}

static size_t checkedAdd(
    size_t a,
    size_t b)
{
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::runtime_error("GPU workspace size overflow while adding byte counts.");

    return a + b;
}

static size_t checkedMultiply(
    size_t a,
    size_t b)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        throw std::runtime_error("GPU workspace size overflow while multiplying byte counts.");

    return a * b;
}

static void addBytes(
    size_t& total,
    size_t count,
    size_t elementSize)
{
    total = checkedAdd(
        total,
        checkedMultiply(count, elementSize));
}

static const mxArray* requireField(
    const mxArray* structure,
    const char* fieldName)
{
    const mxArray* field = mxGetField(structure, 0, fieldName);

    if (field == nullptr)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:missingField",
            "H.%s is required.",
            fieldName);
    }

    return field;
}

static const mxArray* requireCellEntry(
    const mxArray* cellArray,
    mwIndex index,
    const char* fieldName)
{
    if (!mxIsCell(cellArray))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:cellField",
            "H.%s must be a cell array.",
            fieldName);
    }

    if (index >= mxGetNumberOfElements(cellArray))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:cellSize",
            "H.%s does not contain hierarchy level %llu.",
            fieldName,
            static_cast<unsigned long long>(index + 1));
    }

    const mxArray* value = mxGetCell(cellArray, index);

    if (value == nullptr)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:emptyCell",
            "H.%s{%llu} is empty.",
            fieldName,
            static_cast<unsigned long long>(index + 1));
    }

    return value;
}

static void requireRealDoubleArray(
    const mxArray* value,
    mwSize requiredElements,
    const char* description)
{
    if (!mxIsDouble(value) || mxIsComplex(value) || mxGetNumberOfElements(value) != requiredElements)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:doubleArray",
            "%s must be a real double array with %llu entries.",
            description,
            static_cast<unsigned long long>(requiredElements));
    }
}

static void requireInt32Array(
    const mxArray* value,
    mwSize requiredElements,
    const char* description)
{
    if (!mxIsInt32(value) || mxIsComplex(value) || mxGetNumberOfElements(value) != requiredElements)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:int32Array",
            "%s must be an int32 array with %llu entries.",
            description,
            static_cast<unsigned long long>(requiredElements));
    }
}

static int readPositiveIntegerScalar(
    const mxArray* value,
    const char* description)
{
    if (value == nullptr || mxGetNumberOfElements(value) != 1 || !mxIsNumeric(value) || mxIsComplex(value))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:integerScalar",
            "%s must be a real numeric scalar.",
            description);
    }

    const double scalar = mxGetScalar(value);

    if (!std::isfinite(scalar) || scalar < 1.0 || std::floor(scalar) != scalar || scalar > static_cast<double>(INT_MAX))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:integerScalar",
            "%s must be a positive integer representable by int.",
            description);
    }

    return static_cast<int>(scalar);
}

static void validateOneBasedIds(
    const int32_t* ids,
    size_t count,
    int upperBound,
    bool zeroAllowed,
    const char* description)
{
    for (size_t i = 0; i < count; ++i)
    {
        const int32_t value = ids[i];

        const bool valid =
            zeroAllowed
                ? (value >= 0 && value <= upperBound)
                : (value >= 1 && value <= upperBound);

        if (!valid)
        {
            mexErrMsgIdAndTxt(
                "mgpcg_gpu:indexRange",
                "%s contains the invalid value %d at linear position %llu.",
                description,
                static_cast<int>(value),
                static_cast<unsigned long long>(i + 1));
        }
    }
}

static size_t calculateRequiredGPUBytes(
    const SolverContext& solver)
{
    size_t bytes = 0;

    // Double arrays first so every double pointer remains naturally aligned.
    addBytes( bytes, static_cast<size_t>(solver.levels[0].numElements), sizeof(double)); // finest eleModulus

    for (int levelIndex = 0; levelIndex < solver.numLevels - 1; ++levelIndex)
    {
        addBytes(bytes, static_cast<size_t>(solver.levels[levelIndex].numDOFs), sizeof(double)); // diagK
    }

    const size_t finestDOFs = static_cast<size_t>(solver.levels[0].numDOFs);

    addBytes(bytes, 5 * finestDOFs, sizeof(double)); // y, r, z, p, Ap
    addBytes(bytes, 2 * static_cast<size_t>(solver.numCoarseFreeDOFs), sizeof(double)); // reduced coarse RHS and x
    addBytes(bytes, static_cast<size_t>(solver.coarseNNZ), sizeof(double)); // coarse matrix values

    for (int levelIndex = 0; levelIndex < solver.numLevels; ++levelIndex)
    {
        addBytes(bytes, 4 * static_cast<size_t>(solver.levels[levelIndex].numDOFs), sizeof(double)); // rhs, x, residual, temp
    }

    // Integer arrays follow all doubles.
    for (int levelIndex = 0; levelIndex < solver.numLevels; ++levelIndex)
    {
        const Level& level = solver.levels[levelIndex];

        addBytes(bytes, static_cast<size_t>(level.numNodes) * 8, sizeof(int32_t));
        addBytes(bytes, static_cast<size_t>(level.numElements) * 8, sizeof(int32_t));
        addBytes(bytes,static_cast<size_t>(level.numNodes),sizeof(int32_t));
        addBytes(bytes,level.numGridNodes,sizeof(int32_t));
    }

    addBytes(bytes,static_cast<size_t>(solver.numFixedDOFs),sizeof(int32_t));
    addBytes(bytes,static_cast<size_t>(solver.numCoarseFreeDOFs),sizeof(int32_t));
    addBytes(bytes,static_cast<size_t>(solver.numCoarseFreeDOFs) + 1,sizeof(int32_t)); // coarse row offsets
    addBytes(bytes,static_cast<size_t>(solver.coarseNNZ),sizeof(int32_t)); // coarse column indices

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
    if (hierarchyMx == nullptr ||!mxIsStruct(hierarchyMx) ||mxGetNumberOfElements(hierarchyMx) != 1)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:hierarchy",
            "H must be a scalar MATLAB struct.");
    }

    if (!mxIsDouble(bMx) ||mxIsComplex(bMx) ||!mxIsDouble(yMx) ||mxIsComplex(yMx))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:inputType",
            "b and y0 must be real double arrays.");
    }

    if (!std::isfinite(tolerance) || tolerance <= 0.0)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:tolerance",
            "tolerance must be finite and positive.");
    }

    if (maxIterations <= 0)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:maxIterations",
            "maxIterations must be a positive integer.");
    }

    const int numLevels =
        readPositiveIntegerScalar(
            requireField(hierarchyMx, "numLevels"),
            "H.numLevels");

    if (numLevels < 2)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:numLevels",
            "At least two multigrid levels are required.");
    }

    const mxArray* resXField = requireField(hierarchyMx, "resX");
    const mxArray* resYField = requireField(hierarchyMx, "resY");
    const mxArray* resZField = requireField(hierarchyMx, "resZ");
    const mxArray* numNodesField = requireField(hierarchyMx, "numNodes");
    const mxArray* numElementsField = requireField(hierarchyMx, "numElements");
    const mxArray* numDOFsField = requireField(hierarchyMx, "numDOFs");
    const mxArray* spanWidthField = requireField(hierarchyMx, "spanWidth");

    requireInt32Array(resXField,static_cast<mwSize>(numLevels),"H.resX");
    requireInt32Array(resYField,static_cast<mwSize>(numLevels),"H.resY");
    requireInt32Array(resZField,static_cast<mwSize>(numLevels),"H.resZ");
    requireInt32Array(numNodesField,static_cast<mwSize>(numLevels),"H.numNodes");
    requireInt32Array(numElementsField,static_cast<mwSize>(numLevels),"H.numElements");
    requireInt32Array(numDOFsField,static_cast<mwSize>(numLevels),"H.numDOFs");
    requireInt32Array(spanWidthField,static_cast<mwSize>(numLevels - 1),"H.spanWidth");

    const int32_t* resX = static_cast<const int32_t*>(mxGetData(resXField));
    const int32_t* resY = static_cast<const int32_t*>(mxGetData(resYField));
    const int32_t* resZ = static_cast<const int32_t*>(mxGetData(resZField));
    const int32_t* numNodes = static_cast<const int32_t*>(mxGetData(numNodesField));
    const int32_t* numElements = static_cast<const int32_t*>(mxGetData(numElementsField));
    const int32_t* numDOFs = static_cast<const int32_t*>(mxGetData(numDOFsField));
    const int32_t* spanWidth = static_cast<const int32_t*>(mxGetData(spanWidthField));

    const mxArray* nodeToElementsField = requireField(hierarchyMx, "nodeToElements");
    const mxArray* eNodMatField = requireField(hierarchyMx, "eNodMat");
    const mxArray* nodGridIdField = requireField(hierarchyMx, "nodGridId");
    const mxArray* nodMapForwardField = requireField(hierarchyMx, "nodMapForward");
    const mxArray* eleModulusField = requireField(hierarchyMx, "eleModulus");
    const mxArray* diagKField = requireField(hierarchyMx, "diagK");

    if (!mxIsCell(nodeToElementsField) ||
        mxGetNumberOfElements(nodeToElementsField) < static_cast<mwSize>(numLevels) ||
        !mxIsCell(eNodMatField) ||
        mxGetNumberOfElements(eNodMatField) < static_cast<mwSize>(numLevels) ||
        !mxIsCell(nodGridIdField) ||
        mxGetNumberOfElements(nodGridIdField) < static_cast<mwSize>(numLevels) ||
        !mxIsCell(nodMapForwardField) ||
        mxGetNumberOfElements(nodMapForwardField) < static_cast<mwSize>(numLevels) ||
        !mxIsCell(eleModulusField) ||
        mxGetNumberOfElements(eleModulusField) < 1 ||
        !mxIsCell(diagKField) ||
        mxGetNumberOfElements(diagKField) < static_cast<mwSize>(numLevels - 1))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:hierarchyCells",
            "Hierarchy cell fields do not contain the required number of levels.");
    }

    const mxArray* jacobiOmegaMx = mxGetField(hierarchyMx, 0, "jacobiOmega");

    if (jacobiOmegaMx == nullptr)
        jacobiOmegaMx = mxGetField(hierarchyMx, 0, "weightFactorJacobi");

    if (jacobiOmegaMx == nullptr ||
        !mxIsDouble(jacobiOmegaMx) ||
        mxIsComplex(jacobiOmegaMx) ||
        mxGetNumberOfElements(jacobiOmegaMx) != 1)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:jacobiOmega",
            "H must contain scalar jacobiOmega or weightFactorJacobi.");
    }

    const double jacobiOmega = mxGetScalar(jacobiOmegaMx);

    if (!std::isfinite(jacobiOmega))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:jacobiOmega",
            "The Jacobi weight must be finite.");
    }

    solver.initialized = false;
    solver.numLevels = numLevels;
    solver.levels.assign(static_cast<size_t>(numLevels),Level{});
    solver.spanWidths.resize(static_cast<size_t>(numLevels - 1));

    solver.tolerance = tolerance;
    solver.maxIterations = maxIterations;
    solver.jacobiOmega = jacobiOmega;
    solver.h_b =static_cast<const double*>(mxGetData(bMx));
    solver.h_y =static_cast<const double*>(mxGetData(yMx));
    solver.outputRows = mxGetM(bMx);
    solver.outputCols = mxGetN(bMx);

    for (int levelIndex = 0;levelIndex < numLevels;++levelIndex)
    {
        Level& level =
            solver.levels[levelIndex];

        level.nx = static_cast<int>(resX[levelIndex]);
        level.ny = static_cast<int>(resY[levelIndex]);
        level.nz = static_cast<int>(resZ[levelIndex]);
        level.numNodes =static_cast<int>(numNodes[levelIndex]);
        level.numElements =static_cast<int>(numElements[levelIndex]);
        level.numDOFs =static_cast<int>(numDOFs[levelIndex]);

        if (level.nx < 0 ||
            level.ny < 0 ||
            level.nz < 0 ||
            level.numNodes <= 0 ||
            level.numElements <= 0 ||
            level.numDOFs <= 0)
        {
            mexErrMsgIdAndTxt(
                "mgpcg_gpu:levelDimensions",
                "Hierarchy level %d contains invalid dimensions or counts.",
                levelIndex + 1);
        }

        if (level.numDOFs != 3 * level.numNodes)
        {
            mexErrMsgIdAndTxt(
                "mgpcg_gpu:numDOFs",
                "H.numDOFs(%d) must equal 3*H.numNodes(%d).",
                levelIndex + 1,
                levelIndex + 1);
        }

        const size_t nyNodes =static_cast<size_t>(level.ny) + 1;
        const size_t nxNodes =static_cast<size_t>(level.nx) + 1;
        const size_t nzNodes =static_cast<size_t>(level.nz) + 1;

        level.numGridNodes =checkedMultiply(checkedMultiply(nyNodes, nxNodes),nzNodes);

        if (level.numGridNodes >
            static_cast<size_t>(INT_MAX))
        {
            mexErrMsgIdAndTxt(
                "mgpcg_gpu:gridIndexRange",
                "Hierarchy level %d has more grid nodes than the int-based transfer kernels support.",
                levelIndex + 1);
        }

        const mxArray* nodeToElementsMx = requireCellEntry(nodeToElementsField,static_cast<mwIndex>(levelIndex),"nodeToElements");
        const mxArray* eNodMatMx = requireCellEntry(eNodMatField,static_cast<mwIndex>(levelIndex),"eNodMat");
        const mxArray* nodGridIdMx = requireCellEntry(nodGridIdField,static_cast<mwIndex>(levelIndex),"nodGridId");
        const mxArray* nodMapForwardMx = requireCellEntry(nodMapForwardField,static_cast<mwIndex>(levelIndex),"nodMapForward");

        requireInt32Array(nodeToElementsMx,static_cast<mwSize>(level.numNodes) * 8,"H.nodeToElements{level}");
        requireInt32Array(eNodMatMx,static_cast<mwSize>(level.numElements) * 8,"H.eNodMat{level}");
        requireInt32Array(nodGridIdMx,static_cast<mwSize>(level.numNodes),"H.nodGridId{level}");
        requireInt32Array(nodMapForwardMx,static_cast<mwSize>(level.numGridNodes),"H.nodMapForward{level}");

        level.h_nodeToElements = static_cast<const int32_t*>(mxGetData(nodeToElementsMx));
        level.h_eNodMat = static_cast<const int32_t*>(mxGetData(eNodMatMx));
        level.h_nodGridId = static_cast<const int32_t*>(mxGetData(nodGridIdMx));
        level.h_nodMapForward = static_cast<const int32_t*>(mxGetData(nodMapForwardMx));

        validateOneBasedIds(
            level.h_nodGridId,
            level.numNodes,
            static_cast<int>(level.numGridNodes),
            false,
            "H.nodGridId{level}");

        validateOneBasedIds(
            level.h_nodMapForward,
            static_cast<int>(level.numGridNodes),
            level.numNodes,
            true,
            "H.nodMapForward{level}");

        if (levelIndex < numLevels - 1)
        {
            const mxArray* diagKMx =
                requireCellEntry(diagKField,static_cast<mwIndex>(levelIndex),"diagK");

            requireRealDoubleArray(diagKMx,static_cast<mwSize>(level.numDOFs),"H.diagK{level}");

            level.h_dK = static_cast<const double*>(mxGetData(diagKMx));

            for (int dof = 0;dof < level.numDOFs;++dof)
            {
                if (!std::isfinite(level.h_dK[dof]) ||level.h_dK[dof] == 0.0)
                {
                    mexErrMsgIdAndTxt(
                        "mgpcg_gpu:diagK",
                        "H.diagK{%d} contains a zero or non-finite value at entry %d.",
                        levelIndex + 1,
                        dof + 1);
                }
            }

            const int width = static_cast<int>(spanWidth[levelIndex]);

            if (width <= 0)
            {
                mexErrMsgIdAndTxt(
                    "mgpcg_gpu:spanWidth",
                    "H.spanWidth(%d) must be positive.",
                    levelIndex + 1);
            }

            solver.spanWidths[levelIndex] = width;
        }
    }

    const mxArray* eleModulusMx = requireCellEntry(eleModulusField,0,"eleModulus");

    requireRealDoubleArray(eleModulusMx,static_cast<mwSize>(solver.levels[0].numElements),"H.eleModulus{1}");

    solver.levels[0].h_eleModulus = static_cast<const double*>(mxGetData(eleModulusMx));

    // The unchanged K-by-U kernel relies on valid one-based connectivity.
    validateOneBasedIds(
        solver.levels[0].h_nodeToElements,
        static_cast<size_t>(solver.levels[0].numNodes) * 8,
        solver.levels[0].numElements,
        true,
        "H.nodeToElements{1}");

    validateOneBasedIds(
        solver.levels[0].h_eNodMat,
        static_cast<size_t>(solver.levels[0].numElements) * 8,
        solver.levels[0].numNodes,
        false,
        "H.eNodMat{1}");

    const mxArray* fixedDOFIdsMx = requireField(hierarchyMx, "fixedDOFIds");

    if (!mxIsInt32(fixedDOFIdsMx) || mxIsComplex(fixedDOFIdsMx) || mxGetNumberOfElements(fixedDOFIdsMx) > static_cast<mwSize>(INT_MAX))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:fixedDOFIds",
            "H.fixedDOFIds must be an int32 array with at most INT_MAX entries.");
    }

    solver.h_fixedDOFIds = static_cast<const int32_t*>(mxGetData(fixedDOFIdsMx));
    solver.numFixedDOFs = static_cast<int>(mxGetNumberOfElements(fixedDOFIdsMx));

    if (solver.numFixedDOFs > 0)
    {
        validateOneBasedIds(
            solver.h_fixedDOFIds,
            solver.numFixedDOFs,
            solver.levels[0].numDOFs,
            false,
            "H.fixedDOFIds");
    }

    const mxArray* coarseFreeDOFIdsMx = requireField(hierarchyMx, "coarseFreeDOFIds");

    if (!mxIsInt32(coarseFreeDOFIdsMx) ||
        mxIsComplex(coarseFreeDOFIdsMx) ||
        mxGetNumberOfElements(coarseFreeDOFIdsMx) == 0 ||
        mxGetNumberOfElements(coarseFreeDOFIdsMx) >
            static_cast<mwSize>(INT_MAX))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:coarseFreeDOFIds",
            "H.coarseFreeDOFIds must be a nonempty int32 array with at most INT_MAX entries.");
    }

    solver.h_coarseFreeDOFIds = static_cast<const int32_t*>( mxGetData(coarseFreeDOFIdsMx));
    solver.numCoarseFreeDOFs = static_cast<int>(mxGetNumberOfElements(coarseFreeDOFIdsMx));

    validateOneBasedIds(
        solver.h_coarseFreeDOFIds,
        solver.numCoarseFreeDOFs,
        solver.levels.back().numDOFs,
        false,
        "H.coarseFreeDOFIds");

    const mxArray* keMx = requireField(hierarchyMx, "Ke");

    requireRealDoubleArray(keMx,24 * 24, "H.Ke");

    solver.h_Ke =static_cast<const double*>(mxGetData(keMx));

    if (mxGetNumberOfElements(bMx) !=static_cast<mwSize>(solver.levels[0].numDOFs) ||mxGetNumberOfElements(yMx) !=static_cast<mwSize>(solver.levels[0].numDOFs))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:vectorSize",
            "b and y0 must contain finest-level numDOFs entries.");
    }

    const mxArray* coarseKFreeMx = requireField(hierarchyMx, "coarseKFree");

    if (!mxIsSparse(coarseKFreeMx) || !mxIsDouble(coarseKFreeMx) || mxIsComplex(coarseKFreeMx))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:coarseKFree",
            "H.coarseKFree must be a real sparse double matrix.");
    }

    const mwSize coarseRows = mxGetM(coarseKFreeMx);
    const mwSize coarseCols = mxGetN(coarseKFreeMx);

    if (coarseRows != coarseCols || coarseRows != static_cast<mwSize>(solver.numCoarseFreeDOFs))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:coarseKFreeSize",
            "H.coarseKFree must be numCoarseFreeDOFs-by-numCoarseFreeDOFs.");
    }

    const mwIndex* jc = mxGetJc(coarseKFreeMx);
    const mwIndex* ir = mxGetIr(coarseKFreeMx);
    const mwIndex nnz = jc[coarseCols];

    if (coarseRows > static_cast<mwSize>(INT_MAX) || nnz > static_cast<mwIndex>(INT_MAX))
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:coarseKFreeIndexRange",
            "H.coarseKFree exceeds the int32 CSR limits used by this implementation.");
    }

    solver.h_coarseRowOffsets.resize(static_cast<size_t>(coarseRows) + 1);
    solver.h_coarseColIndices.resize(static_cast<size_t>(nnz));

    for (mwSize i = 0; i <= coarseRows; ++i)
    {
        solver.h_coarseRowOffsets[i] = static_cast<int32_t>(jc[i]);
    }

    for (mwIndex i = 0; i < nnz; ++i)
    {
        solver.h_coarseColIndices[i] = static_cast<int32_t>(ir[i]);
    }

    solver.h_coarseValues = static_cast<const double*>(mxGetData(coarseKFreeMx));
    solver.coarseNNZ = static_cast<int64_t>(nnz);
    solver.size = calculateRequiredGPUBytes(solver);
}

static void initializeCoarseSolver(
    SolverContext& solver);

static void initializeGPU(
    SolverContext& solver)
{
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    size_t freeMemory = 0;
    size_t totalMemory = 0;

    CUDA_CHECK(cudaMemGetInfo(&freeMemory, &totalMemory));

    if (solver.size > freeMemory)
    {
        char message[512];
        std::snprintf(
            message,
            sizeof(message),
            "The CUDA device has %zu bytes free, but the explicit MGPCG workspace requires %zu bytes. cuDSS will also allocate internal memory.",
            freeMemory,
            solver.size);
        throw std::runtime_error(message);
    }

    CUDA_CHECK(cudaMalloc(&solver.d_workspace,solver.size));

    char* dW = static_cast<char*>(solver.d_workspace);
    size_t offset = 0;

    auto takeDouble = [&](size_t count) -> double*
        {
            const size_t bytes = checkedMultiply(count, sizeof(double));

            if (offset > solver.size || bytes > solver.size - offset)
            {
                throw std::runtime_error("Double workspace layout exceeds the allocated buffer.");
            }

            double* pointer = reinterpret_cast<double*>(dW + offset);
            offset += bytes;
            return pointer;
        };

    auto takeInt32 = [&](size_t count) -> int32_t*
        {
            const size_t bytes = checkedMultiply(count, sizeof(int32_t));

            if (offset > solver.size || bytes > solver.size - offset)
            {
                throw std::runtime_error("Integer workspace layout exceeds the allocated buffer.");
            }

            int32_t* pointer = reinterpret_cast<int32_t*>(dW + offset);
            offset += bytes;
            return pointer;
        };

    // All double arrays.
    solver.levels[0].d_eleModulus = takeDouble(static_cast<size_t>(solver.levels[0].numElements));

    for (int levelIndex = 0;levelIndex < solver.numLevels - 1;++levelIndex)
    {
        solver.levels[levelIndex].d_dK = takeDouble(static_cast<size_t>(solver.levels[levelIndex].numDOFs));
    }

    const size_t finestDOFs = static_cast<size_t>(solver.levels[0].numDOFs);

    solver.d_y = takeDouble(finestDOFs);
    solver.d_r = takeDouble(finestDOFs);
    solver.d_z = takeDouble(finestDOFs);
    solver.d_p = takeDouble(finestDOFs);
    solver.d_Ap = takeDouble(finestDOFs);

    solver.d_coarseRhsFree = takeDouble(static_cast<size_t>(solver.numCoarseFreeDOFs));
    solver.d_coarseXFree = takeDouble(static_cast<size_t>(solver.numCoarseFreeDOFs));
    solver.d_coarseValues = takeDouble(static_cast<size_t>(solver.coarseNNZ));

    for (int levelIndex = 0; levelIndex < solver.numLevels; ++levelIndex)
    {
        Level& level = solver.levels[levelIndex];

        level.d_rhs = takeDouble(static_cast<size_t>(level.numDOFs));
        level.d_x =takeDouble(static_cast<size_t>(level.numDOFs));
        level.d_residual =takeDouble(static_cast<size_t>(level.numDOFs));
        level.d_temp = takeDouble(static_cast<size_t>(level.numDOFs));
        level.d_rTilde = nullptr;
    }

    // All integer arrays.
    for (int levelIndex = 0; levelIndex < solver.numLevels; ++levelIndex)
    {
        Level& level = solver.levels[levelIndex];

        level.d_nodeToElements = takeInt32(static_cast<size_t>(level.numNodes) * 8);
        level.d_eNodMat = takeInt32(static_cast<size_t>(level.numElements) * 8);
        level.d_nodGridId = takeInt32(static_cast<size_t>(level.numNodes));
        level.d_nodMapForward = takeInt32(level.numGridNodes);
    }

    solver.d_fixedDOFIds = takeInt32(static_cast<size_t>(solver.numFixedDOFs));
    solver.d_coarseFreeDOFIds = takeInt32(static_cast<size_t>(solver.numCoarseFreeDOFs));
    solver.d_coarseRowOffsets = takeInt32(static_cast<size_t>(solver.numCoarseFreeDOFs) + 1);
    solver.d_coarseColIndices = takeInt32(static_cast<size_t>(solver.coarseNNZ));

    if (offset != solver.size)
    {
        throw std::runtime_error("Internal GPU workspace byte count does not match the allocated size.");
    }

    for (int levelIndex = 0; levelIndex < solver.numLevels; ++levelIndex)
    {
        Level& level = solver.levels[levelIndex];

        CUDA_CHECK(
            cudaMemcpy(level.d_nodeToElements, level.h_nodeToElements,
                static_cast<size_t>(level.numNodes) * 8 * sizeof(int32_t),
                cudaMemcpyHostToDevice));

        CUDA_CHECK(
            cudaMemcpy(level.d_eNodMat, level.h_eNodMat,
                static_cast<size_t>(level.numElements) * 8 * sizeof(int32_t),
                cudaMemcpyHostToDevice));

        CUDA_CHECK(
            cudaMemcpy(level.d_nodGridId, level.h_nodGridId,
                static_cast<size_t>(level.numNodes) * sizeof(int32_t),
                cudaMemcpyHostToDevice));

        CUDA_CHECK(
            cudaMemcpy(level.d_nodMapForward, level.h_nodMapForward,
                level.numGridNodes * sizeof(int32_t),
                cudaMemcpyHostToDevice));

        if (levelIndex == 0)
        {
            CUDA_CHECK(
                cudaMemcpy(level.d_eleModulus, level.h_eleModulus,
                    static_cast<size_t>(level.numElements) * sizeof(double),
                    cudaMemcpyHostToDevice));
        }

        if (levelIndex < solver.numLevels - 1)
        {
            CUDA_CHECK(
                cudaMemcpy(level.d_dK, level.h_dK,
                    static_cast<size_t>(level.numDOFs) * sizeof(double),
                    cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(
            cudaMemset(
                level.d_rhs, 0,
                static_cast<size_t>(level.numDOFs) * sizeof(double)));
        CUDA_CHECK(
            cudaMemset(level.d_x, 0,
                static_cast<size_t>(level.numDOFs) * sizeof(double)));
        CUDA_CHECK(
            cudaMemset(level.d_residual, 0,
                static_cast<size_t>(level.numDOFs) * sizeof(double)));
        CUDA_CHECK(
            cudaMemset(level.d_temp, 0,
                static_cast<size_t>(level.numDOFs) * sizeof(double)));
    }

    if (solver.numFixedDOFs > 0)
    {
        CUDA_CHECK(
            cudaMemcpy(solver.d_fixedDOFIds, solver.h_fixedDOFIds,
                static_cast<size_t>(solver.numFixedDOFs) * sizeof(int32_t),
                cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(
        cudaMemcpy(solver.d_coarseFreeDOFIds, solver.h_coarseFreeDOFIds,
            static_cast<size_t>(solver.numCoarseFreeDOFs) * sizeof(int32_t),
            cudaMemcpyHostToDevice));

    CUDA_CHECK(
        cudaMemcpy(solver.d_coarseRowOffsets, solver.h_coarseRowOffsets.data(),
            (static_cast<size_t>(solver.numCoarseFreeDOFs) + 1) * sizeof(int32_t),
            cudaMemcpyHostToDevice));

    CUDA_CHECK(
        cudaMemcpy(solver.d_coarseColIndices, solver.h_coarseColIndices.data(),
            static_cast<size_t>(solver.coarseNNZ) * sizeof(int32_t),
            cudaMemcpyHostToDevice));

    CUDA_CHECK(
        cudaMemcpy(solver.d_coarseValues, solver.h_coarseValues,
            static_cast<size_t>(solver.coarseNNZ) * sizeof(double),
            cudaMemcpyHostToDevice));

    CUDA_CHECK(
        cudaMemcpy(
            solver.d_y,
            solver.h_y,
            finestDOFs * sizeof(double),
            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        solver.d_r,
        solver.h_b,
        finestDOFs * sizeof(double),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(
        cudaMemset(
            solver.d_z,
            0,
            finestDOFs * sizeof(double)));
    CUDA_CHECK(
        cudaMemset(
            solver.d_p,
            0,
            finestDOFs * sizeof(double)));
    CUDA_CHECK(
        cudaMemset(
            solver.d_Ap,
            0,
            finestDOFs * sizeof(double)));
    CUDA_CHECK(
        cudaMemset(
            solver.d_coarseRhsFree,
            0,
            static_cast<size_t>(
                solver.numCoarseFreeDOFs) *
                sizeof(double)));
    CUDA_CHECK(
        cudaMemset(
            solver.d_coarseXFree,
            0,
            static_cast<size_t>(
                solver.numCoarseFreeDOFs) *
                sizeof(double)));

    CUDA_CHECK(
        cudaMemcpyToSymbol(
            c_Ke,
            solver.h_Ke,
            24 * 24 * sizeof(double)));

    CUBLAS_CHECK(
        cublasCreate(
            &solver.cublasHandle));
    CUBLAS_CHECK(
        cublasSetPointerMode(
            solver.cublasHandle,
            CUBLAS_POINTER_MODE_HOST));

    initializeCoarseSolver(solver);

    solver.initialized = true;

    MEX_PRINT(
        "MGPCG explicit GPU workspace: %.3f GiB (device free before cuDSS internals: %.3f GiB)",
        static_cast<double>(solver.size) /
            (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(freeMemory) /
            (1024.0 * 1024.0 * 1024.0));
}

static void initializeCoarseSolver(
    SolverContext& solver)
{
    const int64_t n =
        static_cast<int64_t>(
            solver.numCoarseFreeDOFs);

    CUDSS_CHECK(
        cudssCreate(
            &solver.cudssHandle));
    CUDSS_CHECK(
        cudssSetStream(
            solver.cudssHandle,
            static_cast<cudaStream_t>(0)));
    CUDSS_CHECK(
        cudssConfigCreate(
            &solver.cudssConfig));
    CUDSS_CHECK(
        cudssDataCreate(
            solver.cudssHandle,
            &solver.cudssData));

    CUDSS_CHECK(
        cudssMatrixCreateCsr(
            &solver.cudssA,
            n,
            n,
            solver.coarseNNZ,
            solver.d_coarseRowOffsets,
            nullptr,
            solver.d_coarseColIndices,
            solver.d_coarseValues,
            CUDSS_R_32I,
            CUDSS_R_32I,
            CUDSS_R_64F,
            CUDSS_MTYPE_SPD,
            CUDSS_MVIEW_LOWER,
            CUDSS_BASE_ZERO));

    CUDSS_CHECK(
        cudssMatrixCreateDn(
            &solver.cudssB,
            n,
            1,
            n,
            solver.d_coarseRhsFree,
            CUDSS_R_64F,
            CUDSS_LAYOUT_COL_MAJOR));

    CUDSS_CHECK(
        cudssMatrixCreateDn(
            &solver.cudssX,
            n,
            1,
            n,
            solver.d_coarseXFree,
            CUDSS_R_64F,
            CUDSS_LAYOUT_COL_MAJOR));

    CUDSS_CHECK(
        cudssExecute(
            solver.cudssHandle,
            CUDSS_PHASE_ANALYSIS,
            solver.cudssConfig,
            solver.cudssData,
            solver.cudssA,
            solver.cudssX,
            solver.cudssB));

    CUDSS_CHECK(
        cudssExecute(
            solver.cudssHandle,
            CUDSS_PHASE_FACTORIZATION,
            solver.cudssConfig,
            solver.cudssData,
            solver.cudssA,
            solver.cudssX,
            solver.cudssB));

    // Factorization can be asynchronous. Synchronize before reading INFO.
    CUDA_CHECK(cudaDeviceSynchronize());

    int info = 0;
    size_t sizeWritten = 0;

    CUDSS_CHECK(
        cudssDataGet(
            solver.cudssHandle,
            solver.cudssData,
            CUDSS_DATA_INFO,
            &info,
            sizeof(info),
            &sizeWritten));

    if (sizeWritten != sizeof(info))
    {
        throw std::runtime_error(
            "cuDSS returned an unexpected CUDSS_DATA_INFO size.");
    }

    if (info != 0)
    {
        char message[512];
        std::snprintf(
            message,
            sizeof(message),
            "cuDSS coarse Cholesky factorization failed at reordered 1-based minor %d.",
            info);
        throw std::runtime_error(message);
    }
}

static void zeroFixedDOFs(
    SolverContext& solver,
    double* d_vector)
{
    if (solver.numFixedDOFs == 0)
        return;

    constexpr int blockSize = 256;
    const int gridSize =
        gridSizeFor(
            solver.numFixedDOFs,
            blockSize);

    zeroSelectedDOFsKernel<<<gridSize, blockSize>>>(
        d_vector,
        solver.d_fixedDOFIds,
        solver.numFixedDOFs);

    CUDA_CHECK(cudaGetLastError());
}

static void solveCoarsest(
    SolverContext& solver,
    const double* d_rhs,
    double* d_x)
{
    constexpr int blockSize = 256;
    const int n =
        solver.numCoarseFreeDOFs;
    const int gridSize =
        gridSizeFor(
            n,
            blockSize);

    CUDA_CHECK(
        cudaMemset(
            d_x,
            0,
            static_cast<size_t>(
                solver.levels.back().numDOFs) *
                sizeof(double)));

    CUDA_CHECK(
        cudaMemset(
            solver.d_coarseXFree,
            0,
            static_cast<size_t>(n) *
                sizeof(double)));

    gatherSelectedDOFsKernel<<<gridSize, blockSize>>>(
        d_rhs,
        solver.d_coarseFreeDOFIds,
        solver.d_coarseRhsFree,
        n);
    CUDA_CHECK(cudaGetLastError());

    CUDSS_CHECK(
        cudssExecute(
            solver.cudssHandle,
            CUDSS_PHASE_SOLVE,
            solver.cudssConfig,
            solver.cudssData,
            solver.cudssA,
            solver.cudssX,
            solver.cudssB));

    // cuDSS and the kernels below use stream 0, so the scatter is ordered
    // after the solve without a host synchronization.
    scatterSelectedDOFsKernel<<<gridSize, blockSize>>>(
        solver.d_coarseXFree,
        solver.d_coarseFreeDOFIds,
        d_x,
        n);
    CUDA_CHECK(cudaGetLastError());
}

static void destroySolver(
    SolverContext& solver) noexcept
{
    if (solver.cudssA != nullptr)
    {
        (void)cudssMatrixDestroy(
            solver.cudssA);
        solver.cudssA = nullptr;
    }

    if (solver.cudssX != nullptr)
    {
        (void)cudssMatrixDestroy(
            solver.cudssX);
        solver.cudssX = nullptr;
    }

    if (solver.cudssB != nullptr)
    {
        (void)cudssMatrixDestroy(
            solver.cudssB);
        solver.cudssB = nullptr;
    }

    if (solver.cudssData != nullptr &&
        solver.cudssHandle != nullptr)
    {
        (void)cudssDataDestroy(
            solver.cudssHandle,
            solver.cudssData);
        solver.cudssData = nullptr;
    }

    if (solver.cudssConfig != nullptr)
    {
        (void)cudssConfigDestroy(
            solver.cudssConfig);
        solver.cudssConfig = nullptr;
    }

    if (solver.cudssHandle != nullptr)
    {
        (void)cudssDestroy(
            solver.cudssHandle);
        solver.cudssHandle = nullptr;
    }

    if (solver.cublasHandle != nullptr)
    {
        (void)cublasDestroy(
            solver.cublasHandle);
        solver.cublasHandle = nullptr;
    }

    if (solver.d_workspace != nullptr)
    {
        (void)cudaFree(
            solver.d_workspace);
        solver.d_workspace = nullptr;
    }

    solver.initialized = false;
}

static void applyVcycle(
    SolverContext& solver,
    const double* d_fineResidual,
    double* d_fineCorrection)
{
    constexpr int blockSize = 256;
    const int lastLevel = solver.numLevels - 1;

    Level& finest = solver.levels[0];

    CUDA_CHECK(
        cudaMemcpy(finest.d_rhs, d_fineResidual,
            static_cast<size_t>(finest.numDOFs) * sizeof(double),
            cudaMemcpyDeviceToDevice));

    for (int levelIndex = 0; levelIndex < solver.numLevels; ++levelIndex)
    {
        Level& level = solver.levels[levelIndex];

        CUDA_CHECK(
            cudaMemset(
                level.d_x, 0,
                static_cast<size_t>(level.numDOFs) * sizeof(double)));

        CUDA_CHECK(
            cudaMemset(
                level.d_temp, 0,
                static_cast<size_t>(level.numDOFs) * sizeof(double)));
    }

    // Restriction
    for (int fineIndex = 0; fineIndex < lastLevel; ++fineIndex)
    {
        Level& fine = solver.levels[fineIndex];
        Level& coarse = solver.levels[fineIndex + 1];

        const int smoothGrid = gridSizeFor(fine.numDOFs, blockSize);

        if (fineIndex == 0)
        {
            // The unchanged kernel writes x. Its rTilde pointer assignment is
            // local to the kernel, so pass x for both pointer arguments.
            dampedJacobiSmootherKernelFine<<<smoothGrid, blockSize>>>(
                fine.d_rhs,
                fine.d_dK,
                fine.d_x,
                fine.d_x,
                solver.jacobiOmega,
                fine.numDOFs);
        }
        else
        {
            dampedJacobiSmootherKernelCoarse<<<smoothGrid, blockSize>>>(
                fine.d_rhs,
                fine.d_dK,
                fine.d_x,
                solver.jacobiOmega,
                fine.numDOFs);
        }
        CUDA_CHECK(cudaGetLastError());

        const int restrictionGrid = gridSizeFor(coarse.numNodes, blockSize);

        restrictResidualKernel<<<restrictionGrid, blockSize>>>(
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
    nvtxRangePushA("MGPCG:solveCoarsest");
    // Solve the coarsest level system with cuDSS.
    solveCoarsest(solver, solver.levels[lastLevel].d_rhs, solver.levels[lastLevel].d_x);
    nvtxRangePop();
    // Interpolation and post-smoothing
    for (int coarseIndex = lastLevel; coarseIndex >= 1; --coarseIndex)
    {
        const int fineIndex = coarseIndex - 1;

        Level& fine = solver.levels[fineIndex];
        Level& coarse = solver.levels[coarseIndex];

        const int interpolationGrid = gridSizeFor(fine.numNodes, blockSize);

        // fine.d_temp = P * coarse.d_x
        interpolateResidualKernel<<<interpolationGrid, blockSize>>>(
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

        const int dofGrid = gridSizeFor(fine.numDOFs, blockSize);

        // Existing first Jacobi term + interpolated coarse correction.
        addVectorsInPlaceKernel<<<dofGrid, blockSize>>>(
            fine.d_x,
            fine.d_temp,
            fine.numDOFs);
        CUDA_CHECK(cudaGetLastError());

        // Reuse temp for the second omega*r/diagK term.
        if (fineIndex == 0)
        {
            dampedJacobiSmootherKernelFine<<<dofGrid, blockSize>>>(
                fine.d_rhs,
                fine.d_dK,
                fine.d_temp,
                fine.d_temp,
                solver.jacobiOmega,
                fine.numDOFs);
        }
        else
        {
            dampedJacobiSmootherKernelCoarse<<<dofGrid, blockSize>>>(
                fine.d_rhs,
                fine.d_dK,
                fine.d_temp,
                solver.jacobiOmega,
                fine.numDOFs);
        }
        CUDA_CHECK(cudaGetLastError());

        addVectorsInPlaceKernel<<<dofGrid, blockSize>>>(
            fine.d_x,
            fine.d_temp,
            fine.numDOFs);
        CUDA_CHECK(cudaGetLastError());
    }

    zeroFixedDOFs(solver, finest.d_x);

    CUDA_CHECK(
        cudaMemcpy(d_fineCorrection, finest.d_x,
            static_cast<size_t>(finest.numDOFs) * sizeof(double),
            cudaMemcpyDeviceToDevice));
}

__global__ void substractVectorsInPlaceKernel(
    const double* a,
    double* b,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        b[idx] = a[idx] - b[idx];
    }
}

static void applyFinestOperator(
    SolverContext& solver,
    const double* d_x,
    double* d_Ax)
{
    constexpr int blockSize = 256;
    Level& finest = solver.levels[0];

    const int gridSize = gridSizeFor(finest.numNodes, blockSize);

    kbyu_kernel<<<gridSize, blockSize>>>(
        d_x,
        d_Ax,
        finest.d_nodeToElements,
        finest.d_eNodMat,
        finest.d_eleModulus,
        finest.numNodes,
        finest.numElements);

    CUDA_CHECK(cudaGetLastError());

    zeroFixedDOFs(solver, d_Ax);
}

static void runMGPCG(
    SolverContext& solver)
{
    if (!solver.initialized || solver.levels.empty() || solver.cublasHandle == nullptr)
    {
        throw std::runtime_error("The solver context is not fully initialized.");
    }

    const int n = solver.levels[0].numDOFs;
    cublasHandle_t handle = solver.cublasHandle;

    const double one = 1.0;
    const double minusOne = -1.0;

    solver.iterations = 0;
    solver.relativeResidual = 0.0;
    solver.converged = false;

    zeroFixedDOFs(solver, solver.d_r);
    zeroFixedDOFs(solver, solver.d_y);

    double normB = 0.0;

    CUBLAS_CHECK(
        cublasDnrm2(
            handle,
            n,
            solver.d_r,
            1,
            &normB));

    // r = b - A*y
    nvtxRangePushA("MGPCG:applyFinestOperator");
    applyFinestOperator(solver, solver.d_y,solver.d_Ap);
    nvtxRangePop();

    nvtxRangePushA("MGPCG:initialResidual");

    CUBLAS_CHECK(
        cublasDaxpy(
            handle,
            n,
            &minusOne,
            solver.d_Ap,
            1,
            solver.d_r,
            1));

    nvtxRangePop();
    zeroFixedDOFs(solver, solver.d_r);

    double residualNorm = 0.0;

    CUBLAS_CHECK(
        cublasDnrm2(
            handle,
            n,
            solver.d_r,
            1,
            &residualNorm));

    const double residualDenominator = (normB > 0.0) ? normB : 1.0;

    solver.relativeResidual = residualNorm / residualDenominator;

    if (!std::isfinite(solver.relativeResidual))
    {
        throw std::runtime_error("The initial relative residual is not finite.");
    }

    if (solver.relativeResidual < solver.tolerance || residualNorm == 0.0)
    {
        solver.converged = true;
        return;
    }

    // z = M^{-1}r; p = z; rho = z'*r
    nvtxRangePushA("MGPCG:applyVcycle1");
    applyVcycle(solver, solver.d_r, solver.d_z);
    nvtxRangePop();

    CUBLAS_CHECK(
        cublasDcopy(
            handle,
            n,
            solver.d_z,
            1,
            solver.d_p,
            1));

    double rho = 0.0;

    CUBLAS_CHECK(
        cublasDdot(
            handle,
            n,
            solver.d_z,
            1,
            solver.d_r,
            1,
            &rho));

    if (!std::isfinite(rho) || rho <= 0.0)
    {
        throw std::runtime_error("PCG breakdown: the initial z'*r is non-positive or non-finite.");
    }

    // Start the main PCG iteration loop.
    nvtxRangePushA("MGPCG:mainLoop");
    for (int its = 1; its <= solver.maxIterations; ++its)
    {
        nvtxRangePushA("MGPCG:applyFinestOperator");
        applyFinestOperator(solver, solver.d_p, solver.d_Ap);
        nvtxRangePop();
        double pAp = 0.0;

        CUBLAS_CHECK(
            cublasDdot(
                handle,
                n,
                solver.d_p,
                1,
                solver.d_Ap,
                1,
                &pAp));

        if (!std::isfinite(pAp) || pAp <= 0.0)
        {
            char message[512];
            std::snprintf(
                message,
                sizeof(message),
                "PCG breakdown at iteration %d: p'*A*p is non-positive or non-finite.",
                its);
            throw std::runtime_error(message);
        }

        const double alpha = rho / pAp;

        if (!std::isfinite(alpha))
        {
            throw std::runtime_error("PCG breakdown: alpha is not finite.");
        }

        const double minusAlpha = -alpha;

        CUBLAS_CHECK(
            cublasDaxpy(
                handle,
                n,
                &alpha,
                solver.d_p,
                1,
                solver.d_y,
                1));

        CUBLAS_CHECK(
            cublasDaxpy(
                handle,
                n,
                &minusAlpha,
                solver.d_Ap,
                1,
                solver.d_r,
                1));

        zeroFixedDOFs(solver, solver.d_y);
        zeroFixedDOFs(solver, solver.d_r);

        CUBLAS_CHECK(
            cublasDnrm2(
                handle,
                n,
                solver.d_r,
                1,
                &residualNorm));

        solver.iterations = its;
        solver.relativeResidual = residualNorm / residualDenominator;

        if (!std::isfinite(solver.relativeResidual))
        {
            throw std::runtime_error("PCG produced a non-finite relative residual.");
        }

        if (solver.relativeResidual < solver.tolerance)
        {
            solver.converged = true;

            MEX_PRINT("CG solver converged at iteration %d with relative residual %.6e",
                its,
                solver.relativeResidual);
            break;
        }
        nvtxRangePushA("MGPCG:applyVcycle2");
        applyVcycle(solver, solver.d_r, solver.d_z);
        nvtxRangePop();
        double rhoNew = 0.0;

        CUBLAS_CHECK(
            cublasDdot(
                handle,
                n,
                solver.d_z,
                1,
                solver.d_r,
                1,
                &rhoNew));

        if (!std::isfinite(rhoNew) || rhoNew <= 0.0)
        {
            char message[512];
            std::snprintf(
                message,
                sizeof(message),
                "PCG breakdown at iteration %d: z'*r is non-positive or non-finite.",
                its);
            throw std::runtime_error(message);
        }

        const double beta = rhoNew / rho;

        if (!std::isfinite(beta))
        {
            throw std::runtime_error("PCG breakdown: beta is not finite.");
        }

        // p = z + beta*p
        CUBLAS_CHECK(
            cublasDscal(
                handle,
                n,
                &beta,
                solver.d_p,
                1));

        CUBLAS_CHECK(
            cublasDaxpy(
                handle,
                n,
                &one,
                solver.d_z,
                1,
                solver.d_p,
                1));

        rho = rhoNew;
    }

    nvtxRangePop();

    if (!solver.converged && solver.iterations == solver.maxIterations)
    {
        mexWarnMsgIdAndTxt(
            "mgpcg_gpu:maxIterations",
            "Exceeded the maximum iteration count. Final relative residual: %.6e.",
            solver.relativeResidual);
    }
}

static void createMATLABOutputs(
    const SolverContext& solver,
    int nlhs,
    mxArray* plhs[])
{
    plhs[0] = mxCreateDoubleMatrix( solver.outputRows, solver.outputCols, mxREAL);

    CUDA_CHECK(
        cudaMemcpy(mxGetData(plhs[0]), solver.d_y,
            static_cast<size_t>(solver.levels[0].numDOFs) * sizeof(double),
            cudaMemcpyDeviceToHost));

    if (nlhs >= 2)
    {
        plhs[1] = mxCreateDoubleScalar(static_cast<double>(solver.iterations));
    }

    if (nlhs >= 3)
    {
        plhs[2] = mxCreateDoubleScalar(solver.relativeResidual);
    }
}

void mexFunction(
    int nlhs,
    mxArray* plhs[],
    int nrhs,
    const mxArray* prhs[])
{
    if (nrhs != 5)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:nrhs",
            "Usage: [y, iterations, relativeResidual] = Solving_MGPCG_GPU(b, tolerance, maxIterations, y0, H)");
    }

    if (nlhs < 1 || nlhs > 3)
    {
        mexErrMsgIdAndTxt(
            "mgpcg_gpu:nlhs", "The function supports one to three outputs.");
    }

    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1)
    {
        mexErrMsgIdAndTxt("mgpcg_gpu:tolerance", "tolerance must be a real double scalar.");
    }

    if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1)
    {
        mexErrMsgIdAndTxt("mgpcg_gpu:maxIterations", "maxIterations must be a real double scalar containing a positive integer.");
    }

    const double tolerance = mxGetScalar(prhs[1]);
    const double maxIterationsValue = mxGetScalar(prhs[2]);

    if (!std::isfinite(maxIterationsValue) || maxIterationsValue < 1.0 || std::floor(maxIterationsValue) != maxIterationsValue || maxIterationsValue > static_cast<double>(INT_MAX))
    {
        mexErrMsgIdAndTxt("mgpcg_gpu:maxIterations", "maxIterations must be a positive integer representable by int.");
    }

    const int maxIterations = static_cast<int>(maxIterationsValue);

    std::string failureMessage;
    {
        SolverContext solver;
        try
        {
            initializeGPUHierarchy(
                solver,
                prhs[4],
                prhs[0],
                prhs[3],
                tolerance,
                maxIterations);

            initializeGPU(solver);

            runMGPCG(solver);

            createMATLABOutputs(solver, nlhs, plhs);
        }
        catch (const std::exception& exception)
        {
            failureMessage =
                exception.what();
        }
        catch (...)
        {
            failureMessage =
                "Unknown C++ exception in Solving_MGPCG_GPU.";
        }

        destroySolver(solver);
    }

    if (!failureMessage.empty())
    {
        mexErrMsgIdAndTxt("mgpcg_gpu:runtime","%s",failureMessage.c_str());
    }
}
