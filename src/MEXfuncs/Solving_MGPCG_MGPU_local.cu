#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudss.h>
#include <nccl.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <omp.h>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
/*
Notes on implementation:
    - One additional issue: throwing a C++ exception from one OpenMP thread while other threads are waiting at a barrier is unsafe. Eventually, CUDA/NCCL errors should be handled through a shared failure mechanism or by aborting the collective operation consistently.
    - Check Cuda device setting
    - every gpu can have a cublas handle, but only one cudss handle is needed for the coarsest level solve. The cudss handle is created on GPU 0 and used for the coarsest level solve on all GPUs.

*/
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

// NCCL errors are C++ errors; never terminate the MATLAB host with exit().
static std::runtime_error makeNcclError(
    ncclResult_t status, const char* expression, const char* file, int line)
{
    char message[1024];
    std::snprintf(message, sizeof(message),
        "NCCL call failed at %s:%d: %s -> %s (status %d)",
        file, line, expression, ncclGetErrorString(status), static_cast<int>(status));
    return std::runtime_error(message);
}

#define NCCL_CHECK(call)                                                     \
    do {                                                                     \
        const ncclResult_t status__ = (call);                                 \
        if (status__ != ncclSuccess)                                          \
            throw makeNcclError(status__, #call, __FILE__, __LINE__);          \
    } while (0)

// Aborting a failed worker must also release peers waiting at a CPU barrier.
// This replaces ONLY synchronization's error path; no new CUDA syncs are added.
class FailureAwareBarrier
{
    const int participants_;
    std::atomic<int> arrivals_{0};
    std::atomic<unsigned> generation_{0};
    std::atomic<bool> failed_{false};
    mutable std::mutex errorMutex_;
    std::exception_ptr firstError_;

public:
    explicit FailureAwareBarrier(int participants) : participants_(participants) {}

    void fail(std::exception_ptr error) noexcept
    {
        try {
            std::lock_guard<std::mutex> lock(errorMutex_);
            if (!firstError_)
                firstError_ = error;
        } catch (...) { /* reporting must never throw */ }
        failed_.store(true, std::memory_order_release);
    }

    void wait()
    {
        if (failed_.load(std::memory_order_acquire))
            throw std::runtime_error("Multi-GPU worker failed before synchronization.");

        const unsigned expected = generation_.load(std::memory_order_acquire);
        if (arrivals_.fetch_add(1, std::memory_order_acq_rel) == participants_ - 1)
        {
            arrivals_.store(0, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
        }
        else
        {
            unsigned spins = 0;
            while (generation_.load(std::memory_order_acquire) == expected)
            {
                if (failed_.load(std::memory_order_acquire))
                    throw std::runtime_error("Multi-GPU worker failed during synchronization.");
                if ((++spins & 255u) == 0)
                    std::this_thread::yield();
            }
        }
        if (failed_.load(std::memory_order_acquire))
            throw std::runtime_error("Multi-GPU worker failed during synchronization.");
    }

    std::exception_ptr error() const
    {
        std::lock_guard<std::mutex> lock(errorMutex_);
        return firstError_;
    }
};

static thread_local FailureAwareBarrier* currentWorkerBarrier = nullptr;

static void workerBarrier()
{
    if (!currentWorkerBarrier)
        throw std::runtime_error("Multi-GPU synchronization used outside a worker.");
    currentWorkerBarrier->wait();
}

struct ScopedWorkerBarrier
{
    explicit ScopedWorkerBarrier(FailureAwareBarrier& barrier) noexcept
    {
        currentWorkerBarrier = &barrier;
    }
    ~ScopedWorkerBarrier() noexcept { currentWorkerBarrier = nullptr; }
    ScopedWorkerBarrier(const ScopedWorkerBarrier&) = delete;
    ScopedWorkerBarrier& operator=(const ScopedWorkerBarrier&) = delete;
};

// A temporary device scalar has exactly the same allocation lifetime on success,
// but is also freed if a CUDA/NCCL check throws.
class ScopedDeviceScalar
{
    double* ptr_ = nullptr;
public:
    ScopedDeviceScalar()
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(double)));
    }
    ~ScopedDeviceScalar() noexcept
    {
        if (ptr_) (void)cudaFree(ptr_);
    }
    void releaseChecked()
    {
        if (!ptr_) return;
        double* temporary = ptr_;
        ptr_ = nullptr;
        CUDA_CHECK(cudaFree(temporary));
    }
    double* get() const noexcept { return ptr_; }
    ScopedDeviceScalar(const ScopedDeviceScalar&) = delete;
    ScopedDeviceScalar& operator=(const ScopedDeviceScalar&) = delete;
};

class ScopedNcclGroup
{
    bool started_ = false;
public:
    ScopedNcclGroup() { NCCL_CHECK(ncclGroupStart()); started_ = true; }
    void finish()
    {
        if (!started_) return;
        started_ = false;
        NCCL_CHECK(ncclGroupEnd());
    }
    ~ScopedNcclGroup() noexcept
    {
        // If an operation threw while the group was open, the outer worker
        // failure handler aborts the communicator. Do not risk blocking in
        // ncclGroupEnd() during stack unwinding.
    }
    ScopedNcclGroup(const ScopedNcclGroup&) = delete;
    ScopedNcclGroup& operator=(const ScopedNcclGroup&) = delete;
};

struct ScopedNvtxRange
{
    explicit ScopedNvtxRange(const char* name) { nvtxRangePushA(name); }
    ~ScopedNvtxRange() noexcept { nvtxRangePop(); }
    ScopedNvtxRange(const ScopedNvtxRange&) = delete;
    ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
};

// C++ level 0 corresponds to MATLAB meshHierarchy_(1).
struct StreamData
{
    cudaStream_t rightStream = nullptr;
    cudaStream_t leftStream = nullptr;
    cudaStream_t computeStream = nullptr;
    cudaStream_t ncclStream = nullptr;
};

struct HaloSide
{
    int neighborGPU = -1;            // CUDA device ID; -1 for no neighbor.
    std::vector<int32_t> sendLocal;  // MATLAB 1-based positions in local nodes.
    std::vector<int32_t> sendDestLocal; // MATLAB 1-based positions on receiver.
    int32_t* d_sendLocal = nullptr;
    double* d_sendBuffer = nullptr;
};

struct InterfaceSide
{
    int neighborGPU = -1;

    std::vector<int32_t> sendLocal;
    std::vector<int32_t> sendDestLocal; //identifies corresponding nodes on the receiver.

    int32_t* d_sendLocal = nullptr; //identifies nodes on the sender.
    double* d_sendBuffer = nullptr;

    // Incoming message on this side
    int recvCount = 0;
    std::vector<int32_t> recvLocal;
    int32_t* d_recvLocal = nullptr; //the receiving GPU's device copy of the incoming sender's sendDestLocal, in the same order as the message.
    // example--->GPU0.rightInterface.sendDestLocal --> GPU 1.leftInterface.d_recvLocal.
    double* d_recvBuffer = nullptr;
};

struct coarsestLevelData
{
    int numCoarseFreeDOFs = 0;
    int coarseNNZ = 0;

    const int32_t* h_coarseRowPtr = nullptr;
    const int32_t* h_coarseColInd = nullptr;
    const double* h_coarseValues = nullptr;

    int32_t* d_coarseRowPtr = nullptr;
    int32_t* d_coarseColInd = nullptr;
    double* d_coarseValues = nullptr;

    double* d_coarseRHS = nullptr;
    double* d_coarseX = nullptr;

    cudssHandle_t cudssHandle = nullptr;
    cudssConfig_t cudssConfig = nullptr;
    cudssData_t cudssData = nullptr;
    cudssMatrix_t cudssA = nullptr;
    cudssMatrix_t cudssB = nullptr;
    cudssMatrix_t cudssX = nullptr;

    // All allocated on GPU 0.
    std::vector<double*> d_recvResidual;
    std::vector<double*> d_sendResidual;
    std::vector<int32_t*> d_globalNodeIds;
    std::vector<int32_t*> d_localNodeIds;

    // Number of normal local coarse nodes per GPU.
    std::vector<int> localNodeCounts;

    // Full global coarse vectors on GPU 0.
    double* d_globalRHS = nullptr;
    double* d_globalX = nullptr;

    double* d_coarseRhsFree = nullptr;
    double* d_coarseXFree = nullptr;
    int32_t* d_coarseFreeDOFIds = nullptr;
    std::vector<int32_t> h_rowOffsetsOwned, h_colIndicesOwned, h_freeIdsOwned;
    std::vector<uint8_t*> d_ownedMasks;
    std::vector<double> h_coarseValuesOwned;
    bool initialized = false;
    int globalNumDOFs = 0;
};

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
    int numLeftSharedNodes = 0;

    // MATLAB-owned host data; pointers are valid during this MEX call.
    const int32_t* h_nodeToElements = nullptr;
    const int32_t* h_eNodMat = nullptr;
    const int32_t* h_nodGridId = nullptr;
    const int32_t* h_nodMapForward = nullptr;
    const int32_t* h_sharedNodesLocal = nullptr;
    const double* h_eleModulus = nullptr;
    const double* h_dK = nullptr;
    const uint8_t* h_ownedNodeMask = nullptr;
    std::vector<uint8_t> ownedMask;
    bool explicitOwnership = false;
    std::vector<int32_t> globalNodeIds;

    // Device hierarchy data. This operation only needs the geometry maps.
    int32_t* d_nodeToElements = nullptr;
    int32_t* d_eNodMat = nullptr;
    int32_t* d_nodGridId = nullptr;
    int32_t* d_nodMapForward = nullptr;
    int32_t* d_sharedNodesLocal = nullptr;
    double* d_eleModulus = nullptr;
    double* d_dK = nullptr;
    uint8_t* d_ownedNodeMask = nullptr;

    // Partition information and per-call vectors.
    const int32_t* h_globalNodeIds = nullptr;
    int numBufferNodes = 0;
    double* d_fineResidual = nullptr;    // [normal local | left halo | right halo]
    double* d_coarseResidual = nullptr;  // normal local nodes only
    HaloSide left, right;

    // V-cycle workspace.
    double* d_rhs = nullptr;
    double* d_x = nullptr;
    double* d_rTilde = nullptr;
    
    // KbyU kernel partitioning information for this level.
    InterfaceSide leftInterface, rightInterface;
};

struct singleGPUData
{
    bool initialized = false;

    int deviceId = -1;
    int numGPUs = 0;

    int numLevels = 0;
    size_t size = 0;
    double tolerance = 0.0;
    int maxIterations = 0;
    double jacobiOmega = 0.0;

    std::vector<Level> levels;     // levels[0] = fine, levels[1] = coarse
    std::vector<int> spanWidths;
    StreamData streams; // per-level streams for communication and computation

    // Kept from the existing data layout for compatibility with later stages.
    const double* h_Ke = nullptr;
    const double* h_b = nullptr;
    const double* h_y = nullptr;
    const double* h_r = nullptr;

    double* d_y = nullptr;
    double* d_r = nullptr;
    double* d_p = nullptr;
    double* d_Ap = nullptr;


    // CUDA events for timing.
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    float totalVcycleTimeMs = 0.0f;
    float totalSpMVTimeMs = 0.0f;

    // Data for coarsest level solve using cuDSS. 
    int iterations = 0;
    double relativeResidual = 0.0;
    bool converged = false;
    cublasHandle_t cublasHandle = nullptr;
    int numFixedDOFs = 0;
    const int32_t* h_fixedDOFIds = nullptr;
    std::vector<int32_t> fixedDOFIds;
    int32_t* d_fixedDOFIds = nullptr;
    void* d_workspace = nullptr;
    double* d_dotPartials = nullptr;
    double* d_dotScalar = nullptr;
    int partialCount = 0;
    std::vector<double> localB, localY;
    coarsestLevelData coarsestLevel;
};
// Global coarse resources exist only on GPU 0; VcycleShared is retained for
// compatibility with the existing multi-GPU solver call graph.
using GlobalCoarseSolve = coarsestLevelData;
struct VcycleShared {};

__global__ void ownedDotPartialKernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    const uint8_t* __restrict__ ownedMask,
    double* __restrict__ partial,
    int numNodes)
{
    extern __shared__ double sums[];

    const int tid = threadIdx.x;
    const int index = blockIdx.x * blockDim.x + tid;

    double sum = 0.0;

    if (index < numNodes && ownedMask[index])
    {
        const int base = 3 * index;

        sum = a[base]     * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2];
    }

    sums[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
            sums[tid] += sums[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = sums[0];
}
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
    (void)rTilde;  // The kernel writes x; assigning a pointer here would not update the caller.
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
__global__ void addDampedJacobiKernel(
    const double* __restrict__ rhs,
    const double* __restrict__ diagK,
    double* __restrict__ x,
    double omega,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        x[i] += omega * rhs[i] / diagK[i];
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

        // Connectivity indices are range-checked once during host initialization.
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
        if (localNode < 0) continue;

        // Build element displacement vector Ue
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int n = elemNodes[j];
            // Element-node indices are validated during host initialization.
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

    fineResidual[output + 0] += resultX;
    fineResidual[output + 1] += resultY;
    fineResidual[output + 2] += resultZ;
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

__global__ void gatherGlobalCoarseResidualKernel(
    const double* __restrict__ localResidual,
    double* __restrict__ globalResidual,
    const int32_t* __restrict__ globalNodeIds,
    const uint8_t* __restrict__ ownedMask,
    int numLocalNodes)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numLocalNodes || !ownedMask[i]) return;
    const int src = 3 * i;
    const int dst = 3 * (globalNodeIds[i] - 1);
    // A unique global-node owner is validated during host initialization.
    globalResidual[dst]     = localResidual[src];
    globalResidual[dst + 1] = localResidual[src + 1];
    globalResidual[dst + 2] = localResidual[src + 2];
}

__global__ void scatterGlobalCoarseResidualKernel(
    double* __restrict__ sendBuffer,
    const double* __restrict__ globalResidual,
    const int32_t* __restrict__ globalNodeIds,
    int numLocalNodes)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numLocalNodes)
        return;

    int src = 3 * (globalNodeIds[i] - 1);
    int dst = 3 * i;

    sendBuffer[dst]     = globalResidual[src];
    sendBuffer[dst + 1] = globalResidual[src + 1];
    sendBuffer[dst + 2] = globalResidual[src + 2];
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

static const mxArray* requireField(const mxArray* structure, const char* name)
{
    const mxArray* value = mxGetField(structure,0,name);
    if (!value) throw std::runtime_error(std::string("Missing field ")+name);
    return value;
}
static const mxArray* requireCellEntry(const mxArray* cell, mwIndex index,
                                        const char* fieldName)
{
    if (!mxIsCell(cell) || index >= mxGetNumberOfElements(cell))
        throw std::runtime_error(std::string(fieldName)+" must contain all hierarchy levels.");
    const mxArray* entry = mxGetCell(cell,index);
    if (!entry) throw std::runtime_error(std::string(fieldName)+" has an empty cell.");
    return entry;
}
static void requireRealDoubleArray(const mxArray* value, mwSize n, const char* what)
{
    if (!value || !mxIsDouble(value) || mxIsComplex(value) || mxGetNumberOfElements(value)!=n)
        throw std::runtime_error(std::string(what)+" must be a real double array of the required size.");
}
static void requireInt32Array(const mxArray* value, mwSize n, const char* what)
{
    if (!value || !mxIsInt32(value) || mxIsComplex(value) || mxGetNumberOfElements(value)!=n)
        throw std::runtime_error(std::string(what)+" must be an int32 array of the required size.");
}
static int readPositiveIntegerScalar(const mxArray* value, const char* what)
{
    if (!value || !mxIsNumeric(value) || mxIsComplex(value) || mxGetNumberOfElements(value)!=1)
        throw std::runtime_error(std::string(what)+" must be a positive numeric scalar.");
    const double x=mxGetScalar(value);
    if (!std::isfinite(x) || x<1 || x>double(INT_MAX) || std::floor(x)!=x)
        throw std::runtime_error(std::string(what)+" must be a positive integer within int32 range.");
    return int(x);
}
static void validateOneBasedIds(const int32_t* ids,size_t count,int upperBound,
                                bool zeroAllowed,const char* what)
{
    for (size_t i=0;i<count;++i)
        if (ids[i]>(int32_t)upperBound || ids[i]<(zeroAllowed?0:1))
            throw std::runtime_error(std::string(what)+" has invalid one-based indices.");
}

// Partition contract: H.partitions is a cell array of per-GPU hierarchy structs.
// Per-GPU fields match the single-GPU hierarchy, plus globalNodeIds{level},
// optionally ownedNodeMask{level}, leftHalo/rightHalo{level} and leftInterface/
// rightInterface (struct or cell). The shared H supplies Ke, jacobiOmega,
// coarseKFree, coarseFreeDOFIds and global input/output vectors.
static const mxArray* optionalField(const mxArray* H, const char* name)
{
    return mxGetField(H, 0, name);
}

static const mxArray* optionalLevelField(const mxArray* H, const char* name,
                                          int level)
{
    const mxArray* f = optionalField(H, name);
    if (!f) return nullptr;
    return mxIsCell(f) ? (static_cast<mwIndex>(level) < mxGetNumberOfElements(f)
                           ? mxGetCell(f, level) : nullptr) : (level == 0 ? f : nullptr);
}

static std::vector<int32_t> readIndexVector(const mxArray* a, int upper,
                                             bool allowZero, const char* what)
{
    if (!a || !mxIsInt32(a) || mxIsComplex(a) || mxGetNumberOfElements(a) > mwSize(INT_MAX))
        throw std::runtime_error(std::string(what) + " must be an int32 array.");
    const size_t count = mxGetNumberOfElements(a);
    const int32_t* p = static_cast<const int32_t*>(mxGetData(a));
    validateOneBasedIds(p, count, upper, allowZero, what);
    return std::vector<int32_t>(p, p + count);
}

static void readSideIndices(const mxArray* sideMx, HaloSide& side,
                            int ownNodes, const char* description)
{
    if (!sideMx || mxIsEmpty(sideMx)) return;
    if (!mxIsStruct(sideMx) || mxGetNumberOfElements(sideMx) != 1)
        throw std::runtime_error(std::string(description) + " must be a scalar struct.");
    const mxArray* send = mxGetField(sideMx, 0, "sendLocal");
    const mxArray* dest = mxGetField(sideMx, 0, "sendDestLocal");
    if (!send || !dest)
        throw std::runtime_error(std::string(description) + " needs sendLocal and sendDestLocal.");
    side.sendLocal = readIndexVector(send, ownNodes, false, description);
    // Destination indices checked against receiver after all partitions parsed.
    side.sendDestLocal = readIndexVector(dest, INT_MAX, false, description);
    if (side.sendLocal.size() != side.sendDestLocal.size())
        throw std::runtime_error(std::string(description) + " index counts disagree.");
}

static void readSideIndices(const mxArray* sideMx, InterfaceSide& side,
                            int ownNodes, const char* description)
{
    if (!sideMx || mxIsEmpty(sideMx)) return;
    if (!mxIsStruct(sideMx) || mxGetNumberOfElements(sideMx) != 1)
        throw std::runtime_error(std::string(description) + " must be a scalar struct.");
    const mxArray* send = mxGetField(sideMx, 0, "sendLocal");
    const mxArray* dest = mxGetField(sideMx, 0, "sendDestLocal");
    if (!send || !dest)
        throw std::runtime_error(std::string(description) + " needs sendLocal and sendDestLocal.");
    side.sendLocal = readIndexVector(send, ownNodes, false, description);
    side.sendDestLocal = readIndexVector(dest, INT_MAX, false, description);
    if (side.sendLocal.size() != side.sendDestLocal.size())
        throw std::runtime_error(std::string(description) + " index counts disagree.");
}

static size_t calculateRequiredGPUBytes(const singleGPUData& G)
{
    size_t bytes = 0;
    const Level& fine = G.levels.front();
    addBytes(bytes, fine.numElements, sizeof(double));
    for (int l = 0; l + 1 < G.numLevels; ++l)
        addBytes(bytes, G.levels[l].numDOFs, sizeof(double));
    addBytes(bytes, 4 * size_t(fine.numDOFs), sizeof(double));
    for (int l = 1; l < G.numLevels; ++l)
        addBytes(bytes, 2 * size_t(G.levels[l].numDOFs), sizeof(double));
    addBytes(bytes, G.partialCount + 1, sizeof(double));
    for (const Level& L : G.levels)
        for (const HaloSide* p : {&L.left, &L.right})
            addBytes(bytes, 3 * p->sendLocal.size(), sizeof(double));
    for (const InterfaceSide* p : {&fine.leftInterface, &fine.rightInterface}) {
        addBytes(bytes, 3 * p->sendLocal.size(), sizeof(double));
        addBytes(bytes, 3 * size_t(p->recvCount), sizeof(double));
    }
    addBytes(bytes, 8 * size_t(fine.numNodes), sizeof(int32_t));
    addBytes(bytes, 8 * size_t(fine.numElements), sizeof(int32_t));
    for (const Level& L : G.levels) {
        addBytes(bytes, L.numNodes, sizeof(int32_t));
        addBytes(bytes, L.numGridNodes, sizeof(int32_t));
    }
    addBytes(bytes, G.numFixedDOFs, sizeof(int32_t));
    for (const Level& L : G.levels)
        for (const HaloSide* p : {&L.left, &L.right})
            addBytes(bytes, p->sendLocal.size(), sizeof(int32_t));
    for (const InterfaceSide* p : {&fine.leftInterface, &fine.rightInterface}) {
        addBytes(bytes, p->sendLocal.size(), sizeof(int32_t));
        addBytes(bytes, p->recvCount, sizeof(int32_t));
    }
    for (const Level& L : G.levels) addBytes(bytes, L.numNodes, sizeof(uint8_t));
    return bytes;
}

static void initializeGPUHierarchy(std::vector<singleGPUData>& Gs,
                                    GlobalCoarseSolve& C, const mxArray* H,
                                    const mxArray* bMx, const mxArray* yMx,
                                    double tolerance, int maxIterations)
{
    if (!H || !mxIsStruct(H) || mxGetNumberOfElements(H) != 1)
        throw std::runtime_error("H must be a scalar struct.");
    if (!mxIsDouble(bMx) || mxIsComplex(bMx) || !mxIsDouble(yMx) ||
        mxIsComplex(yMx) || mxGetNumberOfElements(bMx) != mxGetNumberOfElements(yMx) ||
        mxGetNumberOfElements(bMx) == 0 || mxGetNumberOfElements(bMx) % 3)
        throw std::runtime_error("b and y0 must be equal-length real double arrays of 3 DOFs/node.");
    if (!(tolerance > 0.0) || !std::isfinite(tolerance) || maxIterations <= 0)
        throw std::runtime_error("Invalid tolerance or maxIterations.");
    if (mxGetNumberOfElements(bMx) > mwSize(INT_MAX))
        throw std::runtime_error("Global fine vector exceeds int32 indexing limits.");

    const mxArray* parts = requireField(H, "partitions");
    if (!mxIsCell(parts) || mxGetNumberOfElements(parts) == 0 ||
        mxGetNumberOfElements(parts) > mwSize(INT_MAX))
        throw std::runtime_error("H.partitions must be a nonempty cell array.");
    const int ng = static_cast<int>(mxGetNumberOfElements(parts));
    if (const mxArray* declared = optionalField(H, "numGPUs"))
        if (readPositiveIntegerScalar(declared, "H.numGPUs") != ng)
            throw std::runtime_error("H.numGPUs differs from numel(H.partitions).");
    int available = 0;
    CUDA_CHECK(cudaGetDeviceCount(&available));
    if (ng > available)
        throw std::runtime_error("Not enough CUDA devices for H.partitions.");

    const mxArray* omegaMx = optionalField(H, "jacobiOmega");
    if (!omegaMx) omegaMx = requireField(H, "weightFactorJacobi");
    if (!mxIsDouble(omegaMx) || mxIsComplex(omegaMx) || mxGetNumberOfElements(omegaMx) != 1)
        throw std::runtime_error("H.jacobiOmega must be a real scalar.");
    const double omega = mxGetScalar(omegaMx);
    if (!std::isfinite(omega)) throw std::runtime_error("Jacobi weight is not finite.");
    const mxArray* ke = requireField(H, "Ke");
    requireRealDoubleArray(ke, 24*24, "H.Ke");
    const double* Ke = static_cast<const double*>(mxGetData(ke));
    const size_t globalFineNodes = mxGetNumberOfElements(bMx) / 3;
    const double* b = static_cast<const double*>(mxGetData(bMx));
    const double* y = static_cast<const double*>(mxGetData(yMx));

    Gs.resize(ng);
    int nlev = -1;
    for (int g = 0; g < ng; ++g) {
        const mxArray* P = mxGetCell(parts, g);
        if (!P || !mxIsStruct(P) || mxGetNumberOfElements(P) != 1)
            throw std::runtime_error("Each H.partitions entry must be a scalar struct.");
        singleGPUData& G = Gs[g];
        G.deviceId = g; G.numGPUs = ng;
        G.tolerance = tolerance; G.maxIterations = maxIterations;
        G.jacobiOmega = omega; G.h_Ke = Ke;
        G.numLevels = readPositiveIntegerScalar(requireField(P, "numLevels"), "partition.numLevels");
        if (G.numLevels < 2 || (nlev != -1 && nlev != G.numLevels))
            throw std::runtime_error("All partitions must use the same number of levels (at least 2).");
        nlev = G.numLevels;
        G.levels.resize(G.numLevels);
        G.spanWidths.resize(G.numLevels-1);

        const char* names[] = {"resX", "resY", "resZ", "numNodes", "numElements", "numDOFs", "spanWidth"};
        const int32_t* arrays[7];
        for (int f=0; f<7; ++f) {
            const mxArray* a = requireField(P, names[f]);
            requireInt32Array(a, f==6 ? nlev-1 : nlev, names[f]);
            arrays[f] = static_cast<const int32_t*>(mxGetData(a));
        }
        const mxArray* nodesToElem = requireField(P,"nodeToElements");
        const mxArray* elemNodes = requireField(P,"eNodMat");
        const mxArray* gridIds = requireField(P,"nodGridId");
        const mxArray* map = requireField(P,"nodMapForward");
        const mxArray* modulus = requireField(P,"eleModulus");
        const mxArray* diagonal = requireField(P,"diagK");
        const mxArray* globalIds = requireField(P,"globalNodeIds");
        if (!mxIsCell(globalIds) || mxGetNumberOfElements(globalIds)<mwSize(nlev))
            throw std::runtime_error("partition.globalNodeIds must be a cell array, one vector per level.");
        for (int l=0; l<nlev; ++l) {
            Level& L=G.levels[l];
            L.nx=arrays[0][l]; L.ny=arrays[1][l]; L.nz=arrays[2][l];
            L.numNodes=arrays[3][l]; L.numElements=arrays[4][l]; L.numDOFs=arrays[5][l];
            if (L.nx<0 || L.ny<0 || L.nz<0 || L.numNodes<=0 || L.numElements<=0 ||
                L.numNodes > INT_MAX/3 || L.numDOFs != 3*L.numNodes)
                throw std::runtime_error("Invalid partition level dimensions.");
            L.numGridNodes=checkedMultiply(checkedMultiply(size_t(L.nx)+1,
                                  size_t(L.ny)+1),size_t(L.nz)+1);
            if (L.numGridNodes>size_t(INT_MAX))
                throw std::runtime_error("Partition grid exceeds int32 indexing limits.");
            const mxArray* nodesMx = requireCellEntry(nodesToElem,l,"nodeToElements");
            const mxArray* elemMx = requireCellEntry(elemNodes,l,"eNodMat");
            const mxArray* gridMx = requireCellEntry(gridIds,l,"nodGridId");
            const mxArray* mapMx = requireCellEntry(map,l,"nodMapForward");
            requireInt32Array(nodesMx,size_t(L.numNodes)*8,"partition.nodeToElements{l}");
            requireInt32Array(elemMx,size_t(L.numElements)*8,"partition.eNodMat{l}");
            requireInt32Array(gridMx,L.numNodes,"partition.nodGridId{l}");
            requireInt32Array(mapMx,L.numGridNodes,"partition.nodMapForward{l}");
            L.h_nodeToElements=static_cast<const int32_t*>(mxGetData(nodesMx));
            L.h_eNodMat=static_cast<const int32_t*>(mxGetData(elemMx));
            L.h_nodGridId=static_cast<const int32_t*>(mxGetData(gridMx));
            L.h_nodMapForward=static_cast<const int32_t*>(mxGetData(mapMx));
            validateOneBasedIds(L.h_nodGridId,L.numNodes,int(L.numGridNodes),false,"nodGridId");
            validateOneBasedIds(L.h_nodMapForward,L.numGridNodes,L.numNodes,true,"nodMapForward");
            L.globalNodeIds=readIndexVector(requireCellEntry(globalIds,l,"globalNodeIds"),
                                             INT_MAX,false,"globalNodeIds");
            if (L.globalNodeIds.size()!=size_t(L.numNodes))
                throw std::runtime_error("globalNodeIds length differs from local numNodes.");
            L.h_globalNodeIds=L.globalNodeIds.data();
            if (l==0)
                validateOneBasedIds(L.h_globalNodeIds,L.numNodes,int(globalFineNodes),false,"fine globalNodeIds");
            if (const mxArray* m = optionalLevelField(P,"ownedNodeMask",l)) {
                if ((!mxIsLogical(m) && !mxIsUint8(m)) || mxGetNumberOfElements(m)!=mwSize(L.numNodes))
                    throw std::runtime_error("ownedNodeMask{level} must be logical/uint8 of numNodes length.");
                const uint8_t* ptr=static_cast<const uint8_t*>(mxGetData(m));
                L.ownedMask.assign(ptr,ptr+L.numNodes);
                for (uint8_t value : L.ownedMask)
                    if (value>1) throw std::runtime_error("ownedNodeMask entries must be 0 or 1.");
                L.explicitOwnership=true;
            }
            if (l<nlev-1) {
                const mxArray* d= requireCellEntry(diagonal,l,"diagK");
                requireRealDoubleArray(d,L.numDOFs,"partition.diagK{l}");
                L.h_dK=static_cast<const double*>(mxGetData(d));
                for (int i=0;i<L.numDOFs;++i)
                    if (!std::isfinite(L.h_dK[i]) || L.h_dK[i]==0.0)
                        throw std::runtime_error("diagK has a zero/non-finite entry.");
                G.spanWidths[l]=arrays[6][l];
                if (G.spanWidths[l]<=0)
                    throw std::runtime_error("spanWidth must be positive.");
            }
            L.left.neighborGPU=g>0 ? g-1 : -1;
            L.right.neighborGPU=g+1<ng ? g+1 : -1;
            readSideIndices(optionalLevelField(P,"leftHalo",l),L.left,L.numNodes,"leftHalo");
            readSideIndices(optionalLevelField(P,"rightHalo",l),L.right,L.numNodes,"rightHalo");
            if ((g==0 && !L.left.sendLocal.empty()) ||
                (g==ng-1 && !L.right.sendLocal.empty()))
                throw std::runtime_error("Halo declared on an external boundary.");
        }
        Level& fine = G.levels.front();
        const mxArray* e= requireCellEntry(modulus,0,"eleModulus");
        requireRealDoubleArray(e,fine.numElements,"partition.eleModulus{1}");
        fine.h_eleModulus=static_cast<const double*>(mxGetData(e));
        validateOneBasedIds(fine.h_nodeToElements,size_t(fine.numNodes)*8,
                            fine.numElements,true,"nodeToElements{1}");
        validateOneBasedIds(fine.h_eNodMat,size_t(fine.numElements)*8,
                            fine.numNodes,false,"eNodMat{1}");
        fine.leftInterface.neighborGPU=g>0 ? g-1:-1;
        fine.rightInterface.neighborGPU=g+1<ng ? g+1:-1;
        readSideIndices(optionalLevelField(P,"leftInterface",0),fine.leftInterface,
                        fine.numNodes,"leftInterface");
        readSideIndices(optionalLevelField(P,"rightInterface",0),fine.rightInterface,
                        fine.numNodes,"rightInterface");
        G.partialCount=gridSizeFor(fine.numNodes,256);
        G.localB.resize(fine.numDOFs); G.localY.resize(fine.numDOFs);
        for (int node=0;node<fine.numNodes;++node) {
            const size_t glob=3*size_t(fine.globalNodeIds[node]-1), loc=3*size_t(node);
            for (int c=0;c<3;++c) {
                G.localB[loc+c]=b[glob+c];
                G.localY[loc+c]=y[glob+c];
            }
        }
        G.h_b=G.localB.data(); G.h_y=G.localY.data();
        if (const mxArray* f=optionalField(P,"fixedDOFIds")) {
            G.fixedDOFIds=readIndexVector(f,fine.numDOFs,false,"partition.fixedDOFIds");
        } else if (const mxArray* f=optionalField(H,"fixedDOFIds")) {
            const std::vector<int32_t> globalFixed=
                readIndexVector(f,int(mxGetNumberOfElements(bMx)),false,"H.fixedDOFIds");
            const std::unordered_set<int32_t> fset(globalFixed.begin(),globalFixed.end());
            for (int i=0;i<fine.numNodes;++i) {
                for (int c=0;c<3;++c) {
                    const int32_t globalId=3*(fine.globalNodeIds[i]-1)+c+1;
                    if (fset.count(globalId)) G.fixedDOFIds.push_back(3*i+c+1);
                }
            }
        }
        G.numFixedDOFs=static_cast<int>(G.fixedDOFIds.size());
        G.h_fixedDOFIds=G.fixedDOFIds.data();
    }

    // Honor all explicitly supplied owners first, then assign unclaimed nodes
    // to the first partition containing them. Exactly one owner per global ID.
    for (int l=0;l<nlev;++l) {
        std::unordered_set<int32_t> assigned;
        for (singleGPUData& G : Gs) {
            const Level& L=G.levels[l];
            if (!L.explicitOwnership) continue;
            for (int i=0;i<L.numNodes;++i)
                if (L.ownedMask[i] && !assigned.insert(L.globalNodeIds[i]).second)
                    throw std::runtime_error("A global node has multiple explicit owners.");
        }
        for (singleGPUData& G : Gs) {
            Level& L=G.levels[l];
            if (!L.explicitOwnership) {
                L.ownedMask.resize(L.numNodes);
                for (int i=0;i<L.numNodes;++i)
                    L.ownedMask[i]=uint8_t(assigned.insert(L.globalNodeIds[i]).second);
            }
            L.h_ownedNodeMask=L.ownedMask.data();
        }
        for (const singleGPUData& G: Gs)
            for (int32_t id : G.levels[l].globalNodeIds)
                if (!assigned.count(id))
                    throw std::runtime_error("A global node has no owning partition.");
    }
    {
        std::unordered_set<int32_t> owners;
        for (const auto& G:Gs) {
            const Level& L=G.levels[0];
            for (int i=0;i<L.numNodes;++i)
                if (L.ownedMask[i]) owners.insert(L.globalNodeIds[i]);
        }
        if (owners.size()!=globalFineNodes)
            throw std::runtime_error("Global finest b/y0 contains nodes not owned by a partition.");
    }

    // Cross-check that peer copy mappings refer to the same global node.
    for (int g=0;g<ng;++g) {
        for (int l=0;l<nlev;++l) {
            Level& L=Gs[g].levels[l];
            for (bool right : {false,true}) {
                HaloSide& S=right?L.right:L.left;
                if (S.sendLocal.empty()) continue;
                const Level& dst=Gs[S.neighborGPU].levels[l];
                for (size_t j=0;j<S.sendLocal.size();++j) {
                    const int a=S.sendLocal[j]-1,b=S.sendDestLocal[j]-1;
                    if (b<0 || b>=dst.numNodes ||
                        L.globalNodeIds[a]!=dst.globalNodeIds[b])
                        throw std::runtime_error("Halo source and receiver global IDs disagree.");
                    if (j && S.sendDestLocal[j]!=S.sendDestLocal[j-1]+1)
                        throw std::runtime_error("Halo destination indices must be contiguous.");
                }
            }
        }
        Level& L=Gs[g].levels[0];
        for (bool right : {false,true}) {
            InterfaceSide& S=right?L.rightInterface:L.leftInterface;
            if (S.sendLocal.empty()) continue;
            if (S.neighborGPU<0) throw std::runtime_error("Interface on an external boundary.");
            const Level& dst=Gs[S.neighborGPU].levels[0];
            for (size_t j=0;j<S.sendLocal.size();++j) {
                const int a=S.sendLocal[j]-1,b=S.sendDestLocal[j]-1;
                if (b<0 || b>=dst.numNodes ||
                    L.globalNodeIds[a]!=dst.globalNodeIds[b])
                    throw std::runtime_error("Interface source and destination global IDs disagree.");
            }
        }
    }
    // Reject incomplete halo maps rather than silently computing with stale ghosts.
    for (int l=0;l<nlev;++l) {
        for (int g=0;g+1<ng;++g) {
            const Level& left=Gs[g].levels[l];
            const Level& right=Gs[g+1].levels[l];
            std::unordered_map<int32_t,int> positions;
            for (int j=0;j<right.numNodes;++j)
                positions.emplace(right.globalNodeIds[j],j);
            for (int i=0;i<left.numNodes;++i) {
                const auto it=positions.find(left.globalNodeIds[i]);
                if (it==positions.end()) continue;
                const int j=it->second;
                const bool leftOwner=left.ownedMask[i]!=0;
                const bool rightOwner=right.ownedMask[j]!=0;
                if (leftOwner == rightOwner)
                    throw std::runtime_error("A shared node must have exactly one owner.");
                const HaloSide& send=leftOwner?left.right:right.left;
                const int source=leftOwner?i+1:j+1;
                const int target=leftOwner?j+1:i+1;
                bool found=false;
                for (size_t k=0;k<send.sendLocal.size();++k)
                    if (send.sendLocal[k]==source && send.sendDestLocal[k]==target) {
                        found=true;break;
                    }
                if (!found) throw std::runtime_error(
                    "Missing owner-to-ghost halo entry for a shared global node.");
            }
        }
    }
    for (int g=0;g<ng;++g) {
        Level& L=Gs[g].levels.front();
        if (g>0) L.leftInterface.recvLocal=Gs[g-1].levels.front().rightInterface.sendDestLocal;
        if (g+1<ng) L.rightInterface.recvLocal=Gs[g+1].levels.front().leftInterface.sendDestLocal;
        L.leftInterface.recvCount=int(L.leftInterface.recvLocal.size());
        L.rightInterface.recvCount=int(L.rightInterface.recvLocal.size());
        Gs[g].size=calculateRequiredGPUBytes(Gs[g]);
    }

    // Convert MATLAB sparse CSC to cuDSS CSR of transpose (same for SPD K).
    const mxArray* A=requireField(H,"coarseKFree");
    if (!mxIsSparse(A) || !mxIsDouble(A) || mxIsComplex(A))
        throw std::runtime_error("H.coarseKFree must be real sparse double.");
    const mxArray* freeMx=requireField(H,"coarseFreeDOFIds");
    C.h_freeIdsOwned=readIndexVector(freeMx,INT_MAX,false,"H.coarseFreeDOFIds");
    if (C.h_freeIdsOwned.empty()) throw std::runtime_error("coarseFreeDOFIds cannot be empty.");
    C.numCoarseFreeDOFs=int(C.h_freeIdsOwned.size());
    if (mxGetM(A)!=mwSize(C.numCoarseFreeDOFs) || mxGetN(A)!=mwSize(C.numCoarseFreeDOFs))
        throw std::runtime_error("coarseKFree must be numCoarseFreeDOFs-square.");
    const Level& coarse0=Gs[0].levels.back();
    int maxGlobalCoarse=0;
    for (const auto& G: Gs)
        for (int32_t id: G.levels.back().globalNodeIds)
            maxGlobalCoarse=std::max(maxGlobalCoarse,int(id));
    if (maxGlobalCoarse<=0 || maxGlobalCoarse>INT_MAX/3)
        throw std::runtime_error("Invalid global coarse node count.");
    C.globalNumDOFs=3*maxGlobalCoarse;
    validateOneBasedIds(C.h_freeIdsOwned.data(),C.h_freeIdsOwned.size(),
                        C.globalNumDOFs,false,"coarseFreeDOFIds");
    const mwIndex* jc=mxGetJc(A);
    const mwIndex* ir=mxGetIr(A);
    const mwIndex nnz=jc[mxGetN(A)];
    if (nnz>mwIndex(INT_MAX)) throw std::runtime_error("coarseKFree nnz exceeds int32 range.");
    C.coarseNNZ=int(nnz);
    C.h_rowOffsetsOwned.resize(C.numCoarseFreeDOFs+1);
    C.h_colIndicesOwned.resize(nnz);
    for (int i=0;i<=C.numCoarseFreeDOFs;++i) C.h_rowOffsetsOwned[i]=int32_t(jc[i]);
    for (mwIndex i=0;i<nnz;++i) C.h_colIndicesOwned[i]=int32_t(ir[i]);
    C.h_coarseRowPtr=C.h_rowOffsetsOwned.data();
    C.h_coarseColInd=C.h_colIndicesOwned.data();
    C.h_coarseValues=static_cast<const double*>(mxGetData(A));
    C.localNodeCounts.resize(ng);
    for (int g=0;g<ng;++g) C.localNodeCounts[g]=Gs[g].levels.back().numNodes;
}

static void initializeCoarseSolver(GlobalCoarseSolve& C)
{
    CUDA_CHECK(cudaSetDevice(0));
    const int64_t n=C.numCoarseFreeDOFs;
    CUDSS_CHECK(cudssCreate(&C.cudssHandle));
    CUDSS_CHECK(cudssSetStream(C.cudssHandle, static_cast<cudaStream_t>(0)));
    CUDSS_CHECK(cudssConfigCreate(&C.cudssConfig));
    CUDSS_CHECK(cudssDataCreate(C.cudssHandle,&C.cudssData));
    CUDSS_CHECK(cudssMatrixCreateCsr(&C.cudssA,n,n,C.coarseNNZ,
        C.d_coarseRowPtr,nullptr,C.d_coarseColInd,C.d_coarseValues,
        CUDSS_R_32I,CUDSS_R_32I,CUDSS_R_64F,
        CUDSS_MTYPE_SPD,CUDSS_MVIEW_LOWER,CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&C.cudssB,n,1,n,C.d_coarseRhsFree,
                                    CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&C.cudssX,n,1,n,C.d_coarseXFree,
                                    CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssExecute(C.cudssHandle,CUDSS_PHASE_ANALYSIS,
                             C.cudssConfig,C.cudssData,C.cudssA,C.cudssX,C.cudssB));
    CUDSS_CHECK(cudssExecute(C.cudssHandle,CUDSS_PHASE_FACTORIZATION,
                             C.cudssConfig,C.cudssData,C.cudssA,C.cudssX,C.cudssB));
    CUDA_CHECK(cudaDeviceSynchronize());
    int info=0; size_t sizeWritten=0;
    CUDSS_CHECK(cudssDataGet(C.cudssHandle,C.cudssData,CUDSS_DATA_INFO,
                             &info,sizeof(info),&sizeWritten));
    if (sizeWritten!=sizeof(info) || info!=0)
        throw std::runtime_error("cuDSS coarse factorization failed (INFO="+std::to_string(info)+").");
    C.initialized=true;
}

// C++11-compatible typed CUDA allocation helper.  Avoid generic lambdas
// (lambda parameters declared with auto require C++14 or newer).
template <typename T>
static void allocateDeviceArray(T*& ptr, size_t bytes)
{
    if (bytes != 0)
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), bytes));
}

static void initializeGPUs(std::vector<singleGPUData>& Gs, GlobalCoarseSolve& C)
{
    for (singleGPUData& G : Gs) {
        CUDA_CHECK(cudaSetDevice(G.deviceId));
        CUDA_CHECK(cudaFree(0));
        size_t freeMem=0,totalMem=0;
        CUDA_CHECK(cudaMemGetInfo(&freeMem,&totalMem));
        if (G.size>freeMem)
            throw std::runtime_error("GPU "+std::to_string(G.deviceId)+" insufficient memory for workspace.");
        CUDA_CHECK(cudaMalloc(&G.d_workspace,G.size));
        char* base=static_cast<char*>(G.d_workspace);
        size_t offset=0;
        auto takeD = [&](size_t n)->double* {
            const size_t bytes=checkedMultiply(n,sizeof(double));
            if (offset>G.size || bytes>G.size-offset) throw std::runtime_error("Double workspace overflow.");
            double* p=reinterpret_cast<double*>(base+offset);offset+=bytes;return p;
        };
        auto takeI = [&](size_t n)->int32_t* {
            const size_t bytes=checkedMultiply(n,sizeof(int32_t));
            if (offset>G.size || bytes>G.size-offset) throw std::runtime_error("Int32 workspace overflow.");
            int32_t* p=reinterpret_cast<int32_t*>(base+offset);offset+=bytes;return p;
        };
        auto takeB = [&](size_t n)->uint8_t* {
            if (offset>G.size || n>G.size-offset) throw std::runtime_error("Mask workspace overflow.");
            uint8_t* p=reinterpret_cast<uint8_t*>(base+offset);offset+=n;return p;
        };
        Level& fine=G.levels.front();
        fine.d_eleModulus=takeD(fine.numElements);
        for (int l=0;l+1<G.numLevels;++l) G.levels[l].d_dK=takeD(G.levels[l].numDOFs);
        G.d_y=takeD(fine.numDOFs); G.d_r=takeD(fine.numDOFs);
        G.d_p=takeD(fine.numDOFs); G.d_Ap=takeD(fine.numDOFs);
        fine.d_rhs=G.d_r; fine.d_x=G.d_Ap;
        for (int l=1;l<G.numLevels;++l) {
            G.levels[l].d_rhs=takeD(G.levels[l].numDOFs);
            G.levels[l].d_x=takeD(G.levels[l].numDOFs);
        }
        G.d_dotPartials=takeD(G.partialCount); G.d_dotScalar=takeD(1);
        for (Level& L : G.levels)
            for (HaloSide* side : {&L.left,&L.right})
                if (!side->sendLocal.empty()) side->d_sendBuffer=takeD(3*side->sendLocal.size());
        for (InterfaceSide* side : {&fine.leftInterface,&fine.rightInterface}) {
            if (!side->sendLocal.empty()) side->d_sendBuffer=takeD(3*side->sendLocal.size());
            if (side->recvCount) side->d_recvBuffer=takeD(3*size_t(side->recvCount));
        }
        fine.d_nodeToElements=takeI(8*size_t(fine.numNodes));
        fine.d_eNodMat=takeI(8*size_t(fine.numElements));
        for (Level& L : G.levels) {
            L.d_nodGridId=takeI(L.numNodes);
            L.d_nodMapForward=takeI(L.numGridNodes);
        }
        if (G.numFixedDOFs) G.d_fixedDOFIds=takeI(G.numFixedDOFs);
        for (Level& L : G.levels)
            for (HaloSide* side : {&L.left,&L.right})
                if (!side->sendLocal.empty()) side->d_sendLocal=takeI(side->sendLocal.size());
        for (InterfaceSide* side : {&fine.leftInterface,&fine.rightInterface}) {
            if (!side->sendLocal.empty()) side->d_sendLocal=takeI(side->sendLocal.size());
            if (side->recvCount) side->d_recvLocal=takeI(side->recvCount);
        }
        for (Level& L:G.levels) L.d_ownedNodeMask=takeB(L.numNodes);
        if (offset!=G.size) throw std::runtime_error("GPU workspace size/layout mismatch.");
        auto copy = [](void* dst,const void* src,size_t n) {
            if (n) CUDA_CHECK(cudaMemcpy(dst,src,n,cudaMemcpyHostToDevice));
        };
        copy(fine.d_nodeToElements,fine.h_nodeToElements,8*size_t(fine.numNodes)*sizeof(int32_t));
        copy(fine.d_eNodMat,fine.h_eNodMat,8*size_t(fine.numElements)*sizeof(int32_t));
        copy(fine.d_eleModulus,fine.h_eleModulus,size_t(fine.numElements)*sizeof(double));
        for (int l=0;l<G.numLevels;++l) {
            Level& L=G.levels[l];
            copy(L.d_nodGridId,L.h_nodGridId,size_t(L.numNodes)*sizeof(int32_t));
            copy(L.d_nodMapForward,L.h_nodMapForward,L.numGridNodes*sizeof(int32_t));
            copy(L.d_ownedNodeMask,L.h_ownedNodeMask,size_t(L.numNodes));
            if (l+1<G.numLevels)
                copy(L.d_dK,L.h_dK,size_t(L.numDOFs)*sizeof(double));
            for (HaloSide* side : {&L.left,&L.right})
                copy(side->d_sendLocal,side->sendLocal.data(),side->sendLocal.size()*sizeof(int32_t));
            CUDA_CHECK(cudaMemset(L.d_x,0,size_t(L.numDOFs)*sizeof(double)));
            if (l>0) CUDA_CHECK(cudaMemset(L.d_rhs,0,size_t(L.numDOFs)*sizeof(double)));
        }
        for (InterfaceSide* side : {&fine.leftInterface,&fine.rightInterface}) {
            copy(side->d_sendLocal,side->sendLocal.data(),side->sendLocal.size()*sizeof(int32_t));
            copy(side->d_recvLocal,side->recvLocal.data(),side->recvLocal.size()*sizeof(int32_t));
        }
        copy(G.d_fixedDOFIds,G.h_fixedDOFIds,size_t(G.numFixedDOFs)*sizeof(int32_t));
        copy(G.d_y,G.h_y,size_t(fine.numDOFs)*sizeof(double));
        copy(G.d_r,G.h_b,size_t(fine.numDOFs)*sizeof(double));
        CUDA_CHECK(cudaMemset(G.d_p,0,size_t(fine.numDOFs)*sizeof(double)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_Ke,G.h_Ke,24*24*sizeof(double)));
        CUBLAS_CHECK(cublasCreate(&G.cublasHandle));
        CUBLAS_CHECK(cublasSetPointerMode(G.cublasHandle,CUBLAS_POINTER_MODE_HOST));
        G.initialized=true;
    }
    CUDA_CHECK(cudaSetDevice(0));
    auto put = [](void* dst,const void* src,size_t bytes) {
        if (bytes) CUDA_CHECK(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice));
    };
    allocateDeviceArray(C.d_globalRHS,size_t(C.globalNumDOFs)*sizeof(double));
    allocateDeviceArray(C.d_globalX,size_t(C.globalNumDOFs)*sizeof(double));
    allocateDeviceArray(C.d_coarseRhsFree,size_t(C.numCoarseFreeDOFs)*sizeof(double));
    allocateDeviceArray(C.d_coarseXFree,size_t(C.numCoarseFreeDOFs)*sizeof(double));
    allocateDeviceArray(C.d_coarseFreeDOFIds,size_t(C.numCoarseFreeDOFs)*sizeof(int32_t));
    allocateDeviceArray(C.d_coarseRowPtr,size_t(C.numCoarseFreeDOFs+1)*sizeof(int32_t));
    allocateDeviceArray(C.d_coarseColInd,size_t(C.coarseNNZ)*sizeof(int32_t));
    allocateDeviceArray(C.d_coarseValues,size_t(C.coarseNNZ)*sizeof(double));
    put(C.d_coarseFreeDOFIds,C.h_freeIdsOwned.data(),size_t(C.numCoarseFreeDOFs)*sizeof(int32_t));
    put(C.d_coarseRowPtr,C.h_coarseRowPtr,size_t(C.numCoarseFreeDOFs+1)*sizeof(int32_t));
    put(C.d_coarseColInd,C.h_coarseColInd,size_t(C.coarseNNZ)*sizeof(int32_t));
    put(C.d_coarseValues,C.h_coarseValues,size_t(C.coarseNNZ)*sizeof(double));
    const size_t ng=Gs.size();
    C.d_recvResidual.resize(ng,nullptr);C.d_sendResidual.resize(ng,nullptr);
    C.d_globalNodeIds.resize(ng,nullptr);C.d_ownedMasks.resize(ng,nullptr);
    for (size_t g=0;g<ng;++g) {
        const Level& L=Gs[g].levels.back();
        const size_t n=L.numNodes;
        allocateDeviceArray(C.d_recvResidual[g],3*n*sizeof(double));
        allocateDeviceArray(C.d_sendResidual[g],3*n*sizeof(double));
        allocateDeviceArray(C.d_globalNodeIds[g],n*sizeof(int32_t));
        allocateDeviceArray(C.d_ownedMasks[g],n*sizeof(uint8_t));
        put(C.d_globalNodeIds[g],L.h_globalNodeIds,n*sizeof(int32_t));
        put(C.d_ownedMasks[g],L.h_ownedNodeMask,n*sizeof(uint8_t));
    }
    initializeCoarseSolver(C);
}

static void zeroFixedDOFs(singleGPUData& G, double* d_vector)
{
    if (G.numFixedDOFs == 0) return;
    zeroSelectedDOFsKernel<<<gridSizeFor(G.numFixedDOFs, 256), 256>>>(
        d_vector, G.d_fixedDOFIds, G.numFixedDOFs);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void reduceOwnedPartialsKernel(const double* partials, int count, double* result)
{
    extern __shared__ double scratch[];
    const int tid = threadIdx.x;
    double x = 0.0;
    for (int i = tid; i < count; i += blockDim.x) x += partials[i];
    scratch[tid] = x;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    if (tid == 0) *result = scratch[0];
}

static double globalDotProduct(singleGPUData& G, ncclComm_t comm,
                               cudaStream_t ncclStream,
                               const double* d_a, const double* d_b)
{
    CUDA_CHECK(cudaSetDevice(G.deviceId));
    const Level& fine = G.levels[0];
    ownedDotPartialKernel<<<G.partialCount, 256, 256 * sizeof(double)>>>(
        d_a, d_b, fine.d_ownedNodeMask, G.d_dotPartials, fine.numNodes);
    CUDA_CHECK(cudaGetLastError());
    reduceOwnedPartialsKernel<<<1, 256, 256 * sizeof(double)>>>(
        G.d_dotPartials, G.partialCount, G.d_dotScalar);
    CUDA_CHECK(cudaGetLastError());
    // The collective stream is non-blocking; establish dependency explicitly.
    CUDA_CHECK(cudaDeviceSynchronize());
    NCCL_CHECK(ncclAllReduce(G.d_dotScalar, G.d_dotScalar, 1, ncclDouble,
                            ncclSum, comm, ncclStream));
    double answer = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&answer, G.d_dotScalar, sizeof(double),
                               cudaMemcpyDeviceToHost, ncclStream));
    CUDA_CHECK(cudaStreamSynchronize(ncclStream));
    return answer;
}

static void globalNorm(double& norm, singleGPUData& G, ncclComm_t comm,
                       cudaStream_t ncclStream, const double* d_r)
{
    const double square = globalDotProduct(G, comm, ncclStream, d_r, d_r);
    if (!std::isfinite(square) || square < -1e-12)
        throw std::runtime_error("Invalid global squared residual norm.");
    norm = std::sqrt(std::max(0.0, square));
}

static void solveCoarsest(coarsestLevelData& C, const double* d_rhs, double* d_x)
{
    constexpr int block = 256;
    const int n = C.numCoarseFreeDOFs;
    CUDA_CHECK(cudaMemset(d_x, 0, size_t(C.globalNumDOFs) * sizeof(double)));
    CUDA_CHECK(cudaMemset(C.d_coarseXFree, 0, size_t(n) * sizeof(double)));
    gatherSelectedDOFsKernel<<<gridSizeFor(n, block), block>>>(
        d_rhs, C.d_coarseFreeDOFIds, C.d_coarseRhsFree, n);
    CUDA_CHECK(cudaGetLastError());
    CUDSS_CHECK(cudssExecute(C.cudssHandle, CUDSS_PHASE_SOLVE,
                             C.cudssConfig, C.cudssData, C.cudssA, C.cudssX, C.cudssB));
    scatterSelectedDOFsKernel<<<gridSizeFor(n, block), block>>>(
        C.d_coarseXFree, C.d_coarseFreeDOFIds, d_x, n);
    CUDA_CHECK(cudaGetLastError());
}

static void destroySolver(coarsestLevelData& C) noexcept
{
    if (cudaSetDevice(0) != cudaSuccess) return;
    if (C.cudssA) { (void)cudssMatrixDestroy(C.cudssA); C.cudssA = nullptr; }
    if (C.cudssX) { (void)cudssMatrixDestroy(C.cudssX); C.cudssX = nullptr; }
    if (C.cudssB) { (void)cudssMatrixDestroy(C.cudssB); C.cudssB = nullptr; }
    if (C.cudssData && C.cudssHandle)
        (void)cudssDataDestroy(C.cudssHandle, C.cudssData);
    C.cudssData = nullptr;
    if (C.cudssConfig) (void)cudssConfigDestroy(C.cudssConfig);
    C.cudssConfig = nullptr;
    if (C.cudssHandle) (void)cudssDestroy(C.cudssHandle);
    C.cudssHandle = nullptr;
    auto freeP = [](void*& p) { if (p) (void)cudaFree(p); p = nullptr; };
    freeP(reinterpret_cast<void*&>(C.d_coarseRowPtr));
    freeP(reinterpret_cast<void*&>(C.d_coarseColInd));
    freeP(reinterpret_cast<void*&>(C.d_coarseValues));
    freeP(reinterpret_cast<void*&>(C.d_coarseRhsFree));
    freeP(reinterpret_cast<void*&>(C.d_coarseXFree));
    freeP(reinterpret_cast<void*&>(C.d_coarseFreeDOFIds));
    freeP(reinterpret_cast<void*&>(C.d_globalRHS));
    freeP(reinterpret_cast<void*&>(C.d_globalX));
    for (auto& p : C.d_recvResidual) freeP(reinterpret_cast<void*&>(p));
    for (auto& p : C.d_sendResidual) freeP(reinterpret_cast<void*&>(p));
    for (auto& p : C.d_globalNodeIds) freeP(reinterpret_cast<void*&>(p));
    for (auto& p : C.d_ownedMasks) freeP(reinterpret_cast<void*&>(p));
    C.initialized = false;
}

static void destroySolver(singleGPUData& G) noexcept
{
    if (cudaSetDevice(G.deviceId) != cudaSuccess) return;
    if (G.cublasHandle) (void)cublasDestroy(G.cublasHandle);
    G.cublasHandle = nullptr;
    if (G.d_workspace) (void)cudaFree(G.d_workspace);
    G.d_workspace = nullptr;
    G.initialized = false;
}


static void broadcastCoarseSolution(
    singleGPUData& G,
    GlobalCoarseSolve& C,
    const std::vector<ncclComm_t>& comms)
{
    constexpr int rootGPU = 0;
    constexpr int blockSize = 256;

    const int rank = G.deviceId;
    const int lastLevel = G.numLevels - 1;
    const Level& coarse = G.levels[lastLevel];

    const int localDOFs = 3 * coarse.numNodes;

    // Divide the global coarse solution into each partition's local ordering.
    if (rank == rootGPU)
    {
        for (int g = 0; g < G.numGPUs; ++g)
        {
            const int n = C.localNodeCounts[g];

            if (n == 0)
                continue;

            const int grid = gridSizeFor(n, blockSize);

            scatterGlobalCoarseResidualKernel
                <<<grid, blockSize, 0>>>(
                    C.d_sendResidual[g],
                    C.d_globalX,
                    C.d_globalNodeIds[g],
                    n
                );

            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(0));
    // Copy the global RHS into the local coarse RHS for GPU 0.
    // Broadcast the global coarse solution to all GPUs.
    if (rank == rootGPU)
    {
        // Send GPU 0's local residual.
        if (localDOFs > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                coarse.d_x,
                C.d_sendResidual[rootGPU],
                static_cast<size_t>(localDOFs) * sizeof(double),
                cudaMemcpyDeviceToDevice,
                0
            ));
        }

        // Send the other partitions' residuals.
        ScopedNcclGroup group;

        for (int g = 1; g < G.numGPUs; ++g)
        {
            // Receiving partition's node count.
            const int count = 3 * C.localNodeCounts[g];

            if (count == 0)
                continue;

            NCCL_CHECK(ncclSend(
                C.d_sendResidual[g],
                count,
                ncclDouble,
                g,
                comms[rootGPU],
                0
            ));
        }

        group.finish();
    }
    else if (localDOFs > 0)
    {
        NCCL_CHECK(ncclRecv(
            coarse.d_x,
            localDOFs,
            ncclDouble,
            rootGPU,
            comms[rank],
            0
        ));
    }

    CUDA_CHECK(cudaStreamSynchronize(0));
    workerBarrier();
}

static void assembleGlobalCoarseResidual(
    singleGPUData& G, GlobalCoarseSolve& C,
    const std::vector<ncclComm_t>& comms)
{
    constexpr int rootGPU = 0;
    constexpr int blockSize = 256;

    const int rank = G.deviceId;
    const int lastLevel = G.numLevels - 1;
    const Level& coarse = G.levels[lastLevel];

    const int localDOFs = 3 * coarse.numNodes;

    if (rank == rootGPU)
    {
        // Preserve GPU 0's local residual.
        if (localDOFs > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                C.d_recvResidual[rootGPU],
                coarse.d_rhs,
                static_cast<size_t>(localDOFs) * sizeof(double),
                cudaMemcpyDeviceToDevice,
                0
            ));
        }

        // Receive the other partitions' residuals.
        ScopedNcclGroup group;

        for (int g = 1; g < G.numGPUs; ++g)
        {
            const int count = 3 * C.localNodeCounts[g];

            if (count == 0)
                continue;

            NCCL_CHECK(ncclRecv(
                C.d_recvResidual[g],
                count,
                ncclDouble,
                g,
                comms[rootGPU],
                0
            ));
        }

        group.finish();
    }
    else if (localDOFs > 0)
    {
        NCCL_CHECK(ncclSend(
            coarse.d_rhs,
            localDOFs,
            ncclDouble,
            rootGPU,
            comms[rank],
            0
        ));
    }

    CUDA_CHECK(cudaStreamSynchronize(0));
    workerBarrier();

    if (rank == rootGPU)
    {
        // Reset the global RHS.
        CUDA_CHECK(cudaMemsetAsync(
            C.d_globalRHS,
            0,
            static_cast<size_t>(C.globalNumDOFs) * sizeof(double),
            0
        ));

        // Copy each partition into global ordering.
        for (int g = 0; g < G.numGPUs; ++g)
        {
            const int n = C.localNodeCounts[g];

            if (n == 0)
                continue;

            const int grid = gridSizeFor(n, blockSize);

            gatherGlobalCoarseResidualKernel
                <<<grid, blockSize, 0>>>(
                    C.d_recvResidual[g],
                    C.d_globalRHS,
                    C.d_globalNodeIds[g],
                    C.d_ownedMasks[g],
                    n
                );

            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    workerBarrier();
}

__global__ void packHaloKernel(const double*, const int32_t*, double*, int);
static void copyBoundaryToHalo(int, const double*, int, double*, int, cudaStream_t);

static void sendOneSideHalo(int sourceDeviceId, HaloSide& S,
                            const double* d_src, int neighborDeviceId,
                            double* d_dstNeighbor, cudaStream_t stream)
{
    if (S.sendLocal.empty()) return;
    if (neighborDeviceId < 0 || S.sendLocal.size() != S.sendDestLocal.size())
        throw std::runtime_error("Halo neighbor/index count mismatch.");
    // A peer memcpy can target a contiguous slice only; validate at input time too.
    for (size_t i = 1; i < S.sendDestLocal.size(); ++i)
        if (S.sendDestLocal[i] != S.sendDestLocal[i-1] + 1)
            throw std::runtime_error("Halo destination indices must be contiguous.");
    const int count = static_cast<int>(S.sendLocal.size());
    packHaloKernel<<<gridSizeFor(count, 256), 256, 0, stream>>>(
        d_src, S.d_sendLocal, S.d_sendBuffer, count);
    CUDA_CHECK(cudaGetLastError());
    const size_t offset = 3 * size_t(S.sendDestLocal[0] - 1);
    copyBoundaryToHalo(sourceDeviceId, S.d_sendBuffer, neighborDeviceId,
                       d_dstNeighbor + offset, count, stream);
}


static void applyVcycle(
    singleGPUData& G,
    const double* d_fineResidual,
    double* d_fineCorrection,
    std::vector<singleGPUData>& gpuData,
    StreamData& streams,
    std::vector<ncclComm_t>& ncclComms,
    GlobalCoarseSolve& globalCoarse,
    VcycleShared& scratch)
{
    constexpr int blockSize = 256;
    const int lastLevel = G.numLevels - 1;
    Level& finest = G.levels[0];

    if (finest.d_rhs != d_fineResidual)
    {
        CUDA_CHECK(cudaMemcpy(
            finest.d_rhs,
            d_fineResidual,
            static_cast<size_t>(finest.numDOFs) * sizeof(double),
            cudaMemcpyDeviceToDevice));
    }

    // Exchange finest-level residual halos before the first restriction.

    CUDA_CHECK(cudaDeviceSynchronize());
    workerBarrier();

    if (finest.right.neighborGPU >= 0)
    {
        const int neighbor = finest.right.neighborGPU;

        sendOneSideHalo(
            G.deviceId,
            finest.right,
            finest.d_rhs,
            neighbor,
            gpuData[neighbor].levels[0].d_rhs,
            streams.rightStream
        );
    }

    if (finest.left.neighborGPU >= 0)
    {
        const int neighbor = finest.left.neighborGPU;

        sendOneSideHalo(
            G.deviceId,
            finest.left,
            finest.d_rhs,
            neighbor,
            gpuData[neighbor].levels[0].d_rhs,
            streams.leftStream
        );
    }

    if (finest.right.neighborGPU >= 0)
        CUDA_CHECK(cudaStreamSynchronize(streams.rightStream));

    if (finest.left.neighborGPU >= 0)
        CUDA_CHECK(cudaStreamSynchronize(streams.leftStream));

    workerBarrier();

    // Initialize all level corrections.
    for (int levelIndex = 0; levelIndex < G.numLevels; ++levelIndex)
    {
        Level& level = G.levels[levelIndex];

        CUDA_CHECK(cudaMemset(
            level.d_x,
            0,
            static_cast<size_t>(level.numDOFs) * sizeof(double)));
    }

    // Restriction
    for (int fineIndex = 0; fineIndex < lastLevel; ++fineIndex)
    {
        Level& fine = G.levels[fineIndex];
        Level& coarse = G.levels[fineIndex + 1];

        // One Jacobi smoothing kernel thread per DOF.
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
                G.jacobiOmega,
                fine.numDOFs);
        }
        else
        {
            dampedJacobiSmootherKernelCoarse<<<smoothGrid, blockSize>>>(
                fine.d_rhs,
                fine.d_dK,
                fine.d_x,
                G.jacobiOmega,
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
            G.spanWidths[fineIndex]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        workerBarrier();

        // Send coarse residual to right GPU.
        if (coarse.right.neighborGPU >= 0)
        {
            const int neighbor = coarse.right.neighborGPU;

            sendOneSideHalo(
                G.deviceId,
                coarse.right,
                coarse.d_rhs,
                neighbor,
                gpuData[neighbor].levels[fineIndex + 1].d_rhs,
                streams.rightStream

            );
        }

        // Send coarse residual to left GPU.
        if (coarse.left.neighborGPU >= 0)
        {
            const int neighbor = coarse.left.neighborGPU;

            sendOneSideHalo(
                G.deviceId,
                coarse.left,
                coarse.d_rhs,
                neighbor,
                gpuData[neighbor].levels[fineIndex + 1].d_rhs,
                streams.leftStream
            );
        }

        // Wait for both outgoing transfers.
        if (coarse.right.neighborGPU >= 0)
        {
            CUDA_CHECK(cudaStreamSynchronize(streams.rightStream));
        }

        if (coarse.left.neighborGPU >= 0)
        {
            CUDA_CHECK(cudaStreamSynchronize(streams.leftStream));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        workerBarrier();

    }
    // Assemble global coarse residual into global coarse residual on GPU 0.
    // All GPUs participate.
    assembleGlobalCoarseResidual(
        G, globalCoarse,
        ncclComms
    );
    // Solve the coarsest level on GPU 0 only.
    if (G.deviceId == 0)
    {
        solveCoarsest(
            globalCoarse,
            globalCoarse.d_globalRHS,
            globalCoarse.d_globalX
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    // All OpenMP threads wait here.
    workerBarrier();

    // Broadcast the coarse solution from GPU 0 to other GPUs.
    // Ensure the broadcast is complete before interpolation.
    broadcastCoarseSolution(
        G,
        globalCoarse,
        ncclComms
    );

    CUDA_CHECK(cudaDeviceSynchronize());
    workerBarrier();

    // Interpolation and post-smoothing
    for (int coarseIndex = lastLevel; coarseIndex >= 1; --coarseIndex)
    {
        const int fineIndex = coarseIndex - 1;

        Level& fine = G.levels[fineIndex];
        Level& coarse = G.levels[coarseIndex];

        // Ensure the coarse correction is ready.
        CUDA_CHECK(cudaDeviceSynchronize());

        // ----------------------------------------
        // Exchange coarse correction halos
        // ----------------------------------------

        if (coarse.right.neighborGPU >= 0)
        {
            const int neighbor = coarse.right.neighborGPU;

            sendOneSideHalo(
                G.deviceId,
                coarse.right,
                coarse.d_x,
                neighbor,
                gpuData[neighbor].levels[coarseIndex].d_x,
                streams.rightStream
            );
        }

        if (coarse.left.neighborGPU >= 0)
        {
            const int neighbor = coarse.left.neighborGPU;

            sendOneSideHalo(
                G.deviceId,
                coarse.left,
                coarse.d_x,
                neighbor,
                gpuData[neighbor].levels[coarseIndex].d_x,
                streams.leftStream
            );
        }

        if (coarse.right.neighborGPU >= 0)
        {
            CUDA_CHECK(cudaStreamSynchronize(
                streams.rightStream));
        }

        if (coarse.left.neighborGPU >= 0)
        {
            CUDA_CHECK(cudaStreamSynchronize(
                streams.leftStream));
        }

        // All GPUs have finished sending halo data.
        workerBarrier();

        // ----------------------------------------
        // Interpolation: fine.d_x += P * coarse.d_x
        // ----------------------------------------

        const int interpolationGrid = gridSizeFor(fine.numNodes, blockSize);

        interpolateResidualKernel<<<interpolationGrid, blockSize>>>(
            fine.d_nodGridId,
            coarse.d_nodMapForward,
            coarse.d_x,
            fine.d_x,
            fine.numNodes,
            coarse.nx,
            coarse.ny,
            coarse.nz,
            fine.nx,
            fine.ny,
            fine.nz,
            G.spanWidths[fineIndex]
        );

        CUDA_CHECK(cudaGetLastError());

        // ----------------------------------------
        // Post-smoothing
        // ----------------------------------------

        const int dofGrid = gridSizeFor(fine.numDOFs, blockSize);

        addDampedJacobiKernel<<<dofGrid, blockSize>>>(
            fine.d_rhs,
            fine.d_dK,
            fine.d_x,
            G.jacobiOmega,
            fine.numDOFs
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDeviceSynchronize());

        // All GPUs finish this level before proceeding to the next finer level.
        workerBarrier();
    }
    // Local fixed-DOF indices are constructed at initialization; ghosts are
    // retained in the local correction but excluded from global scalar products.
    zeroFixedDOFs(G, finest.d_x);

    if (d_fineCorrection != finest.d_x)
    {
        CUDA_CHECK(cudaMemcpy(
            d_fineCorrection,
            finest.d_x,
            static_cast<size_t>(finest.numDOFs)
                * sizeof(double),
            cudaMemcpyDeviceToDevice
        ));
    }
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

__global__ void packHaloKernel(const double* localResidual, const int32_t* sendLocal,
                               double* sendBuffer, int numSendNodes)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= numSendNodes) return;
    const int source = 3 * (sendLocal[node] - 1); // MATLAB 1-based
    const int target = 3 * node;
    sendBuffer[target]     = localResidual[source];
    sendBuffer[target + 1] = localResidual[source + 1];
    sendBuffer[target + 2] = localResidual[source + 2];
}

__global__ void addReceivedInterfaceKernel(
    double* d_Ax,
    const double* d_recvBuffer,
    const int32_t* d_recvLocal,
    int numNodes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numNodes) return;

    int dst = 3 * (d_recvLocal[i] - 1);
    int src = 3 * i;

    d_Ax[dst]     += d_recvBuffer[src];
    d_Ax[dst + 1] += d_recvBuffer[src + 1];
    d_Ax[dst + 2] += d_recvBuffer[src + 2];
}

static void copyBoundaryToHalo(
    int sourceId,
    const double* d_sourceLocal,
    int targetId,
    double* d_recvLocal,
    int numBoundaryNodes,
    cudaStream_t stream)
{
    if (numBoundaryNodes <= 0)
        return;

    CUDA_CHECK(cudaSetDevice(sourceId));

    // Copy boundary data to halo region on the same GPU
    CUDA_CHECK(cudaMemcpyPeerAsync(
        d_recvLocal,
        targetId,
        d_sourceLocal,
        sourceId,
        static_cast<size_t>(numBoundaryNodes) * 3 * sizeof(double),
        stream));
}

static void sendOneSideInterface(
    singleGPUData& G,
    InterfaceSide& S,
    const double* d_Ax,
    cudaStream_t stream,
    double* receiveBuffer)
{
    if (S.sendLocal.empty())
        return;

    if (S.neighborGPU < 0)
        throw std::runtime_error("Invalid interface neighbor");

    CUDA_CHECK(cudaSetDevice(G.deviceId));

    const int count =
        static_cast<int>(S.sendLocal.size());

    constexpr int block = 256;
    const int grid = gridSizeFor(count, block);

    packHaloKernel<<<grid, block, 0, stream>>>(
        d_Ax,
        S.d_sendLocal,
        S.d_sendBuffer,
        count
    );

    CUDA_CHECK(cudaGetLastError());

    copyBoundaryToHalo(
        G.deviceId,
        S.d_sendBuffer,
        S.neighborGPU,
        receiveBuffer,
        count,
        stream
    );
}

static void applyFinestOperator(
    singleGPUData& G,
    const double* d_x,
    double* d_Ax,
    StreamData& streams,
    const std::vector<singleGPUData>& gpuData)
{
    constexpr int blockSize = 256;

    Level& finest = G.levels[0];

    // Before SpMV, exchange x ghost values. Use matching neighbor vector.
    CUDA_CHECK(cudaDeviceSynchronize());
    workerBarrier();
    for (const bool right : {true, false}) {
        HaloSide& side = right ? finest.right : finest.left;
        if (side.neighborGPU < 0 || side.sendLocal.empty()) continue;
        const singleGPUData& neighbor = gpuData[side.neighborGPU];
        double* dst = (d_x == G.d_y) ? neighbor.d_y : neighbor.d_p;
        sendOneSideHalo(G.deviceId, side, d_x, side.neighborGPU, dst,
                            right ? streams.rightStream : streams.leftStream);
    }
    if (finest.right.neighborGPU >= 0) CUDA_CHECK(cudaStreamSynchronize(streams.rightStream));
    if (finest.left.neighborGPU >= 0) CUDA_CHECK(cudaStreamSynchronize(streams.leftStream));
    workerBarrier();

    const int gridSize = gridSizeFor(finest.numNodes, blockSize);

    kbyu_kernel<<<gridSize, blockSize>>>(
        d_x,
        d_Ax,
        finest.d_nodeToElements,
        finest.d_eNodMat,
        finest.d_eleModulus,
        finest.numNodes,
        finest.numElements
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Send to the right neighbor.
    if (finest.rightInterface.neighborGPU >= 0)
    {
        const int neighbor = finest.rightInterface.neighborGPU;

        sendOneSideInterface(
            G,
            finest.rightInterface,
            d_Ax,
            streams.rightStream,
            gpuData[neighbor].levels[0] .leftInterface.d_recvBuffer
        );
    }

    // Send to the left neighbor.
    if (finest.leftInterface.neighborGPU >= 0)
    {
        const int neighbor = finest.leftInterface.neighborGPU;

        sendOneSideInterface(
            G,
            finest.leftInterface,
            d_Ax,
            streams.leftStream,
            gpuData[neighbor].levels[0].rightInterface.d_recvBuffer
        );
    }

    if (finest.rightInterface.neighborGPU >= 0 &&
        streams.rightStream != nullptr)
    {
        CUDA_CHECK(cudaStreamSynchronize(streams.rightStream));
    }

    if (finest.leftInterface.neighborGPU >= 0 && streams.leftStream != nullptr)
    {
        CUDA_CHECK(cudaStreamSynchronize(streams.leftStream));
    }

    // Every thread has finished its outgoing copies.
    workerBarrier();

    InterfaceSide& left  = finest.leftInterface;
    InterfaceSide& right = finest.rightInterface;

    // Receive contributions from LEFT neighbor
    if (left.recvCount > 0)
    {
        const int grid = gridSizeFor(left.recvCount, blockSize);

        addReceivedInterfaceKernel
            <<<grid, blockSize>>>(
                d_Ax,
                left.d_recvBuffer,
                left.d_recvLocal,
                left.recvCount
            );

        CUDA_CHECK(cudaGetLastError());
    }

    // Receive contributions from RIGHT neighbor
    if (right.recvCount > 0)
    {
        const int grid =
            gridSizeFor(right.recvCount, blockSize);

        addReceivedInterfaceKernel
            <<<grid, blockSize>>>(
                d_Ax,
                right.d_recvBuffer,
                right.d_recvLocal,
                right.recvCount
            );

        CUDA_CHECK(cudaGetLastError());
    }

    // Wait until both accumulations are finished
    if (left.recvCount > 0)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (right.recvCount > 0)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Apply fixed DOF constraints to the result vector.
    zeroFixedDOFs(G, d_Ax);
    CUDA_CHECK(cudaDeviceSynchronize());
    workerBarrier();
}



// Own only resources instantiated inside runMGPCG. Existing solver buffers
// and cuDSS data remain under their existing ownership/cleanup contract.
class ScopedCommunicationStreams
{
    std::vector<singleGPUData>& gpuData_;
public:
    explicit ScopedCommunicationStreams(std::vector<singleGPUData>& gpuData)
        : gpuData_(gpuData) {}

    void cleanup() noexcept
    {
        for (auto& gpu : gpuData_)
        {
            if (cudaSetDevice(gpu.deviceId) != cudaSuccess)
                continue;
            StreamData& streams = gpu.streams;
            if (streams.leftStream) {
                (void)cudaStreamDestroy(streams.leftStream);
                streams.leftStream = nullptr;
            }
            if (streams.rightStream) {
                (void)cudaStreamDestroy(streams.rightStream);
                streams.rightStream = nullptr;
            }
            if (streams.ncclStream) {
                (void)cudaStreamDestroy(streams.ncclStream);
                streams.ncclStream = nullptr;
            }
            if (streams.computeStream) {
                (void)cudaStreamDestroy(streams.computeStream);
                streams.computeStream = nullptr;
            }
        }
    }

    void cleanupChecked()
    {
        for (auto& gpu : gpuData_)
        {
            CUDA_CHECK(cudaSetDevice(gpu.deviceId));
            StreamData& streams = gpu.streams;
            cudaStream_t* entries[] = {
                &streams.leftStream, &streams.rightStream,
                &streams.ncclStream, &streams.computeStream
            };
            for (cudaStream_t* entry : entries)
            {
                if (!*entry) continue;
                cudaStream_t temporary = *entry;
                *entry = nullptr;
                CUDA_CHECK(cudaStreamDestroy(temporary));
            }
        }
    }

    ~ScopedCommunicationStreams() noexcept { cleanup(); }
    ScopedCommunicationStreams(const ScopedCommunicationStreams&) = delete;
    ScopedCommunicationStreams& operator=(const ScopedCommunicationStreams&) = delete;
};

class ScopedNcclCommunicators
{
    std::vector<ncclComm_t>& comms_;
    std::atomic<bool> aborted_{false};
public:
    explicit ScopedNcclCommunicators(std::vector<ncclComm_t>& comms)
        : comms_(comms) {}

    // Called only after a worker has reported an error. Do not mutate entries
    // here: another worker can still be using its communicator concurrently.
    void abortAll() noexcept
    {
        if (aborted_.exchange(true, std::memory_order_acq_rel)) return;
        for (ncclComm_t comm : comms_)
            if (comm) (void)ncclCommAbort(comm);
    }

    void destroyChecked()
    {
        if (aborted_.load(std::memory_order_acquire)) return;
        for (ncclComm_t& comm : comms_)
        {
            if (!comm) continue;
            ncclComm_t tmp = comm;
            comm = nullptr;
            NCCL_CHECK(ncclCommDestroy(tmp));
        }
    }

    ~ScopedNcclCommunicators() noexcept
    {
        if (!aborted_.load(std::memory_order_acquire))
            abortAll();
    }
    ScopedNcclCommunicators(const ScopedNcclCommunicators&) = delete;
    ScopedNcclCommunicators& operator=(const ScopedNcclCommunicators&) = delete;
};

static void runMGPCG(
    std::vector<singleGPUData>& gpuData,
    GlobalCoarseSolve& globalCoarse,
    VcycleShared& vcycleShared,
    int numGPUs)
{
    std::vector<ncclComm_t> comms(numGPUs, nullptr);
    ScopedCommunicationStreams streamsOwner(gpuData);
    ScopedNcclCommunicators communicatorsOwner(comms);
    FailureAwareBarrier failureBarrier(numGPUs);

    const int cpu_cores = omp_get_num_procs();
    mexPrintf("CPU logical threads available: %d\n", cpu_cores);

    omp_set_dynamic(0);
    omp_set_num_threads(numGPUs);

    for (auto& s : gpuData)
    {
        if (!s.initialized || s.levels.empty())
        {
            throw std::runtime_error(
                "The solver context is not fully initialized.");
        }

        if (s.deviceId < 0 || s.deviceId >= numGPUs)
        {
            throw std::runtime_error(
                "Invalid device ID in solver context.");
        }

        if (s.cublasHandle == nullptr)
        {
            throw std::runtime_error(
                "CUBLAS handle is not initialized.");
        }
    }

    // Initialize NCCL communicators.
    std::vector<int> deviceIds(numGPUs);

    for (int i = 0; i < numGPUs; ++i)
        deviceIds[i] = gpuData[i].deviceId;

    NCCL_CHECK(ncclCommInitAll(
        comms.data(),
        numGPUs,
        deviceIds.data()));

    // Communication streams are scoped to runMGPCG for exception-safe cleanup.
    for (int i = 0; i < numGPUs; ++i)
    {
        CUDA_CHECK(cudaSetDevice(gpuData[i].deviceId));

        int leastPriority = 0;
        int greatestPriority = 0;

        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(
            &leastPriority,
            &greatestPriority));

        const int priority = greatestPriority;

        if (i > 0)
        {
            CUDA_CHECK(cudaStreamCreateWithPriority(
                &gpuData[i].streams.leftStream,
                cudaStreamNonBlocking,
                priority));
        }

        if (i < numGPUs - 1)
        {
            CUDA_CHECK(cudaStreamCreateWithPriority(
                &gpuData[i].streams.rightStream,
                cudaStreamNonBlocking,
                priority));
        }

        CUDA_CHECK(cudaStreamCreateWithPriority(
            &gpuData[i].streams.ncclStream,
            cudaStreamNonBlocking,
            priority));

        CUDA_CHECK(cudaStreamCreateWithPriority(
            &gpuData[i].streams.computeStream,
            cudaStreamNonBlocking,
            priority));

    }

    const double one = 1.0;
    const double minusOne = -1.0;

    {
    ScopedNvtxRange mainLoopRange("MGPCG:mainLoop");

    // One CPU thread per GPU.

    #pragma omp parallel num_threads(numGPUs)
    {
        const int g = omp_get_thread_num();
        ScopedWorkerBarrier workerGuard(failureBarrier);
        try
        {

        singleGPUData& solver = gpuData[g];

        CUDA_CHECK(cudaSetDevice(solver.deviceId));

        const int n = solver.levels[0].numDOFs;
        cublasHandle_t handle = solver.cublasHandle;

        solver.iterations = 0;
        solver.relativeResidual = 0.0;
        solver.converged = false;

        zeroFixedDOFs(solver, solver.d_r);
        zeroFixedDOFs(solver, solver.d_y);

        const ncclComm_t comm = comms[g];
        StreamData& streams = solver.streams;
    
        // Ensure constraints are applied before reductions.
        CUDA_CHECK(cudaDeviceSynchronize());

        // -------------------------------------------------
        // Initial global norm
        // -------------------------------------------------

        double normB = 0.0;

        globalNorm(
            normB,
            solver,
            comm,
            streams.ncclStream,
            solver.d_r
        );

        // r = b - A*y

        applyFinestOperator(
            solver,
            solver.d_y,
            solver.d_Ap,
            streams,
            gpuData
        );

        // Local operation
        CUBLAS_CHECK(
            cublasDaxpy(
                handle,
                n,
                &minusOne,
                solver.d_Ap,
                1,
                solver.d_r,
                1));

        zeroFixedDOFs(solver, solver.d_r);

        // Calculate the initial residual norm

        double residualNorm = 0.0;

        globalNorm(
            residualNorm,
            solver,
            comm,
            streams.ncclStream,
            solver.d_r
        );

        const double residualDenominator =
            (normB > 0.0) ? normB : 1.0;

        solver.relativeResidual =
            residualNorm / residualDenominator;

        if (!std::isfinite(solver.relativeResidual))
        {
            throw std::runtime_error(
                "The initial relative residual is not finite.");
        }

        if (solver.relativeResidual < solver.tolerance ||
            residualNorm == 0.0)
        {
            solver.converged = true;
        }

        // Initial preconditioning:
        // z = M^-1*r
        // p = z
        // rho = z'*r

        double rho = 0.0;

        // All workers make the same initial convergence decision because residualNorm is global.
        if (!solver.converged)
        {
            applyVcycle(
                solver,
                solver.d_r,
                solver.d_Ap,
                gpuData,
                streams,
                comms,
                globalCoarse,
                vcycleShared
            );

            // Local operation: p = z
            CUBLAS_CHECK(
                cublasDcopy(
                    handle,
                    n,
                    solver.d_Ap,
                    1,
                    solver.d_p,
                    1));

            // Global rho = z'*r
            rho = globalDotProduct(
                solver,
                comm,
                streams.ncclStream,
                solver.d_Ap,
                solver.d_r
            );

            if (!std::isfinite(rho) || rho <= 0.0)
            {
                throw std::runtime_error(
                    "PCG breakdown: initial z'*r is "
                    "non-positive or non-finite.");
            }
        }

        // Main PCG iteration loop

        for (int its = 1;
             !solver.converged && its <= solver.maxIterations;
             ++its)
        {
            // Ap = A*p

            applyFinestOperator(
                solver,
                solver.d_p,
                solver.d_Ap,
                streams,
                gpuData
            );

            // Global pAp = p'*A*p
            double pAp = globalDotProduct(
                solver,
                comm,
                streams.ncclStream,
                solver.d_p,
                solver.d_Ap
            );

            if (!std::isfinite(pAp) || pAp <= 0.0)
            {
                char message[512];

                std::snprintf(
                    message,
                    sizeof(message),
                    "PCG breakdown at iteration %d: "
                    "p'*A*p is non-positive or non-finite.",
                    its);

                throw std::runtime_error(message);
            }

            const double alpha = rho / pAp;

            if (!std::isfinite(alpha))
            {
                throw std::runtime_error(
                    "PCG breakdown: alpha is not finite.");
            }

            const double minusAlpha = -alpha;

            // y = y + alpha*p

            CUBLAS_CHECK(
                cublasDaxpy(
                    handle,
                    n,
                    &alpha,
                    solver.d_p,
                    1,
                    solver.d_y,
                    1));

            // r = r - alpha*Ap

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

            // Global residual norm

            globalNorm(
                residualNorm,
                solver,
                comm,
                streams.ncclStream,
                solver.d_r
            );

            solver.iterations = its;

            solver.relativeResidual = residualNorm / residualDenominator;

            if (!std::isfinite(solver.relativeResidual))
            {
                throw std::runtime_error(
                    "PCG produced a non-finite "
                    "relative residual.");
            }

            if (solver.relativeResidual < solver.tolerance)
            {
                solver.converged = true;
                break;
            }

            // z = M^-1*r

            {
            ScopedNvtxRange vcycleRange("MGPCG:applyVcycle2");

            applyVcycle(
                solver,
                solver.d_r,
                solver.d_Ap,
                gpuData,
                streams,
                comms,
                globalCoarse,
                vcycleShared
            );

            }

            // Global rhoNew = z'*r

            double rhoNew = globalDotProduct(
                solver,
                comm,
                streams.ncclStream,
                solver.d_Ap,
                solver.d_r
            );

            if (!std::isfinite(rhoNew) || rhoNew <= 0.0)
            {
                char message[512];

                std::snprintf(
                    message,
                    sizeof(message),
                    "PCG breakdown at iteration %d: "
                    "z'*r is non-positive or non-finite.",
                    its);

                throw std::runtime_error(message);
            }

            const double beta = rhoNew / rho;

            if (!std::isfinite(beta))
            {
                throw std::runtime_error(
                    "PCG breakdown: beta is not finite.");
            }

            // p = z + beta*p

            // Local operation: p = beta*p
            CUBLAS_CHECK(
                cublasDscal(
                    handle,
                    n,
                    &beta,
                    solver.d_p,
                    1));

            // Local operation: p = z + p
            CUBLAS_CHECK(
                cublasDaxpy(
                    handle,
                    n,
                    &one,
                    solver.d_Ap,
                    1,
                    solver.d_p,
                    1));

            rho = rhoNew;
        }
        }
        catch (...)
        {
            failureBarrier.fail(std::current_exception());
            // Failure path only: release peers potentially waiting on NCCL.
            communicatorsOwner.abortAll();
        }
    }

    } // End NVTX main-loop range.

    if (std::exception_ptr failure = failureBarrier.error())
        std::rethrow_exception(failure);

    // Every GPU should have the same convergence state.
    const singleGPUData& result = gpuData[0];

    if (result.converged)
    {
        MEX_PRINT(
            "CG solver converged at iteration %d "
            "with relative residual %.6e",
            result.iterations,
            result.relativeResidual);
    }
    else if (result.iterations == result.maxIterations)
    {
        mexWarnMsgIdAndTxt(
            "mgpcg_gpu:maxIterations",
            "Exceeded the maximum iteration count. "
            "Final relative residual: %.6e.",
            result.relativeResidual);
    }

    // Success path. On exceptions, RAII aborts communicators and destroys streams.
    communicatorsOwner.destroyChecked();
    streamsOwner.cleanupChecked();
}

static void createMATLABOutputs(const std::vector<singleGPUData>& Gs,
                                const mxArray* bMx, int nlhs, mxArray* plhs[])
{
    plhs[0]=mxCreateDoubleMatrix(mxGetM(bMx),mxGetN(bMx),mxREAL);
    if (!plhs[0]) throw std::runtime_error("Could not allocate MATLAB output.");
    double* out=static_cast<double*>(mxGetData(plhs[0]));
    const size_t globalDOFs=mxGetNumberOfElements(bMx);
    std::fill(out,out+globalDOFs,0.0);
    for (const singleGPUData& G: Gs) {
        CUDA_CHECK(cudaSetDevice(G.deviceId));
        const Level& fine=G.levels.front();
        std::vector<double> local(fine.numDOFs);
        CUDA_CHECK(cudaMemcpy(local.data(),G.d_y,size_t(fine.numDOFs)*sizeof(double),
                              cudaMemcpyDeviceToHost));
        for (int i=0;i<fine.numNodes;++i) {
            if (!fine.ownedMask[i]) continue;
            const size_t to=3*size_t(fine.globalNodeIds[i]-1), from=3*size_t(i);
            for (int c=0;c<3;++c) out[to+c]=local[from+c];
        }
    }
    if (nlhs>=2) plhs[1]=mxCreateDoubleScalar(double(Gs[0].iterations));
    if (nlhs>=3) plhs[2]=mxCreateDoubleScalar(Gs[0].relativeResidual);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    for (int i=0;i<nlhs;++i) plhs[i]=nullptr;
    if (nrhs!=5)
        mexErrMsgIdAndTxt("mgpcg_gpu:nrhs",
            "Usage: [y, iterations, relativeResidual] = mgpcg_multigpu(b, tolerance, maxIterations, y0, H)");
    if (nlhs<1 || nlhs>3)
        mexErrMsgIdAndTxt("mgpcg_gpu:nlhs","One to three outputs are supported.");
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetNumberOfElements(prhs[1])!=1 ||
        !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetNumberOfElements(prhs[2])!=1)
        mexErrMsgIdAndTxt("mgpcg_gpu:input","Tolerance and maxIterations must be real double scalars.");
    const double tol=mxGetScalar(prhs[1]);
    const double itValue=mxGetScalar(prhs[2]);
    if (!(tol>0.0) || !std::isfinite(tol) || !std::isfinite(itValue) ||
        itValue<1 || itValue>double(INT_MAX) || std::floor(itValue)!=itValue)
        mexErrMsgIdAndTxt("mgpcg_gpu:input","Invalid tolerance or maxIterations.");
    std::string failure;
    {
        std::vector<singleGPUData> Gs;
        GlobalCoarseSolve C;
        VcycleShared scratch;
        try {
            initializeGPUHierarchy(Gs,C,prhs[4],prhs[0],prhs[3],tol,int(itValue));
            initializeGPUs(Gs,C);
            runMGPCG(Gs,C,scratch,int(Gs.size()));
            createMATLABOutputs(Gs,prhs[0],nlhs,plhs);
        } catch (const std::exception& e) {
            failure=e.what();
        } catch (...) {
            failure="Unexpected C++ exception in multi-GPU MGPCG solver.";
        }
        // Destroy coarse cuDSS first; everything it references lives on GPU 0.
        destroySolver(C);
        for (singleGPUData& G:Gs) destroySolver(G);
    }
    if (!failure.empty()) {
        for (int i=0;i<nlhs;++i)
            if (plhs[i]) { mxDestroyArray(plhs[i]); plhs[i]=nullptr; }
        mexErrMsgIdAndTxt("mgpcg_gpu:runtime","%s",failure.c_str());
    }
}
