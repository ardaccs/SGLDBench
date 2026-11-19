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
    double* d_coarseFreeDOFIds = nullptr;
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
    coarsetLevelData coarsestLevel;
};
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
    int numLocalNodes)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numLocalNodes)
        return;
    
    const int src = 3 * i;
    const int dst = 3 * (globalNodeIds[i] - 1);
    // Todo->The coarsest-level assembly requires a clear ownership rule: 
    //If two GPU partitions supply values for the same global coarse node, 
    //one overwrites the other. That is correct only if you have already assigned 
    //ownership and each global node is written exactly once.
    globalResidual[dst]     = localResidual[src];
    globalResidual[dst + 1] = localResidual[src + 1];
    globalResidual[dst + 2] = localResidual[src + 2];
}

__global__ void scatterGlobalCoarseResidualKernel(
    const double* __restrict__ sendBuffer,
    double* __restrict__ globalResidual,
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
    addBytes(bytes, static_cast<size_t>(solver.levels[0].numElements), sizeof(double)); // finest eleModulus

    for (int levelIndex = 0; levelIndex < solver.numLevels - 1; ++levelIndex)
    {
        addBytes(bytes, static_cast<size_t>(solver.levels[levelIndex].numDOFs), sizeof(double)); // diagK
    }

    const size_t finestDOFs = static_cast<size_t>(solver.levels[0].numDOFs);

    addBytes(bytes, 4 * finestDOFs, sizeof(double)); // y, r, p, Ap
    addBytes(bytes, 2 * static_cast<size_t>(solver.numCoarseFreeDOFs), sizeof(double)); // reduced coarse RHS and x
    addBytes(bytes, static_cast<size_t>(solver.coarseNNZ), sizeof(double)); // coarse matrix values

    // Finest rhs aliases d_r and finest x aliases d_Ap.
    // Therefore separate rhs/x storage is needed only for levels 1...N-1.
    for (int levelIndex = 1; levelIndex < solver.numLevels; ++levelIndex)
    {
        addBytes(
            bytes,
            2 * static_cast<size_t>(solver.levels[levelIndex].numDOFs),
            sizeof(double));
    }

    const Level& finest = solver.levels[0];

    // eNodMat and nodeToElements.
    addBytes(
        bytes,
        static_cast<size_t>(finest.numNodes) * 8,
        sizeof(int32_t));

    addBytes(
        bytes,
        static_cast<size_t>(finest.numElements) * 8,
        sizeof(int32_t));

    // Integer arrays follow all doubles.
    for (int levelIndex = 0; levelIndex < solver.numLevels; ++levelIndex)
    {
        const Level& level = solver.levels[levelIndex];

        addBytes(bytes, static_cast<size_t>(level.numNodes), sizeof(int32_t));
        addBytes(bytes, level.numGridNodes, sizeof(int32_t));
    }

    addBytes(bytes, static_cast<size_t>(solver.numFixedDOFs), sizeof(int32_t));
    addBytes(bytes, static_cast<size_t>(solver.numCoarseFreeDOFs), sizeof(int32_t));
    addBytes(bytes, static_cast<size_t>(solver.numCoarseFreeDOFs) + 1, sizeof(int32_t));
    addBytes(bytes, static_cast<size_t>(solver.coarseNNZ), sizeof(int32_t));

    return bytes;
}

static void initializeGPUHierarchy()
{
}

static void initializeCoarseSolver(
    SolverContext& solver);

static void initializeGPUs()
{
}

static void zeroFixedDOFs(
    singleGPUData& G,
    double* d_vector)
{
    if (G.numFixedDOFs == 0)
        return;

    constexpr int blockSize = 256;

    const int gridSize =
        gridSizeFor(G.numFixedDOFs, blockSize);

    zeroSelectedDOFsKernel<<<gridSize, blockSize>>>(
        d_vector,
        G.d_fixedDOFIds,
        G.numFixedDOFs
    );

    CUDA_CHECK(cudaGetLastError());
}


static double globalDotProduct(
    singleGPUData& solver,
    ncclComm_t comm,
    cudaStream_t ncclStream,
    const double* d_a,
    const double* d_b)
{
    CUDA_CHECK(cudaSetDevice(solver.deviceId));

    const int n = solver.levels[0].numDOFs;

    // Step 1: Compute local dot product with cuBLAS.
    double localDot = 0.0;

    const int offset = 3 * solver.levels[0].numLeftSharedNodes;

    const int ownedDOFs = solver.levels[0].numDOFs - offset;

    cublasDdot(
        handle,
        ownedDOFs,
        d_a + offset,
        1,
        d_b + offset,
        1,
        &localDot);

    // Step 2: Allocate device scalar for NCCL.
    ScopedDeviceScalar scopedDot;
    double* d_dot = scopedDot.get();

    CUDA_CHECK(cudaMemcpy(
        d_dot,
        &localDot,
        sizeof(double),
        cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    workerBarrier();
    // Step 3: Sum local dot products across all GPUs.
    //CHECK--->GLOVAL_FOR OT WHY TWO D_DOTSßßß
    NCCL_CHECK(
        ncclAllReduce(
            d_dot,
            d_dot,
            1,
            ncclDouble,
            ncclSum,
            comm,
            ncclStream));

    // Wait until NCCL finishes.
    CUDA_CHECK(cudaStreamSynchronize(ncclStream));

    // Step 4: Retrieve global result on each GPU worker.
    double globalDot = 0.0;
    CUDA_CHECK(cudaMemcpy(
        &globalDot,
        d_dot,
        sizeof(double),
        cudaMemcpyDeviceToHost));

    scopedDot.releaseChecked();
    return globalDot;
}

static void globalNorm(
    double& residualNorm,
    singleGPUData& solver,
    ncclComm_t comm,
    cudaStream_t ncclStream,
    const double* d_r)
{
    const Level& fine = solver.levels[0];

    // Skip the left interface, which is owned by the previous GPU.
    const int offset = 3 * fine.numLeftSharedNodes;

    const int ownedDOFs = fine.numDOFs - offset;

    const double* d_ownedR = d_r + offset;

    double localNorm = 0.0;

    CUBLAS_CHECK(cublasDnrm2(
        handle,
        ownedDOFs,
        d_ownedR,
        1,
        &localNorm
    ));

    const double localSquaredNorm = localNorm * localNorm;

    // Square the local norm.
    double localSquaredNorm = localNorm * localNorm;

    // Allocate device scalar for NCCL.
    ScopedDeviceScalar scopedNorm;
    double* d_norm = scopedNorm.get();

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpyAsync(
        d_norm,
        &localSquaredNorm,
        sizeof(double),
        cudaMemcpyHostToDevice,
        ncclStream));

    // Sum squared norms across all GPUs.
    NCCL_CHECK(
        ncclAllReduce(
            d_norm,
            d_norm,
            1,
            ncclDouble,
            ncclSum,
            comm,
            ncclStream));

    // Copy global squared norm to host.
    double globalSquaredNorm = 0.0;

    CUDA_CHECK(cudaMemcpyAsync(
        &globalSquaredNorm,
        d_norm,
        sizeof(double),
        cudaMemcpyDeviceToHost,
        ncclStream));

    CUDA_CHECK(cudaStreamSynchronize(ncclStream));

    scopedNorm.releaseChecked();
    residualNorm = std::sqrt(globalSquaredNorm);
}

static void solveCoarsest(
    coarsestLevelData& solver,
    const double* d_rhs,
    double* d_x)
{
    constexpr int n = solver.numCoarseFreeDOFs;
    const int gridSize =
        gridSizeFor(
            n,
            blockSize);

    CUDA_CHECK(cudaMemset(d_x, 0, static_cast<size_t>(n * sizeof(double blockSize = 256; const int n =))));

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
    coarsestLevelData& solver) noexcept
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

    const int localDOFs = 3 * G[lastLevel].numNodes;

    // Divide the global coarse solution into each partition's local ordering.
    if (rank == rootGPU)
    {
        for (int g = 0; g < G.numGPUs; ++g)
        {
            const int n = C.localNodeCounts[g];

            if (n == 0)
                continue;

            const int grid = gridSizeFor(n, blockSize);

            scatterGlobalCoarseSolutionKernel
                <<<grid, blockSize, 0>>>(
                    C.d_sendResidual[g],
                    C.d_globalX,
                    C.d_localNodeIds[g],
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
    singleGPUData& G,
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
                G.coarsestLevel.d_recvResidual[rootGPU],
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
            const int count = 3 * G.coarsestLevel.localNodeCounts[g];

            if (count == 0)
                continue;

            NCCL_CHECK(ncclRecv(
                G.coarsestLevel.d_recvResidual[g],
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
            G.coarsestLevel.d_globalRHS,
            0,
            static_cast<size_t>(G.coarsestLevel.globalNumDOFs) * sizeof(double),
            0
        ));

        // Copy each partition into global ordering.
        for (int g = 0; g < G.numGPUs; ++g)
        {
            const int n = G.coarsestLevel.localNodeCounts[g];

            if (n == 0)
                continue;

            const int grid = gridSizeFor(n, blockSize);

            gatherGlobalCoarseResidualKernel
                <<<grid, blockSize, 0>>>(
                    G.coarsestLevel.d_recvResidual[g],
                    G.coarsestLevel.d_globalRHS,
                    G.coarsestLevel.d_globalNodeIds[g],
                    n
                );

            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    workerBarrier();
}

static void sendOneSideHalo(
    int sourceDeviceId,
    HaloSide& S,
    const double* d_src,
    int neighborDeviceId,
    double* d_dstNeighbor,
    cudaStream_t stream)
{
    if (S.sendLocal.empty() || neighborDeviceId < 0)
        return;
    //Todo---> Remove this check later. And maybe put it in the validation or initialization function.
    if (S.sendLocal.size() != S.sendDestLocal.size()) {
        throw std::runtime_error("Halo index count mismatch");
    }

    for (size_t j = 1; j < S.sendDestLocal.size(); ++j) {
        if (S.sendDestLocal[j] != S.sendDestLocal[j - 1] + 1) {
            throw std::runtime_error(
                "Non-contiguous sendDestLocal: halo memcpy is invalid");
        }
    }
    const int count = static_cast<int>(S.sendLocal.size());
    const int grid = gridSizeFor(count, 256);

    packHaloKernel<<<grid, 256, 0, stream>>>(
        d_src,
        S.d_sendLocal,
        S.d_sendBuffer,
        count
    );
    CUDA_CHECK(cudaGetLastError());

    // Destination offset in the neighbor's buffer (MATLAB 1-based index)
    const size_t offset = 3 * static_cast<size_t>(S.sendDestLocal[0] - 1);
    if (j && dest != S->sendDestLocal[j-1] + 1) {
        throw std::runtime_error(
            "sendDestLocal must be contiguous in the halo buffer.");
    }
    copyBoundaryToHalo(
        sourceDeviceId,
        S.d_sendBuffer,
        neighborDeviceId,
        d_dstNeighbor + offset,
        count,
        stream
    );
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

        // Todo-->why??
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
                G.deviceId,,
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
        G,
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
        G.coarsestLevel,
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
    // TODO---> fixed-DOF indices must be local to the current GPU, and your correction output convention should clearly specify whether ghost nodes are included. THis is valid for everytime this is called.
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
    int8 sourceId,
    double* d_sourceLocal,
    int8 targetId,
    double* d_recvLoval,
    int numBoundaryNodes,
    cudaStream_t stream)
{
    if (numBoundaryNodes <= 0)
        return;

    CUDA_CHECK(cudaSetDevice(sourceId));

    // Copy boundary data to halo region on the same GPU
    CUDA_CHECK(cudaMemcpyPeerAsync(
        d_recvLoval,
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
    const int numGPUs = G.numGPUs;
    const int deviceId = G.deviceId;

    // Exchange finest-level interface contributions with neighboring GPUs.
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

    constexpr int blockSize = 256;

    Level& finest = G.levels[0];

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
            <<<grid, blockSize, 0, streams.leftStream>>>(
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
            <<<grid, blockSize, 0, streams.rightStream>>>(
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
        CUDA_CHECK(cudaStreamSynchronize(streams.leftStream));
    }

    if (right.recvCount > 0)
    {
        CUDA_CHECK(cudaStreamSynchronize(streams.rightStream));
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

    // Initialize communication streams.------->TODO: put this in initializationFunction
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

        ncclComm_t* comm = &comms[g];
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
            initializeGPUHierarchy();

            initializeGPUs();

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
