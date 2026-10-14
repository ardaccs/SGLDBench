#include "mex.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <omp.h>

// Only the original restriction kernel is retained verbatim below.
// The new packing kernel is needed because MATLAB sendLocal is an index list,
// not a contiguous memory region.

static void cudaCheck(cudaError_t error, const char* call, const char* file, int line)
{
    if (error != cudaSuccess) {
        std::ostringstream msg;
        msg << "CUDA error in " << call << " at " << file << ":" << line
            << ": " << cudaGetErrorString(error);
        throw std::runtime_error(msg.str());
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

// TODO: Optimization: Calculate inner computation first so that it overlaps
//       with communication of shared nodes.
// TODO: Optimization: Use NVSHMEM.
#define MEX_PRINT(...) do { mexPrintf(__VA_ARGS__); mexPrintf("\n"); mexEvalString("drawnow;"); } while (0)

struct StreamData
{
    cudaStream_t rightStream = nullptr;
    cudaStream_t leftStream = nullptr;
};

struct HaloSide
{
    int neighborGPU = -1;            // CUDA device ID; -1 for no neighbor.
    std::vector<int32_t> sendLocal;  // MATLAB 1-based positions in local nodes.
    std::vector<int32_t> sendDestLocal; // MATLAB 1-based positions on receiver.
    int32_t* d_sendLocal = nullptr;
    double* d_sendBuffer = nullptr;
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

    // MATLAB-owned host data; pointers are valid during this MEX call.
    const int32_t* h_nodeToElements = nullptr;
    const int32_t* h_eNodMat = nullptr;
    const int32_t* h_nodGridId = nullptr;
    const int32_t* h_nodMapForward = nullptr;
    const double* h_eleModulus = nullptr;
    const double* h_dK = nullptr;

    // Device hierarchy data. This operation only needs the geometry maps.
    int32_t* d_nodeToElements = nullptr;
    int32_t* d_eNodMat = nullptr;
    int32_t* d_nodGridId = nullptr;
    int32_t* d_nodMapForward = nullptr;
    double* d_eleModulus = nullptr;
    double* d_dK = nullptr;

    // Partition information and per-call vectors.
    const int32_t* h_globalNodeIds = nullptr;
    int numBufferNodes = 0;
    double* d_fineResidual = nullptr;    // [normal local | left halo | right halo]
    double* d_coarseResidual = nullptr;  // normal local nodes only
    HaloSide left, right;
};

struct singleGPUData
{
    int deviceId = -1;
    int numNodes = 0;
    int numElements = 0;
    int numSharedNodes = 0;
    int resX = 0;
    int resY = 0;
    int resZ = 0;
    std::vector<Level> levels;     // levels[0] = fine, levels[1] = coarse
    std::vector<int> spanWidths;

    // Kept from the existing data layout for compatibility with later stages.
    const int32_t* h_nodeToElements = nullptr;
    const int32_t* h_eNodMat = nullptr;
    const int32_t* h_sharedNodesLocal = nullptr;
    const int32_t* h_coarseNodeGridId = nullptr;
    const int32_t* h_fineNodeMapForward = nullptr;
    const double* h_eleModulus = nullptr;
    const double* h_fineResidual = nullptr;
    int32_t* d_nodeToElements = nullptr;
    int32_t* d_eNodMat = nullptr;
    int32_t* d_sharedNodesLocal = nullptr;
    double* d_eleModulus = nullptr;
};

static const mxArray* requireField(const mxArray* structure, mwIndex index,
                                   const char* name)
{
    const mxArray* field = mxGetField(structure, index, name);
    if (!field) {
        throw std::runtime_error(std::string("Missing field: ") + name);
    }
    return field;
}

static int scalarInt(const mxArray* a, const char* name)
{
    if (!a || !mxIsNumeric(a) || mxIsComplex(a) || mxIsSparse(a)
        || mxGetNumberOfElements(a) != 1) {
        throw std::runtime_error(std::string(name) + " must be a numeric scalar.");
    }
    double value = mxGetScalar(a);
    if (!std::isfinite(value) || value < 0 || value > INT_MAX || std::floor(value) != value) {
        throw std::runtime_error(std::string(name) + " must be a nonnegative integer.");
    }
    return static_cast<int>(value);
}

static const int32_t* int32Array(const mxArray* a, const char* name)
{
    if (!a || !mxIsInt32(a) || mxIsComplex(a) || mxIsSparse(a)) {
        throw std::runtime_error(std::string(name) + " must be a full int32 array.");
    }
    return static_cast<const int32_t*>(mxGetData(a));
}

static std::vector<int32_t> readIndices(const mxArray* a, const char* name)
{
    const int32_t* values = int32Array(a, name);
    const size_t count = mxGetNumberOfElements(a);
    if (!count) return {};
    return std::vector<int32_t>(values, values + count);
}

static const mxArray* getPartition(const mxArray* H, int level, int g, int numGPUs)
{
    const mxArray* cells = requireField(H, level, "partitions");
    if (!mxIsCell(cells) || mxGetNumberOfElements(cells) != (size_t)numGPUs) {
        throw std::runtime_error("Each requested level must have exactly numGPUs partitions.");
    }
    const mxArray* P = mxGetCell(cells, g);
    if (!P || !mxIsStruct(P) || mxGetNumberOfElements(P) != 1) {
        throw std::runtime_error("Each partition must be a scalar struct.");
    }
    return P;
}

static void readHaloSide(const mxArray* P, const char* direction,
                         HaloSide& side, int expectedNeighbor, int g)
{
    const mxArray* halo = requireField(P, 0, "halo");
    const mxArray* sideMx = requireField(halo, 0, direction);
    int matNeighbor = scalarInt(requireField(sideMx, 0, "neighborGPU"), "neighborGPU");
    if (matNeighbor != (expectedNeighbor < 0 ? 0 : expectedNeighbor + 1)) {
        throw std::runtime_error("Halo neighborGPU is inconsistent with partition order.");
    }
    side.neighborGPU = expectedNeighbor;
    side.sendLocal = readIndices(requireField(sideMx, 0, "sendLocal"), "sendLocal");
    side.sendDestLocal = readIndices(requireField(sideMx, 0, "sendDestLocal"), "sendDestLocal");
    if (side.sendLocal.size() != side.sendDestLocal.size()) {
        throw std::runtime_error("Halo sendLocal/sendDestLocal lengths differ.");
    }
    if (expectedNeighbor < 0 && !side.sendLocal.empty()) {
        throw std::runtime_error("Boundary partition cannot send to a missing neighbor.");
    }
    (void)g;
}

static void initializeGPUData(const mxArray* hierarchyMx,
                              int fineLevelIndex, int numGPUs,
                              std::vector<singleGPUData>& gpuData)
{
    const int spanWidth = scalarInt(requireField(hierarchyMx, fineLevelIndex + 1, "spanWidth"), "spanWidth");
    if (spanWidth <= 0) throw std::runtime_error("spanWidth must be positive.");

    gpuData.resize(numGPUs);
    for (int g = 0; g < numGPUs; ++g) {
        singleGPUData& G = gpuData[g];
        G.deviceId = g;
        G.levels.resize(2);
        G.spanWidths.push_back(spanWidth);
        for (int k = 0; k < 2; ++k) {
            Level& L = G.levels[k];
            const int hierarchyIndex = fineLevelIndex + k;
            const mxArray* P = getPartition(hierarchyMx, hierarchyIndex, g, numGPUs);
            L.nx = scalarInt(requireField(hierarchyMx, hierarchyIndex, "resX"), "resX");
            L.ny = scalarInt(requireField(hierarchyMx, hierarchyIndex, "resY"), "resY");
            L.nz = scalarInt(requireField(hierarchyMx, hierarchyIndex, "resZ"), "resZ");
            L.numGridNodes = (size_t)(L.nx + 1) * (size_t)(L.ny + 1) * (size_t)(L.nz + 1);
            if (L.numGridNodes > (size_t)INT_MAX) {
                throw std::runtime_error("Grid exceeds 32-bit indexing used by the unchanged kernel.");
            }
            L.numNodes = scalarInt(requireField(P, 0, "numNodes"), "numNodes");
            L.numElements = scalarInt(requireField(P, 0, "numElements"), "numElements");
            L.numDOFs = L.numNodes * 3;
            L.h_globalNodeIds = int32Array(requireField(P, 0, "globalNodeIds"), "globalNodeIds");
            if (mxGetNumberOfElements(requireField(P, 0, "globalNodeIds")) != (size_t)L.numNodes) {
                throw std::runtime_error("Partition globalNodeIds length does not match numNodes.");
            }
            L.h_nodGridId = int32Array(requireField(P, 0, "nodGridId"), "nodGridId");
            if (mxGetNumberOfElements(requireField(P, 0, "nodGridId")) != (size_t)L.numNodes) {
                throw std::runtime_error("Partition nodGridId length does not match numNodes.");
            }
            if (k == 0) {
                L.numBufferNodes = scalarInt(requireField(P, 0, "numBufferNodes"), "numBufferNodes");
                if (L.numBufferNodes < L.numNodes) {
                    throw std::runtime_error("numBufferNodes must include all local nodes.");
                }
                const mxArray* forward = requireField(P, 0, "nodMapForwardWithHalo");
                L.h_nodMapForward = int32Array(forward, "nodMapForwardWithHalo");
                if (mxGetNumberOfElements(forward) != L.numGridNodes) {
                    throw std::runtime_error("nodMapForwardWithHalo must index the entire fine Cartesian grid.");
                }
                readHaloSide(P, "left", L.left, g > 0 ? g - 1 : -1, g);
                readHaloSide(P, "right", L.right, g + 1 < numGPUs ? g + 1 : -1, g);
            }
        }
        Level& fine = G.levels[0];
        Level& coarse = G.levels[1];
        if (fine.nx != coarse.nx * spanWidth ||
            fine.ny != coarse.ny * spanWidth ||
            fine.nz != coarse.nz * spanWidth) {
            throw std::runtime_error("Fine/coarse Cartesian resolutions do not match spanWidth.");
        }
        G.numNodes = fine.numNodes;
        G.numElements = fine.numElements;
        G.resX = fine.nx; G.resY = fine.ny; G.resZ = fine.nz;
        G.h_coarseNodeGridId = coarse.h_nodGridId;
        G.h_fineNodeMapForward = fine.h_nodMapForward;
    }
    // Verify that every sender destination agrees with the neighbor's halo.
    for (int g = 0; g < numGPUs; ++g) {
        Level& L = gpuData[g].levels[0];
        for (HaloSide* S : {&L.left, &L.right}) {
            for (size_t j = 0; j < S->sendLocal.size(); ++j) {
                int32_t local = S->sendLocal[j];
                int32_t dest = S->sendDestLocal[j];
                if (local < 1 || local > L.numNodes) {
                    throw std::runtime_error("sendLocal index outside normal local nodes.");
                }
                if (dest < 1 || dest > gpuData[S->neighborGPU].levels[0].numBufferNodes) {
                    throw std::runtime_error("sendDestLocal index outside neighbor buffer.");
                }
                if (j && dest != S->sendDestLocal[j-1] + 1) {
                    throw std::runtime_error("sendDestLocal must be contiguous in the halo buffer.");
                }
            }
        }
    }
}

static void initializeGPUs(std::vector<singleGPUData>& gpuData,
                           const double* globalFineResidual, size_t numGlobalFineNodes)
{
    for (auto& G : gpuData) {
        CUDA_CHECK(cudaSetDevice(G.deviceId));
        Level& fine = G.levels[0];
        Level& coarse = G.levels[1];
        std::vector<double> localResidual((size_t)fine.numBufferNodes * 3, 0.0);
        for (int n = 0; n < fine.numNodes; ++n) {
            int32_t globalId = fine.h_globalNodeIds[n]; // MATLAB 1-based
            if (globalId < 1 || (size_t)globalId > numGlobalFineNodes) {
                throw std::runtime_error("Fine partition globalNodeIds contains an invalid index.");
            }
            for (int component = 0; component < 3; ++component)
                localResidual[3 * (size_t)n + component] =
                    globalFineResidual[3 * (size_t)(globalId-1) + component];
        }
        // cudaMalloc(0) is invalid on some CUDA runtimes; allocate at least one slot.
        CUDA_CHECK(cudaMalloc((void**)&fine.d_fineResidual,
                              std::max((size_t)1, (size_t)fine.numBufferNodes * 3) * sizeof(double)));
        if (!localResidual.empty()) {
            CUDA_CHECK(cudaMemcpy(fine.d_fineResidual, localResidual.data(),
                                  localResidual.size() * sizeof(double), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMalloc((void**)&fine.d_nodMapForward,
                              fine.numGridNodes * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(fine.d_nodMapForward, fine.h_nodMapForward,
                              fine.numGridNodes * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void**)&coarse.d_nodGridId,
                              std::max(1, coarse.numNodes) * sizeof(int32_t)));
        if (coarse.numNodes) {
            CUDA_CHECK(cudaMemcpy(coarse.d_nodGridId, coarse.h_nodGridId,
                                  (size_t)coarse.numNodes * sizeof(int32_t), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMalloc((void**)&coarse.d_coarseResidual,
                              (size_t)std::max(1, coarse.numNodes * 3) * sizeof(double)));
        for (HaloSide* S : {&fine.left, &fine.right}) {
            const size_t count = S->sendLocal.size();
            if (!count) continue;
            CUDA_CHECK(cudaMalloc((void**)&S->d_sendLocal, count * sizeof(int32_t)));
            CUDA_CHECK(cudaMemcpy(S->d_sendLocal, S->sendLocal.data(),
                                  count * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc((void**)&S->d_sendBuffer, count * 3 * sizeof(double)));
        }
    }
    mexPrintf("\nAll GPU mesh data initialized.\n");
}

static void freeGPUs(std::vector<singleGPUData>& gpuData,
                     std::vector<StreamData>& comm_stream) noexcept
{
    for (auto& G : gpuData) {
        if (G.deviceId < 0 || cudaSetDevice(G.deviceId) != cudaSuccess) continue;
        for (Level& L : G.levels) {
            if (L.d_nodeToElements) cudaFree(L.d_nodeToElements);
            if (L.d_eNodMat) cudaFree(L.d_eNodMat);
            if (L.d_nodGridId) cudaFree(L.d_nodGridId);
            if (L.d_nodMapForward) cudaFree(L.d_nodMapForward);
            if (L.d_eleModulus) cudaFree(L.d_eleModulus);
            if (L.d_dK) cudaFree(L.d_dK);
            if (L.d_fineResidual) cudaFree(L.d_fineResidual);
            if (L.d_coarseResidual) cudaFree(L.d_coarseResidual);
            for (HaloSide* S : {&L.left, &L.right}) {
                if (S->d_sendLocal) cudaFree(S->d_sendLocal);
                if (S->d_sendBuffer) cudaFree(S->d_sendBuffer);
                S->d_sendLocal = nullptr;
                S->d_sendBuffer = nullptr;
            }
            L.d_nodeToElements = nullptr;
            L.d_eNodMat = nullptr;
            L.d_nodGridId = nullptr;
            L.d_nodMapForward = nullptr;
            L.d_eleModulus = nullptr;
            L.d_dK = nullptr;
            L.d_fineResidual = nullptr;
            L.d_coarseResidual = nullptr;
        }
        if (G.d_nodeToElements) cudaFree(G.d_nodeToElements);
        if (G.d_eNodMat) cudaFree(G.d_eNodMat);
        if (G.d_eleModulus) cudaFree(G.d_eleModulus);
        if (G.d_sharedNodesLocal) cudaFree(G.d_sharedNodesLocal);
        G.d_nodeToElements = nullptr;
        G.d_eNodMat = nullptr;
        G.d_eleModulus = nullptr;
        G.d_sharedNodesLocal = nullptr;
        if ((size_t)G.deviceId < comm_stream.size()) {
            StreamData& S = comm_stream[G.deviceId];
            if (S.leftStream) cudaStreamDestroy(S.leftStream);
            if (S.rightStream) cudaStreamDestroy(S.rightStream);
            S.leftStream = nullptr;
            S.rightStream = nullptr;
        }
    }
}

// Copy from the sender's packed, contiguous double[3*numBoundaryNodes]
// into the destination's already-contiguous halo slots.
static cudaError_t copyBoundaryToHalo(
    int sourceId, const double* d_sourceLocal,
    int targetId, double* d_recvLocal,
    int numBoundaryNodes, cudaStream_t stream)
{
    if (numBoundaryNodes <= 0) return cudaSuccess;
    cudaError_t error = cudaSetDevice(sourceId);
    if (error != cudaSuccess) return error;
    return cudaMemcpyPeerAsync(d_recvLocal, targetId,
                               d_sourceLocal, sourceId,
                               (size_t)numBoundaryNodes * 3 * sizeof(double), stream);
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


static void configurePeerAccess(int numGPUs)
{
    for (int g = 0; g < numGPUs; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        for (int neighbor : {g - 1, g + 1}) {
            if (neighbor < 0 || neighbor >= numGPUs) continue;
            int canAccessPeer = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, g, neighbor));
            if (canAccessPeer) {
                cudaError_t result = cudaDeviceEnablePeerAccess(neighbor, 0);
                if (result != cudaSuccess && result != cudaErrorPeerAccessAlreadyEnabled)
                    CUDA_CHECK(result);
            }
            // Otherwise cudaMemcpyPeerAsync can use runtime-managed staging;
            // no direct NVLink/PCIe peer-access assumption is made here.
        }
    }
}

// Used inside OpenMP regions: return CUDA failures to the host thread instead
// of calling any MATLAB MEX function from worker threads.
static cudaError_t sendOneSide(singleGPUData& G, HaloSide& S, cudaStream_t stream,
                               const std::vector<singleGPUData>& gpuData)
{
    if (S.sendLocal.empty()) return cudaSuccess;
    if (S.neighborGPU < 0) return cudaErrorInvalidDevice;
    const int count = (int)S.sendLocal.size();
    const int block = 256;
    const int grid = (count + block - 1) / block;
    packHaloKernel<<<grid, block, 0, stream>>>(
        G.levels[0].d_fineResidual, S.d_sendLocal, S.d_sendBuffer, count);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    double* recv = gpuData[S.neighborGPU].levels[0].d_fineResidual
                   + 3 * ((size_t)S.sendDestLocal[0] - 1);
    return copyBoundaryToHalo(G.deviceId, S.d_sendBuffer,
                              S.neighborGPU, recv, count, stream);
}

static void checkWorkerErrors(const std::vector<cudaError_t>& errors, const char* step)
{
    for (size_t g = 0; g < errors.size(); ++g) {
        if (errors[g] != cudaSuccess) {
            std::ostringstream msg;
            msg << "GPU " << g << " failed during " << step << ": "
                << cudaGetErrorString(errors[g]);
            throw std::runtime_error(msg.str());
        }
    }
}

static void runRestriction(std::vector<singleGPUData>& gpuData,
                           std::vector<StreamData>& comm_stream)
{
    const int numGPUs = (int)gpuData.size();
    const int plannedHostNumThreads = numGPUs;
    const int cpu_cores = omp_get_num_procs();
    mexPrintf("CPU logical threads available: %d\n", cpu_cores);
    omp_set_dynamic(0);
    omp_set_num_threads(plannedHostNumThreads);

    comm_stream.resize(numGPUs);
    for (int i = 0; i < numGPUs; ++i) {
        CUDA_CHECK(cudaSetDevice(gpuData[i].deviceId));
        int leastPriority = 0, greatestPriority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        const int priority = greatestPriority;
        if (i > 0) {
            CUDA_CHECK(cudaStreamCreateWithPriority(
                &comm_stream[i].leftStream, cudaStreamNonBlocking, priority));
        }
        if (i < numGPUs - 1) {
            CUDA_CHECK(cudaStreamCreateWithPriority(
                &comm_stream[i].rightStream, cudaStreamNonBlocking, priority));
        }
    }

    std::vector<cudaError_t> errors(numGPUs, cudaSuccess);
    // Stage 1: prepare both left/right packed boundary buffers and post P2P
    // copies to the *neighbor's* halo slots. Each sender owns its streams.
    #pragma omp parallel for num_threads(plannedHostNumThreads)
    for (int i = 0; i < numGPUs; ++i) {
        singleGPUData& G = gpuData[i];
        cudaError_t result = cudaSetDevice(G.deviceId);
        if (result == cudaSuccess)
            result = sendOneSide(G, G.levels[0].right,
                                 comm_stream[i].rightStream, gpuData);
        if (result == cudaSuccess)
            result = sendOneSide(G, G.levels[0].left,
                                 comm_stream[i].leftStream, gpuData);
        errors[i] = result;
    }
    checkWorkerErrors(errors, "halo exchange enqueue");

    // Stage 2: do NOT launch the restriction kernel until both neighboring
    // sends have finished writing this GPU's fine halo buffer.
    #pragma omp parallel for num_threads(plannedHostNumThreads)
    for (int i = 0; i < numGPUs; ++i) {
        cudaError_t result = cudaSetDevice(gpuData[i].deviceId);
        if (result == cudaSuccess && comm_stream[i].rightStream)
            result = cudaStreamSynchronize(comm_stream[i].rightStream);
        if (result == cudaSuccess && comm_stream[i].leftStream)
            result = cudaStreamSynchronize(comm_stream[i].leftStream);
        errors[i] = result;
    }
    checkWorkerErrors(errors, "halo exchange synchronization");

    // Stage 3: every GPU uses its own coarse nodes and its own halo-aware
    // *fine* nodMapForward. The original restriction kernel is unchanged.
    #pragma omp parallel for num_threads(plannedHostNumThreads)
    for (int i = 0; i < numGPUs; ++i) {
        singleGPUData& G = gpuData[i];
        Level& fine = G.levels[0];
        Level& coarse = G.levels[1];
        cudaError_t result = cudaSetDevice(G.deviceId);
        if (result == cudaSuccess && coarse.numNodes > 0) {
            const int block = 256;
            const int grid = (coarse.numNodes + block - 1) / block;
            restrictResidualKernel<<<grid, block>>>(
                coarse.d_nodGridId,
                fine.d_nodMapForward,
                fine.d_fineResidual,
                coarse.d_coarseResidual,
                coarse.numNodes,
                coarse.nx, coarse.ny, coarse.nz,
                fine.nx, fine.ny, fine.nz,
                G.spanWidths[0]);
            result = cudaGetLastError();
            if (result == cudaSuccess) result = cudaDeviceSynchronize();
        }
        errors[i] = result;
    }
    checkWorkerErrors(errors, "restriction kernel");
}

static void assembleOutput(const mxArray* hierarchyMx, int fineLevelIndex,
                           std::vector<singleGPUData>& gpuData, mxArray* plhs[])
{
    const int numCoarseGlobalNodes = scalarInt(
        requireField(hierarchyMx, fineLevelIndex + 1, "numNodes"), "coarse numNodes");
    plhs[0] = mxCreateDoubleMatrix((mwSize)numCoarseGlobalNodes * 3, 1, mxREAL);
    double* out = mxGetDoubles(plhs[0]);
    std::vector<unsigned char> filled(numCoarseGlobalNodes, 0);
    for (auto& G : gpuData) {
        CUDA_CHECK(cudaSetDevice(G.deviceId));
        Level& coarse = G.levels[1];
        if (!coarse.numNodes) continue;
        std::vector<double> local((size_t)coarse.numNodes * 3);
        CUDA_CHECK(cudaMemcpy(local.data(), coarse.d_coarseResidual,
                              local.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (int n = 0; n < coarse.numNodes; ++n) {
            const int32_t gid = coarse.h_globalNodeIds[n] - 1;
            if (gid < 0 || gid >= numCoarseGlobalNodes)
                throw std::runtime_error("Coarse partition globalNodeIds has invalid indices.");
            const size_t dst = (size_t)gid * 3;
            const size_t src = (size_t)n * 3;
            if (!filled[gid]) {
                for (int c = 0; c < 3; ++c) out[dst + c] = local[src + c];
                filled[gid] = 1;
            } else {
                // Shared interface nodes are duplicated by the partitioner.
                // Each GPU must calculate the SAME value: never sum copies.
                for (int c = 0; c < 3; ++c) {
                    const double a = out[dst + c], b = local[src + c];
                    if (!std::isfinite(a) || !std::isfinite(b) ||
                        std::abs(a-b) > 1e-9 * (1 + std::max(std::abs(a), std::abs(b)))) {
                        throw std::runtime_error(
                            "Disagreement at shared coarse node: inspect halo width / split alignment.");
                    }
                }
            }
        }
    }
    for (auto hasValue : filled)
        if (!hasValue) throw std::runtime_error("Coarse partition coverage is incomplete.");
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*
     * Inputs:
     *   0: H, MATLAB mesh hierarchy struct with partition/halo metadata
     *   1: numGPUs, integer scalar
     *   2: fineResidual, [3*H(level).numNodes x 1], real double, OPTIONAL
     *   3: level, MATLAB 1-based fine level, OPTIONAL (defaults to 1)
     * If input 2 is omitted, H(level).residual must exist instead.
     *
     * Output: coarseResidual, [3*H(level+1).numNodes x 1], real double.
     */
    if (nrhs < 2 || nrhs > 4)
        mexErrMsgIdAndTxt("restrict_residual_mgpu:nrhs", "Need H, numGPUs, [fineResidual], [level].");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("restrict_residual_mgpu:nlhs", "Exactly one output is required.");

    std::vector<singleGPUData> gpuData;
    std::vector<StreamData> comm_stream;
    try {
        const mxArray* hierarchyMx = prhs[0];
        if (!mxIsStruct(hierarchyMx))
            throw std::runtime_error("H must be a struct array.");
        int numGPUs = scalarInt(prhs[1], "numGPUs");
        if (numGPUs < 1) throw std::runtime_error("numGPUs must be at least 1.");
        const int numLevels = (int)mxGetNumberOfElements(hierarchyMx);
        const int level = nrhs >= 4 ? scalarInt(prhs[3], "level") : 1;
        if (level < 1 || level >= numLevels)
            throw std::runtime_error("level must have a next (coarser) hierarchy level.");
        const int fineLevelIndex = level - 1;
        const mxArray* residualMx = nrhs >= 3
            ? prhs[2] : mxGetField(hierarchyMx, fineLevelIndex, "residual");
        if (!residualMx || !mxIsDouble(residualMx) || mxIsComplex(residualMx)
            || mxIsSparse(residualMx))
            throw std::runtime_error("Fine residual must be a real full double array: pass input 3 or H(level).residual.");
        const int nFineGlobal = scalarInt(
            requireField(hierarchyMx, fineLevelIndex, "numNodes"), "fine numNodes");
        if (mxGetNumberOfElements(residualMx) != (size_t)nFineGlobal * 3)
            throw std::runtime_error("fineResidual length must equal 3*H(level).numNodes.");

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount < numGPUs)
            throw std::runtime_error("Available CUDA devices fewer than numGPUs.");
        initializeGPUData(hierarchyMx, fineLevelIndex, numGPUs, gpuData);
        configurePeerAccess(numGPUs);
        initializeGPUs(gpuData, mxGetDoubles(residualMx), (size_t)nFineGlobal);
        runRestriction(gpuData, comm_stream);
        assembleOutput(hierarchyMx, fineLevelIndex, gpuData, plhs);
        freeGPUs(gpuData, comm_stream);
    } catch (const std::exception& e) {
        // Synchronize before freeing buffers possibly referenced by pending
        // async copies or kernels; freeGPUs handles partially initialized data.
        for (const auto& G : gpuData) {
            if (G.deviceId >= 0 && cudaSetDevice(G.deviceId) == cudaSuccess)
                cudaDeviceSynchronize();
        }
        freeGPUs(gpuData, comm_stream);
        mexErrMsgIdAndTxt("restrict_residual_mgpu:error", "%s", e.what());
    }
}
