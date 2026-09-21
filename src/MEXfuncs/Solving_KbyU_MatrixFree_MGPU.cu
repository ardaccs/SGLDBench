#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <climits>
#include <cstdio>
#include <vector>


// Elementary stiffness matrix (4.6 kB)
__constant__ double c_Ke[24 * 24];

//TODO:The check fails to freeGPU memory, implement this!
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

struct singleGPUData
{
    int deviceId = -1;

    int numNodes = 0;
    int numElements = 0;
    int numSharedNodes = 0;

    // MATLAB-owned host data.
    // Valid during this MEX call.
    const int32_t* h_nodeToElements = nullptr;
    const int32_t* h_eNodMat = nullptr;
    const int32_t* h_sharedNodesLocal = nullptr;
    const double* h_eleModulus = nullptr;

    // Device data.
    int32_t* d_nodeToElements = nullptr;
    int32_t* d_eNodMat = nullptr;
    int32_t* d_sharedNodesLocal = nullptr;
    double* d_eleModulus = nullptr;
};

static void initializeGPUData(const mxArray* hierarchyMx, const mxArray* bMx, const mxArray* yMx, const size_t numGPUs, std::vector<singleGPUData>& gpuData)
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

    int deviceCount = 0;

    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < static_cast<int>(numGPUs))
    {
        mexErrMsgIdAndTxt(
            "mgpu:devices",
            "Not enough CUDA devices available.");
    }

    gpuData.resize(numGPUs);
    for (size_t i = 0; i < numGPUs; ++i)
    {
        singleGPUData& G = gpuData[i];

        // MATLAB:
        // H(1) -> CUDA device 0
        // H(2) -> CUDA device 1

        G.deviceId = static_cast<int>(i);

        const mxArray* numNodesMx = mxGetField(hierarchyMx, i, "numNodes");
        const mxArray* numElementsMx = mxGetField(hierarchyMx, i, "numElements");
        const mxArray* nodeToElementsMx = mxGetField(hierarchyMx, i, "nodeToElements");
        const mxArray* eNodMatMx = mxGetField(hierarchyMx, i, "eNodMat");
        const mxArray* eleModulusMx = mxGetField(hierarchyMx, i, "eleModulus");
        const mxArray* sharedNodesCell = mxGetField(hierarchyMx, i, "sharedNodesLocal");


        if (!numNodesMx || !numElementsMx || !nodeToElementsMx ||
            !eNodMatMx || !eleModulusMx || !sharedNodesCell)
        {
            mexErrMsgIdAndTxt(
                "mgpu:missingField",
                "A required hierarchy field is missing.");
        }

        double nodesValue = mxGetScalar(numNodesMx);
        double elementsValue = mxGetScalar(numElementsMx);
        G.numNodes = static_cast<int>(nodesValue);
        G.numElements = static_cast<int>(elementsValue);

        G.h_nodeToElements = static_cast<const int32_t*>(mxGetData(nodeToElementsMx));
        G.h_eNodMat = static_cast<const int32_t*>(mxGetData(eNodMatMx));
        G.h_eleModulus = mxGetDoubles(eleModulusMx);

        // Read sharedNodesLocal
        if (!mxIsCell(sharedNodesCell) || mxGetNumberOfElements(sharedNodesCell) != numGPUs)
        {
            mexErrMsgIdAndTxt(
                "mgpu:sharedNodes",
                "sharedNodesLocal must be a numGPUs-sized cell array.");
        }

        //TODO. Update so that this works for more than 2 GPUs. For now, we only support 2 GPUs, so we can assume that the other GPU is the one that owns the shared nodes.
        // For two GPUs:
        //
        // GPU 0 reads H(1).sharedNodesLocal{2}
        // GPU 1 reads H(2).sharedNodesLocal{1}

        size_t otherGPU = (i == 0) ? 1 : 0;

        const mxArray* sharedMx = mxGetCell(sharedNodesCell, otherGPU);

        if (!sharedMx || !mxIsInt32(sharedMx) || mxIsSparse(sharedMx))
        {
            mexErrMsgIdAndTxt(
                "mgpu:sharedNodes",
                "Invalid sharedNodesLocal array.");
        }

        if (mxGetNumberOfElements(sharedMx) > static_cast<size_t>(INT_MAX))
        {
            mexErrMsgIdAndTxt(
                "mgpu:sharedNodes",
                "Too many shared nodes.");
        }

        G.numSharedNodes = static_cast<int>(mxGetNumberOfElements(sharedMx));
        G.h_sharedNodesLocal = static_cast<const int32_t*>(mxGetData(sharedMx));
    }

}

static void initializeGPUs(
    std::vector<singleGPUData>& gpuData,
    const mxArray* KeMx)
{
    // Validate element stiffness matrix
    if (KeMx == nullptr ||
        !mxIsDouble(KeMx) ||
        mxIsComplex(KeMx) ||
        mxIsSparse(KeMx) ||
        mxGetM(KeMx) != 24 ||
        mxGetN(KeMx) != 24)
    {
        mexErrMsgIdAndTxt(
            "mgpu:Ke",
            "Ke must be a full real 24x24 double matrix.");
    }

    const double* h_Ke = mxGetDoubles(KeMx);

    // ========================================================
    // Initialize each GPU
    // ========================================================

    for (size_t i = 0; i < gpuData.size(); ++i)
    {
        singleGPUData& G = gpuData[i];

        // ----------------------------------------------------
        // 1. Select CUDA device
        // ----------------------------------------------------

        CUDA_CHECK(cudaSetDevice(G.deviceId));

        mexPrintf("\nInitializing CUDA device %d\n", G.deviceId);


        // ----------------------------------------------------
        // 2. Calculate memory requirements
        // ----------------------------------------------------

        size_t bytesN2E =
            static_cast<size_t>(G.numNodes) *
            8 * sizeof(int32_t);

        size_t bytesENod =
            static_cast<size_t>(G.numElements) *
            8 * sizeof(int32_t);

        size_t bytesE =
            static_cast<size_t>(G.numElements) *
            sizeof(double);

        size_t bytesShared =
            static_cast<size_t>(G.numSharedNodes) *
            sizeof(int32_t);


        // ----------------------------------------------------
        // 3. Allocate device memory
        // ----------------------------------------------------

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&G.d_nodeToElements),
            bytesN2E));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&G.d_eNodMat),
            bytesENod));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&G.d_eleModulus),
            bytesE));

        // Avoid cudaMalloc with size 0
        if (G.numSharedNodes > 0)
        {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&G.d_sharedNodesLocal),
                bytesShared));
        }


        // ----------------------------------------------------
        // 4. Copy MATLAB data to GPU
        // ----------------------------------------------------

        CUDA_CHECK(cudaMemcpy(
            G.d_nodeToElements,
            G.h_nodeToElements,
            bytesN2E,
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            G.d_eNodMat,
            G.h_eNodMat,
            bytesENod,
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            G.d_eleModulus,
            G.h_eleModulus,
            bytesE,
            cudaMemcpyHostToDevice));

        if (G.numSharedNodes > 0)
        {
            CUDA_CHECK(cudaMemcpy(
                G.d_sharedNodesLocal,
                G.h_sharedNodesLocal,
                bytesShared,
                cudaMemcpyHostToDevice));
        }


        // ----------------------------------------------------
        // 5. Initialize constant memory on this GPU
        // ----------------------------------------------------

        CUDA_CHECK(cudaMemcpyToSymbol(
            c_Ke,
            h_Ke,
            24 * 24 * sizeof(double)));


        // ----------------------------------------------------
        // 6. Print allocation information
        // ----------------------------------------------------

        size_t totalBytes =
            bytesN2E +
            bytesENod +
            bytesE +
            bytesShared;

        mexPrintf(
            "GPU %d:\n"
            "  Nodes:          %d\n"
            "  Elements:       %d\n"
            "  Shared nodes:   %d\n"
            "  Mesh memory:    %.3f MB\n",
            G.deviceId,
            G.numNodes,
            G.numElements,
            G.numSharedNodes,
            totalBytes / (1024.0 * 1024.0));
    }

    mexPrintf("\nAll GPU mesh data initialized.\n");
}
static void freeGPUs(
    std::vector<singleGPUData>& gpuData)
{
    for (auto& G : gpuData)
    {
        CUDA_CHECK(cudaSetDevice(G.deviceId));

        if (G.d_nodeToElements)
            CUDA_CHECK(cudaFree(G.d_nodeToElements));

        if (G.d_eNodMat)
            CUDA_CHECK(cudaFree(G.d_eNodMat));

        if (G.d_eleModulus)
            CUDA_CHECK(cudaFree(G.d_eleModulus));

        if (G.d_sharedNodesLocal)
            CUDA_CHECK(cudaFree(G.d_sharedNodesLocal));

        G.d_nodeToElements = nullptr;
        G.d_eNodMat = nullptr;
        G.d_eleModulus = nullptr;
        G.d_sharedNodesLocal = nullptr;
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

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // MATLAB: [Y, r] = mgpu_kbyu(H, b, y0, Ke, numGPUs)
    // b and y0: [GPU 0 local DOFs; GPU 1 local DOFs].
    // Shared-node values are duplicated in both local vectors.

    if (nrhs != 5 || nlhs < 1 || nlhs > 2)
        mexErrMsgIdAndTxt("mgpu:arguments",
            "Use [Y, r] = mgpu_kbyu(H, b, y0, Ke, numGPUs); r is optional.");

    const mxArray* hierarchyMx = prhs[0];
    const mxArray* bMx         = prhs[1];
    const mxArray* yMx         = prhs[2];
    const mxArray* KeMx        = prhs[3];

    if (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) ||
        mxGetNumberOfElements(prhs[4]) != 1 || mxGetScalar(prhs[4]) != 2)
        mexErrMsgIdAndTxt("mgpu:numGPUs",
            "This version supports exactly two GPUs.");

    const size_t numGPUs = 2;
    std::vector<singleGPUData> gpuData;
    initializeGPUData(hierarchyMx, bMx, yMx, numGPUs, gpuData);

    if (mxIsSparse(bMx) || mxIsSparse(yMx))
        mexErrMsgIdAndTxt("mgpu:inputType", "b and y0 must be full arrays.");

    // Offset of each GPU's local DOFs in the concatenated MATLAB vectors.
    std::vector<mwSize> offset(numGPUs + 1, 0);
    for (size_t i = 0; i < numGPUs; ++i)
    {
        const singleGPUData& G = gpuData[i];
        const mxArray* nodesMx = mxGetField(hierarchyMx, i, "numNodes");
        const mxArray* elementsMx = mxGetField(hierarchyMx, i, "numElements");
        const mxArray* n2eMx = mxGetField(hierarchyMx, i, "nodeToElements");
        const mxArray* enodMx = mxGetField(hierarchyMx, i, "eNodMat");
        const mxArray* eMx = mxGetField(hierarchyMx, i, "eleModulus");

        if (!mxIsNumeric(nodesMx) || mxIsComplex(nodesMx) ||
            mxGetNumberOfElements(nodesMx) != 1 ||
            !mxIsNumeric(elementsMx) || mxIsComplex(elementsMx) ||
            mxGetNumberOfElements(elementsMx) != 1 ||
            mxGetScalar(nodesMx) != static_cast<double>(G.numNodes) ||
            mxGetScalar(elementsMx) != static_cast<double>(G.numElements) ||
            G.numNodes <= 0 || G.numElements <= 0 || G.numNodes > INT_MAX / 3 ||
            !mxIsInt32(n2eMx) || mxIsSparse(n2eMx) ||
            mxGetM(n2eMx) != static_cast<mwSize>(G.numNodes) || mxGetN(n2eMx) != 8 ||
            !mxIsInt32(enodMx) || mxIsSparse(enodMx) ||
            mxGetM(enodMx) != static_cast<mwSize>(G.numElements) || mxGetN(enodMx) != 8 ||
            !mxIsDouble(eMx) || mxIsComplex(eMx) || mxIsSparse(eMx) ||
            mxGetNumberOfElements(eMx) != static_cast<mwSize>(G.numElements))
            mexErrMsgIdAndTxt("mgpu:mesh",
                "Invalid local mesh fields or dimensions in H(%d).", static_cast<int>(i + 1));

        offset[i + 1] = offset[i] + 3 * static_cast<mwSize>(G.numNodes);
    }

    const mwSize numDOFs = offset[numGPUs];
    if (mxGetNumberOfElements(bMx) != numDOFs ||
        mxGetNumberOfElements(yMx) != numDOFs)
        mexErrMsgIdAndTxt("mgpu:vectorSize",
            "b and y0 must each have sum_i(3*H(i).numNodes) entries.");

    // For two GPUs, the two shared-node arrays must match positionally.
    if (gpuData[0].numSharedNodes != gpuData[1].numSharedNodes)
        mexErrMsgIdAndTxt("mgpu:sharedNodes", "Shared-node counts do not match.");

    for (int j = 0; j < gpuData[0].numSharedNodes; ++j)
    {
        const int32_t n0 = gpuData[0].h_sharedNodesLocal[j];
        const int32_t n1 = gpuData[1].h_sharedNodesLocal[j];
        if (n0 < 1 || n0 > gpuData[0].numNodes ||
            n1 < 1 || n1 > gpuData[1].numNodes)
            mexErrMsgIdAndTxt("mgpu:sharedNodes", "Invalid local shared-node index.");
    }

    const double* h_U = mxGetDoubles(yMx);
    initializeGPUs(gpuData, KeMx);  // Uses existing mesh allocation and Ke copies.

    plhs[0] = mxCreateDoubleMatrix(numDOFs, 1, mxREAL);
    double* h_Y = mxGetDoubles(plhs[0]);

    std::vector<double*> d_U(numGPUs, nullptr);
    std::vector<double*> d_Y(numGPUs, nullptr);

    // Start both GPUs; kernels execute independently on their own devices.
    #pragma unroll
    for (size_t i = 0; i < numGPUs; ++i)
    {
        singleGPUData& G = gpuData[i];
        CUDA_CHECK(cudaSetDevice(G.deviceId));

        const size_t bytes = static_cast<size_t>(G.numNodes) * 3 * sizeof(double);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_U[i]), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Y[i]), bytes));
        CUDA_CHECK(cudaMemcpy(d_U[i], h_U + offset[i], bytes, cudaMemcpyHostToDevice));

        const int block = 256;
        const int grid = (G.numNodes + block - 1) / block;
        kbyu_kernel<<<grid, block>>>(
            d_U[i], d_Y[i],
            G.d_nodeToElements, G.d_eNodMat, G.d_eleModulus,
            G.numNodes, G.numElements);
        CUDA_CHECK(cudaGetLastError());
    }

    // Collect each GPU's local, element-partitioned K*U contribution.
    #pragma unroll
    for (size_t i = 0; i < numGPUs; ++i)
    {
        const singleGPUData& G = gpuData[i];
        CUDA_CHECK(cudaSetDevice(G.deviceId));
        CUDA_CHECK(cudaDeviceSynchronize());
        const size_t bytes = static_cast<size_t>(G.numNodes) * 3 * sizeof(double);
        CUDA_CHECK(cudaMemcpy(h_Y + offset[i], d_Y[i], bytes, cudaMemcpyDeviceToHost));
    }

    // Reduce interface nodes: the j-th entries refer to the SAME physical node.
    // Retain the summed result in both local copies.
    for (int j = 0; j < gpuData[0].numSharedNodes; ++j)
    {
        const mwSize a = offset[0] + 3 * static_cast<mwSize>(gpuData[0].h_sharedNodesLocal[j] - 1);
        const mwSize b = offset[1] + 3 * static_cast<mwSize>(gpuData[1].h_sharedNodesLocal[j] - 1);
        for (int c = 0; c < 3; ++c)
        {
            const double value = h_Y[a + c] + h_Y[b + c];
            h_Y[a + c] = value;
            h_Y[b + c] = value;
        }
    }

    if (nlhs == 2)
    {
        plhs[1] = mxCreateDoubleMatrix(numDOFs, 1, mxREAL);
        double* h_r = mxGetDoubles(plhs[1]);
        const double* h_b = mxGetDoubles(bMx);
        for (mwSize j = 0; j < numDOFs; ++j)
            h_r[j] = h_b[j] - h_Y[j];
    }

    // Free temporary displacement/output vectors, then existing mesh storage.
    for (size_t i = 0; i < numGPUs; ++i)
    {
        CUDA_CHECK(cudaSetDevice(gpuData[i].deviceId));
        CUDA_CHECK(cudaFree(d_U[i]));
        CUDA_CHECK(cudaFree(d_Y[i]));
    }
    freeGPUs(gpuData);
}