
/*
 * sgld_pcg_jacobi_cuda_mex.cu
 *
 * PCG MEX with CUDA matrix-free KbyU + CUDA Jacobi preconditioner.
 *
 * MATLAB:
 *   [U, its, relres] = sgld_pcg_jacobi_cuda_mex( ...
 *       F, U0, nodeToElements, eNodMat, E, Ks, diagK, fixedDOFs, ...
 *       tol, maxIT, resX, resY, resZ, 'printP_ON');
 *
 * Use U0 = [] to start from zero.
 *
 * Compile:
 *   clear mex
 *   mexcuda -R2018a src/MEXfuncs/sgld_pcg_jacobi_cuda_mex.cu
 */

#include "mex.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <cstdint>
#include <climits>
#include <cmath>
#include <cstring>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:cuda",                   \
                "CUDA error at %s:%d: %s",                                   \
                __FILE__, __LINE__, cudaGetErrorString(err__));               \
        }                                                                    \
    } while (0)

#define MEX_PRINT(...)                                                       \
    do {                                                                     \
        mexPrintf(__VA_ARGS__);                                               \
        mexPrintf("\n");                                                     \
        mexEvalString("drawnow;");                                           \
    } while (0)

__global__ void zero_fixed_kernel(
    double* __restrict__ x,
    const unsigned char* __restrict__ fixed,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && fixed[i]) x[i] = 0.0;
}

__global__ void residual_kernel(
    const double* __restrict__ b,
    const double* __restrict__ Ax,
    double* __restrict__ r,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - Ax[i];
}

__global__ void jacobi_kernel(
    const double* __restrict__ r,
    const double* __restrict__ diagK,
    const unsigned char* __restrict__ fixed,
    double* __restrict__ z,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = fixed[i] ? 0.0 : r[i] / diagK[i];
    }
}

__global__ void update_y_r_kernel(
    double* __restrict__ y,
    double* __restrict__ r,
    const double* __restrict__ p,
    const double* __restrict__ Ap,
    double lambda,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += lambda * p[i];
        r[i] -= lambda * Ap[i];
    }
}

__global__ void update_p_kernel(
    double* __restrict__ p,
    const double* __restrict__ z,
    double beta,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = z[i] + beta * p[i];
}

__global__ void kbyu_kernel(
    const double* __restrict__ U,
    double* __restrict__ Y,
    const int32_t* __restrict__ nodeToElements,
    const int32_t* __restrict__ eNodMat,
    const double* __restrict__ E,
    const double* __restrict__ Ke,
    int numNodes,
    int numElements)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= numNodes) return;

    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    int elemNodes[8];
    double Ue[24];

    #pragma unroll
    for (int a = 0; a < 8; ++a)
    {
        int elem = nodeToElements[node + a * numNodes] - 1;

        if (elem < 0 || elem >= numElements)
            continue;

        int localNode = -1;

        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int n = eNodMat[elem + j * numElements] - 1;
            elemNodes[j] = n;
            if (n == node) localNode = j;
        }

        if (localNode < 0)
            continue;

        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int n = elemNodes[j];

            if (n < 0 || n >= numNodes)
            {
                Ue[3*j + 0] = 0.0;
                Ue[3*j + 1] = 0.0;
                Ue[3*j + 2] = 0.0;
                continue;
            }

            int base = 3 * n;
            Ue[3*j + 0] = U[base + 0];
            Ue[3*j + 1] = U[base + 1];
            Ue[3*j + 2] = U[base + 2];
        }

        int row0 = 3 * localNode;

        double y0 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;

        #pragma unroll
        for (int c = 0; c < 24; ++c)
        {
            double u = Ue[c];
            y0 += Ke[(row0 + 0) + c * 24] * u;
            y1 += Ke[(row0 + 1) + c * 24] * u;
            y2 += Ke[(row0 + 2) + c * 24] * u;
        }

        double Ee = E[elem];
        sum0 += Ee * y0;
        sum1 += Ee * y1;
        sum2 += Ee * y2;
    }

    int out = 3 * node;
    Y[out + 0] = sum0;
    Y[out + 1] = sum1;
    Y[out + 2] = sum2;
}

static double device_dot(const double* d_a, const double* d_b, int n)
{
    thrust::device_ptr<const double> a = thrust::device_pointer_cast(d_a);
    thrust::device_ptr<const double> b = thrust::device_pointer_cast(d_b);
    return thrust::inner_product(a, a + n, b, 0.0);
}

static int parse_print_flag(const mxArray* print_mx)
{
    if (mxIsChar(print_mx))
    {
        char* s = mxArrayToString(print_mx);
        if (s == nullptr) return 0;
        int enabled = (std::strcmp(s, "printP_ON") == 0);
        mxFree(s);
        return enabled;
    }

    if (mxIsNumeric(print_mx))
        return ((int)mxGetScalar(print_mx)) != 0;

    return 0;
}

static void launch_kbyu(
    const double* d_U,
    double* d_Y,
    const int32_t* d_nodeToElements,
    const int32_t* d_eNodMat,
    const double* d_E,
    const double* d_Ke,
    const unsigned char* d_fixed,
    int numDOFs,
    int numNodes,
    int numElements,
    int threads)
{
    int nodeBlocks = (numNodes + threads - 1) / threads;
    int dofBlocks  = (numDOFs  + threads - 1) / threads;

    kbyu_kernel<<<nodeBlocks, threads>>>(
        d_U, d_Y, d_nodeToElements, d_eNodMat,
        d_E, d_Ke, numNodes, numElements);

    CUDA_CHECK(cudaGetLastError());

    // Match Solving_KbyU_MatrixFree:
    // productMV(meshHierarchy_(1).fixedDOFs,1) = 0;
    zero_fixed_kernel<<<dofBlocks, threads>>>(d_Y, d_fixed, numDOFs);
    CUDA_CHECK(cudaGetLastError());
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*
     * [U, its, relres] = sgld_pcg_jacobi_cuda_mex( ...
     *      F, U0, nodeToElements, eNodMat, E, Ks, diagK, fixedDOFs, ...
     *      tol, maxIT, nx, ny, nz, printP)
     */

    if (nrhs != 14) {
        mexErrMsgIdAndTxt(
            "sgld_pcg_jacobi_cuda:nrhs",
            "Expected 14 inputs: F, U0, nodeToElements, eNodMat, E, Ks, diagK, fixedDOFs, tol, maxIT, nx, ny, nz, printP.");
    }

    if (nlhs > 3) {
        mexErrMsgIdAndTxt(
            "sgld_pcg_jacobi_cuda:nlhs",
            "Outputs are [U, its, relres].");
    }

    const mxArray* F_mx              = prhs[0];
    const mxArray* U0_mx             = prhs[1];
    const mxArray* nodeToElements_mx = prhs[2];
    const mxArray* eNodMat_mx        = prhs[3];
    const mxArray* E_mx              = prhs[4];
    const mxArray* Ke_mx             = prhs[5];
    const mxArray* diagK_mx          = prhs[6];
    const mxArray* fixed_mx          = prhs[7];

    double tol = mxGetScalar(prhs[8]);
    int maxIT = (int)mxGetScalar(prhs[9]);

    // Kept in the interface for consistency with the KbyU MEX.
    int nx = (int)mxGetScalar(prhs[10]);
    int ny = (int)mxGetScalar(prhs[11]);
    int nz = (int)mxGetScalar(prhs[12]);
    (void)nx; (void)ny; (void)nz;

    int printEnabled = parse_print_flag(prhs[13]);

    if (!mxIsDouble(F_mx) || mxIsComplex(F_mx) || mxIsSparse(F_mx))
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:F", "F must be full real double.");

    if (!mxIsInt32(nodeToElements_mx) || mxIsSparse(nodeToElements_mx))
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:nodeToElements", "nodeToElements must be full int32.");

    if (!mxIsInt32(eNodMat_mx) || mxIsSparse(eNodMat_mx))
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:eNodMat", "eNodMat must be full int32.");

    if (!mxIsDouble(E_mx) || mxIsComplex(E_mx) || mxIsSparse(E_mx))
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:E", "E must be full real double.");

    if (!mxIsDouble(Ke_mx) || mxIsComplex(Ke_mx) || mxIsSparse(Ke_mx))
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:Ks", "Ks must be full real double [24 x 24].");

    if (!mxIsDouble(diagK_mx) || mxIsComplex(diagK_mx) || mxIsSparse(diagK_mx))
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:diagK", "diagK must be full real double.");

    if (!mxIsLogical(fixed_mx) || mxIsSparse(fixed_mx))
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:fixedDOFs", "fixedDOFs must be full logical.");

    size_t numDOFs_sz = mxGetNumberOfElements(F_mx);
    if (numDOFs_sz % 3 != 0)
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:FSize", "length(F) must be divisible by 3.");

    if (numDOFs_sz > INT_MAX)
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:size", "numDOFs exceeds int range.");

    int numDOFs = (int)numDOFs_sz;
    int numNodes = numDOFs / 3;

    size_t numElements_sz = mxGetNumberOfElements(E_mx);
    if (numElements_sz > INT_MAX)
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:size", "numElements exceeds int range.");

    int numElements = (int)numElements_sz;

    if ((int)mxGetM(nodeToElements_mx) != numNodes || mxGetN(nodeToElements_mx) < 8)
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:nodeToElementsSize", "nodeToElements must be [numNodes x 8].");

    if ((int)mxGetM(eNodMat_mx) != numElements || mxGetN(eNodMat_mx) < 8)
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:eNodMatSize", "eNodMat must be [numElements x 8].");

    if (mxGetM(Ke_mx) != 24 || mxGetN(Ke_mx) != 24)
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:KsSize", "Ks must be [24 x 24].");

    if (mxGetNumberOfElements(diagK_mx) != (size_t)numDOFs)
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:diagKSize", "diagK must have length numDOFs.");

    if (mxGetNumberOfElements(fixed_mx) != (size_t)numDOFs)
        mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:fixedSize", "fixedDOFs must have length numDOFs.");

    bool hasU0 = !mxIsEmpty(U0_mx);

    if (hasU0)
    {
        if (!mxIsDouble(U0_mx) || mxIsComplex(U0_mx) || mxIsSparse(U0_mx))
            mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:U0", "U0 must be [] or full real double.");

        if (mxGetNumberOfElements(U0_mx) != (size_t)numDOFs)
            mexErrMsgIdAndTxt("sgld_pcg_jacobi_cuda:U0Size", "U0 must have length numDOFs.");
    }

    const double* h_F     = mxGetDoubles(F_mx);
    const double* h_U0    = hasU0 ? mxGetDoubles(U0_mx) : nullptr;
    const double* h_E     = mxGetDoubles(E_mx);
    const double* h_Ke    = mxGetDoubles(Ke_mx);
    const double* h_diagK = mxGetDoubles(diagK_mx);

    const int32_t* h_nodeToElements =
        static_cast<const int32_t*>(mxGetData(nodeToElements_mx));

    const int32_t* h_eNodMat =
        static_cast<const int32_t*>(mxGetData(eNodMat_mx));

    const mxLogical* h_fixed_logical = mxGetLogicals(fixed_mx);

    unsigned char* h_fixed_u8 =
        static_cast<unsigned char*>(mxMalloc((mwSize)numDOFs * sizeof(unsigned char)));

    for (int i = 0; i < numDOFs; ++i)
        h_fixed_u8[i] = h_fixed_logical[i] ? 1 : 0;

    size_t bytesDOF            = sizeof(double)  * (size_t)numDOFs;
    size_t bytesNodeToElements = sizeof(int32_t) * (size_t)numNodes * (size_t)8;
    size_t bytesENodMat        = sizeof(int32_t) * (size_t)numElements * (size_t)8;
    size_t bytesE              = sizeof(double)  * (size_t)numElements;
    size_t bytesKe             = sizeof(double)  * (size_t)24 * (size_t)24;
    size_t bytesFixed          = sizeof(unsigned char) * (size_t)numDOFs;

    double* d_b = nullptr;
    double* d_y = nullptr;
    double* d_r = nullptr;
    double* d_z = nullptr;
    double* d_p = nullptr;
    double* d_Ap = nullptr;
    double* d_diagK = nullptr;
    double* d_E = nullptr;
    double* d_Ke = nullptr;

    int32_t* d_nodeToElements = nullptr;
    int32_t* d_eNodMat = nullptr;
    unsigned char* d_fixed = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_b, bytesDOF));
    CUDA_CHECK(cudaMalloc((void**)&d_y, bytesDOF));
    CUDA_CHECK(cudaMalloc((void**)&d_r, bytesDOF));
    CUDA_CHECK(cudaMalloc((void**)&d_z, bytesDOF));
    CUDA_CHECK(cudaMalloc((void**)&d_p, bytesDOF));
    CUDA_CHECK(cudaMalloc((void**)&d_Ap, bytesDOF));

    CUDA_CHECK(cudaMalloc((void**)&d_diagK, bytesDOF));
    CUDA_CHECK(cudaMalloc((void**)&d_fixed, bytesFixed));

    CUDA_CHECK(cudaMalloc((void**)&d_nodeToElements, bytesNodeToElements));
    CUDA_CHECK(cudaMalloc((void**)&d_eNodMat, bytesENodMat));
    CUDA_CHECK(cudaMalloc((void**)&d_E, bytesE));
    CUDA_CHECK(cudaMalloc((void**)&d_Ke, bytesKe));

    CUDA_CHECK(cudaMemcpy(d_b, h_F, bytesDOF, cudaMemcpyHostToDevice));

    if (hasU0)
        CUDA_CHECK(cudaMemcpy(d_y, h_U0, bytesDOF, cudaMemcpyHostToDevice));
    else
        CUDA_CHECK(cudaMemset(d_y, 0, bytesDOF));

    CUDA_CHECK(cudaMemcpy(d_diagK, h_diagK, bytesDOF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fixed, h_fixed_u8, bytesFixed, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodeToElements, h_nodeToElements, bytesNodeToElements, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eNodMat, h_eNodMat, bytesENodMat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E, bytesE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ke, h_Ke, bytesKe, cudaMemcpyHostToDevice));

    mxFree(h_fixed_u8);

    const int threads = 256;
    const int dofBlocks = (numDOFs + threads - 1) / threads;

    // normB = norm(b)
    double normB = std::sqrt(device_dot(d_b, d_b, numDOFs));

    int its = 0;
    double resnorm = 0.0;

    if (normB == 0.0)
    {
        plhs[0] = mxCreateDoubleMatrix((mwSize)numDOFs, 1, mxREAL);
        double* h_out = mxGetDoubles(plhs[0]);
        std::memset(h_out, 0, bytesDOF);

        if (nlhs >= 2) plhs[1] = mxCreateDoubleScalar(0.0);
        if (nlhs >= 3) plhs[2] = mxCreateDoubleScalar(0.0);
        goto cleanup;
    }

    // r = b - AtX(y)
    launch_kbyu(
        d_y, d_Ap,
        d_nodeToElements, d_eNodMat,
        d_E, d_Ke, d_fixed,
        numDOFs, numNodes, numElements,
        threads);

    residual_kernel<<<dofBlocks, threads>>>(d_b, d_Ap, d_r, numDOFs);
    CUDA_CHECK(cudaGetLastError());

    // Keep fixed DOFs out of the residual.
    zero_fixed_kernel<<<dofBlocks, threads>>>(d_r, d_fixed, numDOFs);
    CUDA_CHECK(cudaGetLastError());

    // z = Jacobi(r)
    jacobi_kernel<<<dofBlocks, threads>>>(d_r, d_diagK, d_fixed, d_z, numDOFs);
    CUDA_CHECK(cudaGetLastError());

    // p = z
    CUDA_CHECK(cudaMemcpy(d_p, d_z, bytesDOF, cudaMemcpyDeviceToDevice));

    // x1Val = z' * r
    double x1Val = device_dot(d_z, d_r, numDOFs);

    for (its = 1; its <= maxIT; ++its)
    {
        // zVec = AtX(p), named Ap here.
        launch_kbyu(
            d_p, d_Ap,
            d_nodeToElements, d_eNodMat,
            d_E, d_Ke, d_fixed,
            numDOFs, numNodes, numElements,
            threads);

        // lambda = x1Val / (p' * zVec)
        double pAp = device_dot(d_p, d_Ap, numDOFs);

        if (!std::isfinite(pAp) || pAp == 0.0)
        {
            mexErrMsgIdAndTxt(
                "sgld_pcg_jacobi_cuda:pAp",
                "Invalid p' * Ap = %.12e at iteration %d.",
                pAp, its);
        }

        double lambda = x1Val / pAp;

        // y = y + lambda * p
        // r = r - lambda * Ap
        update_y_r_kernel<<<dofBlocks, threads>>>(d_y, d_r, d_p, d_Ap, lambda, numDOFs);
        CUDA_CHECK(cudaGetLastError());

        zero_fixed_kernel<<<dofBlocks, threads>>>(d_y, d_fixed, numDOFs);
        CUDA_CHECK(cudaGetLastError());

        zero_fixed_kernel<<<dofBlocks, threads>>>(d_r, d_fixed, numDOFs);
        CUDA_CHECK(cudaGetLastError());

        resnorm = std::sqrt(device_dot(d_r, d_r, numDOFs)) / normB;

        if (printEnabled)
            MEX_PRINT(" It.: %4d Res.: %16.6e", its, resnorm);

        if (resnorm < tol)
        {
            MEX_PRINT(
                "CG solver converged at iteration%5d to a solution with relative residual%16.6e",
                its, resnorm);
            break;
        }

        // z = Jacobi(r)
        jacobi_kernel<<<dofBlocks, threads>>>(d_r, d_diagK, d_fixed, d_z, numDOFs);
        CUDA_CHECK(cudaGetLastError());

        // x2Val = z' * r
        double x2Val = device_dot(d_z, d_r, numDOFs);

        if (!std::isfinite(x2Val))
        {
            mexErrMsgIdAndTxt(
                "sgld_pcg_jacobi_cuda:x2Val",
                "Invalid z' * r = %.12e at iteration %d.",
                x2Val, its);
        }

        // p = z + x2Val / x1Val * p
        double beta = x2Val / x1Val;

        update_p_kernel<<<dofBlocks, threads>>>(d_p, d_z, beta, numDOFs);
        CUDA_CHECK(cudaGetLastError());

        zero_fixed_kernel<<<dofBlocks, threads>>>(d_p, d_fixed, numDOFs);
        CUDA_CHECK(cudaGetLastError());

        x1Val = x2Val;
    }

    if (its > maxIT)
    {
        its = maxIT;
        mexWarnMsgIdAndTxt("sgld_pcg_jacobi_cuda:maxIT", "Exceed the maximum iterate numbers");
        MEX_PRINT("The iterative process stops at residual = %10.4f", resnorm);
    }

    // Output U
    plhs[0] = mxCreateDoubleMatrix((mwSize)numDOFs, 1, mxREAL);
    CUDA_CHECK(cudaMemcpy(mxGetDoubles(plhs[0]), d_y, bytesDOF, cudaMemcpyDeviceToHost));

    if (nlhs >= 2) plhs[1] = mxCreateDoubleScalar((double)its);
    if (nlhs >= 3) plhs[2] = mxCreateDoubleScalar(resnorm);

cleanup:
    if (d_b) CUDA_CHECK(cudaFree(d_b));
    if (d_y) CUDA_CHECK(cudaFree(d_y));
    if (d_r) CUDA_CHECK(cudaFree(d_r));
    if (d_z) CUDA_CHECK(cudaFree(d_z));
    if (d_p) CUDA_CHECK(cudaFree(d_p));
    if (d_Ap) CUDA_CHECK(cudaFree(d_Ap));
    if (d_diagK) CUDA_CHECK(cudaFree(d_diagK));
    if (d_fixed) CUDA_CHECK(cudaFree(d_fixed));
    if (d_nodeToElements) CUDA_CHECK(cudaFree(d_nodeToElements));
    if (d_eNodMat) CUDA_CHECK(cudaFree(d_eNodMat));
    if (d_E) CUDA_CHECK(cudaFree(d_E));
    if (d_Ke) CUDA_CHECK(cudaFree(d_Ke));
}
