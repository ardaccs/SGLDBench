#include "mex.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        mexErrMsgIdAndTxt("CUDA:error", cudaGetErrorString(err));  \
    }                                                             \
} while (0)

__constant__ double c_Ke[24 * 24];

__device__ __forceinline__ int linearNode(int x, int y, int z, int nx, int ny)
{
    // 0-based dense node coords
    return z * (nx + 1) * (ny + 1) + x * (ny + 1) + y;
}

__device__ __forceinline__ int linearElem(int x, int y, int z, int nx, int ny)
{
    // 0-based dense element coords
    return z * nx * ny + x * ny + y;
}

__device__ __forceinline__ void unpackNodeDense(
    int denseNode0, int nx, int ny,
    int& x, int& y, int& z)
{
    int nyp1 = ny + 1;
    int nxp1 = nx + 1;
    int slice = nyp1 * nxp1;

    z = denseNode0 / slice;
    int rem = denseNode0 - z * slice;
    x = rem / nyp1;
    y = rem - x * nyp1;
}

__global__ void kbyu_vertex_gather_kernel(
    const double* __restrict__ U,
    double* __restrict__ Y,
    const int32_t* __restrict__ nodMapBack,     // compact node -> dense node, MATLAB 1-based
    const int32_t* __restrict__ nodMapForward,  // dense node -> compact node, MATLAB 1-based, 0 inactive
    const int32_t* __restrict__ eleMapForward,  // dense elem -> compact elem, MATLAB 1-based, 0 void
    const double* __restrict__ E,
    int numNodes,
    int nx, int ny, int nz)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= numNodes) return;

    int denseNode0 = nodMapBack[v] - 1; // to 0-based dense node id

    int vx, vy, vz;
    unpackNodeDense(denseNode0, nx, ny, vx, vy, vz);

    double sum[3] = {0.0, 0.0, 0.0};

    /*
        SGLDBench local node order:

             8--------------7
            /|             /|
           5-------------6  |
           | |           |  |
           | 4-----------|--3
           |/            | /
           1-------------2

        tmp = [0 ny+[1 0] -1 (ny+1)*(nx+1)+[0 ny+[1 0] -1]]

        In 0-based dense-node offset form:
    */
    const int nodeStrideY = 1;
    const int nodeStrideX = ny + 1;
    const int nodeStrideZ = (nx + 1) * (ny + 1);

    int localNodeOffset[8] = {
        0,
        nodeStrideX,
        nodeStrideX - 1,
        -1,
        nodeStrideZ,
        nodeStrideZ + nodeStrideX,
        nodeStrideZ + nodeStrideX - 1,
        nodeStrideZ - 1
    };

    /*
        Loop over the 8 possible incident elements.
        For a vertex (vx,vy,vz), incident element anchors are:

        (vx-1/vx, vy-1/vy, vz-1/vz)

        The local node index of the current vertex inside that element
        is determined by which side of the anchor it lies on.
    */
    #pragma unroll
    for (int iz = 0; iz < 2; ++iz) {
        #pragma unroll
        for (int ix = 0; ix < 2; ++ix) {
            #pragma unroll
            for (int iy = 0; iy < 2; ++iy) {

                int ex = vx - ix;
                int ey = vy - iy;
                int ez = vz - iz;

                if (ex < 0 || ex >= nx) continue;
                if (ey < 0 || ey >= ny) continue;
                if (ez < 0 || ez >= nz) continue;

                int elemDense0 = linearElem(ex, ey, ez, nx, ny);
                int elemCompact1 = eleMapForward[elemDense0];

                if (elemCompact1 == 0) continue;

                int elemCompact0 = elemCompact1 - 1;

                /*
                    Anchor node = local node 1.
                    SGLDBench has eNodVec = nodenrs(...)+1.
                    So anchorDense0 is dense nodenrs + 1 in MATLAB,
                    but here we use 0-based dense node IDs.
                */
                int anchorDense0 = linearNode(ex, ey, ez, nx, ny) + 1;

                double Ue[24];

                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int denseNodeJ0 = anchorDense0 + localNodeOffset[j];

                    int compactNode1 = nodMapForward[denseNodeJ0];
                    if (compactNode1 == 0) {
                        Ue[3*j + 0] = 0.0;
                        Ue[3*j + 1] = 0.0;
                        Ue[3*j + 2] = 0.0;
                    } else {
                        int compactNode0 = compactNode1 - 1;
                        int base = 3 * compactNode0;

                        Ue[3*j + 0] = U[base + 0];
                        Ue[3*j + 1] = U[base + 1];
                        Ue[3*j + 2] = U[base + 2];
                    }
                }

                /*
                    Determine local node index of current vertex.

                    ix,iy,iz describe whether the element anchor is behind
                    the current node in x/y/z.

                    Need to match SGLDBench order:
                    local 1: anchor
                    local 2: +x
                    local 3: +x -y
                    local 4: -y
                    local 5: +z
                    local 6: +x +z
                    local 7: +x -y +z
                    local 8: -y +z
                */
                int loc = -1;

                if (ix == 0 && iy == 0 && iz == 0) loc = 0; // node 1
                if (ix == 1 && iy == 0 && iz == 0) loc = 1; // node 2
                if (ix == 1 && iy == 1 && iz == 0) loc = 2; // node 3
                if (ix == 0 && iy == 1 && iz == 0) loc = 3; // node 4
                if (ix == 0 && iy == 0 && iz == 1) loc = 4; // node 5
                if (ix == 1 && iy == 0 && iz == 1) loc = 5; // node 6
                if (ix == 1 && iy == 1 && iz == 1) loc = 6; // node 7
                if (ix == 0 && iy == 1 && iz == 1) loc = 7; // node 8

                int row0 = 3 * loc + 0;
                int row1 = 3 * loc + 1;
                int row2 = 3 * loc + 2;

                double y0 = 0.0;
                double y1 = 0.0;
                double y2 = 0.0;

                #pragma unroll
                for (int k = 0; k < 24; ++k) {
                    double uk = Ue[k];

                    // MATLAB column-major Ke(row,col)
                    y0 += c_Ke[row0 + k * 24] * uk;
                    y1 += c_Ke[row1 + k * 24] * uk;
                    y2 += c_Ke[row2 + k * 24] * uk;
                }

                double Ee = E[elemCompact0];

                sum[0] += Ee * y0;
                sum[1] += Ee * y1;
                sum[2] += Ee * y2;
            }
        }
    }

    int out = 3 * v;
    Y[out + 0] = sum[0];
    Y[out + 1] = sum[1];
    Y[out + 2] = sum[2];
}
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // inputs:
    // U, Ke, nodMapBack, nodMapForward, eleMapForward, E, nx, ny, nz

    if (nrhs != 9) {
        mexErrMsgIdAndTxt("sgld:nrhs", "Need 9 inputs.");
    }

    const mxArray* U_mx = prhs[0];
    const mxArray* Ke_mx = prhs[1];
    const mxArray* nodBack_mx = prhs[2];
    const mxArray* nodForward_mx = prhs[3];
    const mxArray* eleForward_mx = prhs[4];
    const mxArray* E_mx = prhs[5];

    int nx = (int)mxGetScalar(prhs[6]);
    int ny = (int)mxGetScalar(prhs[7]);
    int nz = (int)mxGetScalar(prhs[8]);

    int numDOFs = (int)mxGetNumberOfElements(U_mx);
    int numNodes = numDOFs / 3;
    int numElements = (int)mxGetNumberOfElements(E_mx);

    const double* h_U = mxGetDoubles(U_mx);
    const double* h_Ke = mxGetDoubles(Ke_mx);
    const int32_t* h_nodBack = (const int32_t*)mxGetData(nodBack_mx);
    const int32_t* h_nodForward = (const int32_t*)mxGetData(nodForward_mx);
    const int32_t* h_eleForward = (const int32_t*)mxGetData(eleForward_mx);
    const double* h_E = mxGetDoubles(E_mx);

    plhs[0] = mxCreateDoubleMatrix(numDOFs, 1, mxREAL);
    double* h_Y = mxGetDoubles(plhs[0]);

    double *d_U, *d_Y, *d_E;
    int32_t *d_nodBack, *d_nodForward, *d_eleForward;

    CUDA_CHECK(cudaMalloc(&d_U, numDOFs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Y, numDOFs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_E, numElements * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_nodBack, numNodes * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_nodForward, ((nx+1)*(ny+1)*(nz+1)) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_eleForward, (nx*ny*nz) * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_U, h_U, numDOFs * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E, numElements * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodBack, h_nodBack, numNodes * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodForward, h_nodForward, ((nx+1)*(ny+1)*(nz+1)) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eleForward, h_eleForward, (nx*ny*nz) * sizeof(int32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(c_Ke, h_Ke, 24 * 24 * sizeof(double)));

    int block = 256;
    int grid = (numNodes + block - 1) / block;

    kbyu_vertex_gather_kernel<<<grid, block>>>(
        d_U,
        d_Y,
        d_nodBack,
        d_nodForward,
        d_eleForward,
        d_E,
        numNodes,
        nx, ny, nz
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, numDOFs * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_U);
    cudaFree(d_Y);
    cudaFree(d_E);
    cudaFree(d_nodBack);
    cudaFree(d_nodForward);
    cudaFree(d_eleForward);
}