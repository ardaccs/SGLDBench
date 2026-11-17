// Standalone ownership-aware cuBLAS + NCCL reduction test.
// Compile: nvcc -O2 -std=c++17 -Xcompiler -fopenmp test_global_reductions.cu -lcublas -lnccl -o test_global_reductions
// Run:     ./test_global_reductions 2     or ./test_global_reductions 4
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <limits>

#define CUDA_OK(x) do { cudaError_t s=(x); if(s!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(s)); std::abort();} }while(0)
#define BLAS_OK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d: %d\n",__FILE__,__LINE__,int(s));std::abort();} }while(0)
#define NCCL_OK(x) do { ncclResult_t s=(x); if(s!=ncclSuccess){fprintf(stderr,"NCCL %s:%d: %s\n",__FILE__,__LINE__,ncclGetErrorString(s));std::abort();} }while(0)

struct Level { int numDOFs=0, numLeftSharedNodes=0; };
struct singleGPUData { int deviceId=-1; Level levels[1]; cublasHandle_t handle=nullptr; };

// Same math as the user's globalDotProduct(), corrected handle/error checking.
// Assumes each local vector is ordered [left shared | owned interior/right].
static double globalDotProduct(singleGPUData& solver, ncclComm_t comm,
                               cudaStream_t ncclStream, const double* d_a,
                               const double* d_b) {
    CUDA_OK(cudaSetDevice(solver.deviceId));

    const Level& fine = solver.levels[0];
    const int offset = 3 * fine.numLeftSharedNodes;
    const int ownedDOFs = fine.numDOFs - offset;

    double localDot = 0.0;

    BLAS_OK(cublasDdot(
        solver.handle,
        ownedDOFs,
        d_a + offset,
        1,
        d_b + offset,
        1,
        &localDot));

    double* d_dot = nullptr;
    CUDA_OK(cudaMalloc(&d_dot, sizeof(double)));

    CUDA_OK(cudaMemcpy(
        d_dot,
        &localDot,
        sizeof(double),
        cudaMemcpyHostToDevice));

    CUDA_OK(cudaDeviceSynchronize());

    #pragma omp barrier

    NCCL_OK(ncclAllReduce(
        d_dot,
        d_dot,
        1,
        ncclDouble,
        ncclSum,
        comm,
        ncclStream
    ));

    CUDA_OK(cudaStreamSynchronize(ncclStream));

    // Retrieve global result.
    double globalDot = 0.0;

    CUDA_OK(cudaMemcpy(
        &globalDot,
        d_dot,
        sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    CUDA_OK(cudaFree(d_dot));

    return globalDot;
}

static void globalNorm(double& residualNorm, singleGPUData& solver,
                       ncclComm_t comm, cudaStream_t ncclStream,
                       const double* d_r) {
    CUDA_OK(cudaSetDevice(solver.deviceId));
    const Level& fine=solver.levels[0];
    const int offset=3*fine.numLeftSharedNodes;
    const int ownedDOFs=fine.numDOFs-offset;
    double localNorm=0.0;
    BLAS_OK(cublasDnrm2(solver.handle,ownedDOFs,d_r+offset,1,&localNorm));
    const double localSquaredNorm=localNorm*localNorm;
    double* d_norm=nullptr;
    CUDA_OK(cudaMalloc(&d_norm,sizeof(double)));
    CUDA_OK(cudaMemcpyAsync(d_norm,&localSquaredNorm,sizeof(double),cudaMemcpyHostToDevice,ncclStream));
    NCCL_OK(ncclAllReduce(d_norm,d_norm,1,ncclDouble,ncclSum,comm,ncclStream));
    double globalSquaredNorm=0.0;
    CUDA_OK(cudaMemcpyAsync(&globalSquaredNorm,d_norm,sizeof(double),cudaMemcpyDeviceToHost,ncclStream));
    CUDA_OK(cudaStreamSynchronize(ncclStream));
    CUDA_OK(cudaFree(d_norm));
    residualNorm=std::sqrt(globalSquaredNorm);
}

// Value is deterministic for a given global DOF, including duplicates.
static double avalue(int globalDof,int test) {
    if(test==0) return 1.0;
    if(test==1) return (globalDof%11-5)*0.25;
    if(test==2) return 0.0;
    return std::sin(0.137*(globalDof+1));
}
static double bvalue(int globalDof,int test) {
    if(test==0) return 2.0;
    if(test==1) return (globalDof%7-3)*0.5;
    if(test==2) return 0.0;
    return std::cos(0.213*(globalDof+2));
}
static bool near(double x,double y) {
    return std::isfinite(x) && std::isfinite(y) &&
           std::abs(x-y) <= 2e-11 * std::max({1.0,std::abs(x),std::abs(y)});
}
int main(int argc,char**argv) {
    int available=0; CUDA_OK(cudaGetDeviceCount(&available));
    const int ngpu=argc>1?std::atoi(argv[1]):2;
    if(ngpu<2 || ngpu>available) {
        std::fprintf(stderr,"Need 2..%d visible GPUs; requested %d\n",available,ngpu);
        return 2;
    }
    // Four global nodes in each segment, plus its shared right endpoint.
    // Rank g's local nodes: [4*g, 4*g+1, ..., 4*g+4].
    // For g>0 the first node is a duplicate of previous rank's last node.
    constexpr int stride=4;
    constexpr int localNodes=stride+1;
    const int globalNodes=ngpu*stride+1;
    std::vector<ncclComm_t> comms(ngpu);
    std::vector<int> devs(ngpu);
    for(int g=0;g<ngpu;++g) devs[g]=g;
    NCCL_OK(ncclCommInitAll(comms.data(),ngpu,devs.data()));
    int failed=0;
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(ngpu) reduction(+:failed)
    {
        const int g=omp_get_thread_num();
        singleGPUData solver;
        solver.deviceId=g;
        solver.levels[0].numDOFs=3*localNodes;
        solver.levels[0].numLeftSharedNodes=(g>0)?1:0;
        CUDA_OK(cudaSetDevice(g));
        BLAS_OK(cublasCreate(&solver.handle));
        BLAS_OK(cublasSetPointerMode(solver.handle,CUBLAS_POINTER_MODE_HOST));
        cudaStream_t stream=nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        double *d_a=nullptr,*d_b=nullptr;
        CUDA_OK(cudaMalloc(&d_a,3*localNodes*sizeof(double)));
        CUDA_OK(cudaMalloc(&d_b,3*localNodes*sizeof(double)));
        for(int test=0;test<4;++test) {
            std::vector<double> a(3*localNodes),b(3*localNodes);
            for(int node=0;node<localNodes;++node) {
                for(int comp=0;comp<3;++comp) {
                    int j=3*node+comp;
                    int globalDof=3*(g*stride+node)+comp;
                    a[j]=avalue(globalDof,test);
                    b[j]=bvalue(globalDof,test);
                }
            }
            CUDA_OK(cudaMemcpy(d_a,a.data(),a.size()*sizeof(double),cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(d_b,b.data(),b.size()*sizeof(double),cudaMemcpyHostToDevice));
            // All workers enter each collective in identical order.
            const double dot=globalDotProduct(solver,comms[g],stream,d_a,d_b);
            double norm=std::numeric_limits<double>::quiet_NaN();
            globalNorm(norm,solver,comms[g],stream,d_a);
            // CPU reference: each global DOF counted ONCE.
            double referenceDot=0.0, referenceSq=0.0;
            for(int dof=0;dof<3*globalNodes;++dof) {
                const double x=avalue(dof,test),y=bvalue(dof,test);
                referenceDot+=x*y;
                referenceSq+=x*x;
            }
            const double referenceNorm=std::sqrt(referenceSq);
            const bool ok=near(dot,referenceDot)&&near(norm,referenceNorm);
            #pragma omp critical
            std::printf("test=%d gpu=%d dot=% .14e expected=% .14e norm=% .14e expected=% .14e %s\n",
                        test,g,dot,referenceDot,norm,referenceNorm,ok?"PASS":"FAIL");
            if(!ok) failed++;
        }
        CUDA_OK(cudaFree(d_a)); CUDA_OK(cudaFree(d_b));
        BLAS_OK(cublasDestroy(solver.handle));
        CUDA_OK(cudaStreamDestroy(stream));
    }
    for(int g=0;g<ngpu;++g) { CUDA_OK(cudaSetDevice(g)); NCCL_OK(ncclCommDestroy(comms[g])); }
    std::printf("OVERALL: %s (%d failures)\n",failed?"FAIL":"PASS",failed);
    return failed?1:0;
}
