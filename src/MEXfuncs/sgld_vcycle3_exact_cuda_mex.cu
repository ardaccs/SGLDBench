/*
 * sgld_vcycle3_exact_cuda_mex.cu
 *
 * Correctness-first 3-level CUDA V-cycle for SGLDBench.
 * Uses meshHierarchy_(2/3) transferMat, transferMatCoeffi,
 * solidNodeMapCoarser2Finer, multiGridOperatorRIdense and eNodMat.
 * The coarsest solve is delegated to a MATLAB callback so it can use the
 * same cholFac_/cholPermut_ path as Solving_Vcycle.
 *
 * [z1,r2,r3] = sgld_vcycle3_exact_cuda_mex(
 *   r1, mh2, mh3,
 *   diag1,fixed1,diag2,fixed2,diag3,fixed3,
 *   omega, coarseSolveHandle)
 */

#include "mex.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) \
  mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:cuda", \
  "CUDA error at %s:%d: %s",__FILE__,__LINE__,cudaGetErrorString(e)); } while(0)

#if __CUDA_ARCH__ < 600
__device__ double atomicAddD(double* a,double v){
  unsigned long long* p=(unsigned long long*)a, old=*p, assumed;
  do { assumed=old; old=atomicCAS(p,assumed,__double_as_longlong(v+__longlong_as_double(assumed))); }
  while(assumed!=old);
  return __longlong_as_double(old);
}
#else
__device__ double atomicAddD(double* a,double v){ return atomicAdd(a,v); }
#endif

struct THost {
  const int32_t* T;
  const double* coeff;
  const int32_t* solidMap;
  const double* op;
  const int32_t* eNod;
  int nPatch,nElem,nInter,nFine,nCoarse;
  int opPatchBy8;
};

struct TDev {
  int32_t* T=nullptr;
  double* coeff=nullptr;
  int32_t* solidMap=nullptr;
  double* op=nullptr;
  int32_t* eNod=nullptr;
  int nPatch=0,nElem=0,nInter=0,nFine=0,nCoarse=0,opPatchBy8=0;
};

static const mxArray* field(const mxArray* s,const char* name){
  if(!mxIsStruct(s)||mxGetNumberOfElements(s)!=1)
    mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:struct","Level inputs must be scalar structs.");
  const mxArray* a=mxGetField(s,0,name);
  if(!a) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:field","Missing field %s.",name);
  return a;
}

static THost parseLevel(const mxArray* s,int nFine,int nCoarse){
  THost h{};
  const mxArray* T=field(s,"transferMat");
  const mxArray* C=field(s,"transferMatCoeffi");
  const mxArray* S=field(s,"solidNodeMapCoarser2Finer");
  const mxArray* O=field(s,"multiGridOperatorRIdense");
  const mxArray* E=field(s,"eNodMat");
  const mxArray* N=field(s,"intermediateNumNodes");

  if(!mxIsInt32(T)||mxIsSparse(T)) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:T","transferMat must be full int32.");
  if(!mxIsDouble(C)||mxIsComplex(C)||mxIsSparse(C)) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:C","transferMatCoeffi must be full double.");
  if(!mxIsInt32(S)||mxIsSparse(S)) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:S","solidNodeMapCoarser2Finer must be full int32.");
  if(!mxIsDouble(O)||mxIsComplex(O)||mxIsSparse(O)) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:O","multiGridOperatorRIdense must be full double.");
  if(!mxIsInt32(E)||mxIsSparse(E)) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:E","eNodMat must be full int32.");

  h.nPatch=(int)mxGetM(T); h.nElem=(int)mxGetN(T); h.nInter=(int)mxGetScalar(N);
  h.nFine=nFine; h.nCoarse=nCoarse;
  if(mxGetNumberOfElements(C)!=(size_t)h.nInter) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:Csize","Bad transferMatCoeffi length.");
  if(mxGetNumberOfElements(S)!=(size_t)nFine) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:Ssize","solidNodeMapCoarser2Finer must have nFine entries.");
  if((int)mxGetM(E)!=h.nElem||mxGetN(E)<8) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:Esize","eNodMat must be [nElem x 8].");

  if((int)mxGetM(O)==h.nPatch&&mxGetN(O)==8) h.opPatchBy8=1;
  else if(mxGetM(O)==8&&(int)mxGetN(O)==h.nPatch) h.opPatchBy8=0;
  else mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:Osize","Operator must be [nPatch x 8] or [8 x nPatch].");

  h.T=(const int32_t*)mxGetData(T); h.coeff=mxGetDoubles(C);
  h.solidMap=(const int32_t*)mxGetData(S); h.op=mxGetDoubles(O);
  h.eNod=(const int32_t*)mxGetData(E);
  return h;
}

static unsigned char* fixedU8(const mxArray* a,int n){
  if(!mxIsLogical(a)||mxIsSparse(a)||mxGetNumberOfElements(a)!=(size_t)n)
    mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:fixed","fixedDOFs must be logical with matching length.");
  const mxLogical* p=mxGetLogicals(a);
  unsigned char* q=(unsigned char*)mxMalloc((mwSize)n);
  for(int i=0;i<n;++i) q[i]=p[i]?1:0;
  return q;
}

static void upload(const THost& h,TDev& d){
  d.nPatch=h.nPatch; d.nElem=h.nElem; d.nInter=h.nInter; d.nFine=h.nFine; d.nCoarse=h.nCoarse; d.opPatchBy8=h.opPatchBy8;
  size_t bT=sizeof(int32_t)*(size_t)h.nPatch*h.nElem;
  size_t bC=sizeof(double)*(size_t)h.nInter;
  size_t bS=sizeof(int32_t)*(size_t)h.nFine;
  size_t bO=sizeof(double)*(size_t)h.nPatch*8;
  size_t bE=sizeof(int32_t)*(size_t)h.nElem*8;
  CUDA_CHECK(cudaMalloc((void**)&d.T,bT)); CUDA_CHECK(cudaMemcpy(d.T,h.T,bT,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc((void**)&d.coeff,bC)); CUDA_CHECK(cudaMemcpy(d.coeff,h.coeff,bC,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc((void**)&d.solidMap,bS)); CUDA_CHECK(cudaMemcpy(d.solidMap,h.solidMap,bS,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc((void**)&d.op,bO)); CUDA_CHECK(cudaMemcpy(d.op,h.op,bO,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc((void**)&d.eNod,bE)); CUDA_CHECK(cudaMemcpy(d.eNod,h.eNod,bE,cudaMemcpyHostToDevice));
}

static void freeT(TDev& d){
  if(d.T) cudaFree(d.T); if(d.coeff) cudaFree(d.coeff); if(d.solidMap) cudaFree(d.solidMap);
  if(d.op) cudaFree(d.op); if(d.eNod) cudaFree(d.eNod); std::memset(&d,0,sizeof(d));
}

__device__ __forceinline__ double W(const double* op,int nPatch,int p,int a,int patchBy8){
  return patchBy8 ? op[p+a*nPatch] : op[a+p*8];
}

__global__ void zeroK(double* x,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)x[i]=0.0; }
__global__ void zeroFixedK(double* x,const unsigned char* f,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n&&f[i])x[i]=0.0; }
__global__ void jacobiAssignK(const double* r,const double* d,const unsigned char* f,double* z,double w,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) z[i]=f[i]?0.0:(d[i]!=0.0?w*r[i]/d[i]:0.0);
}
__global__ void jacobiAddK(const double* r,const double* d,const unsigned char* f,double* z,double w,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n&&!f[i]&&d[i]!=0.0)z[i]+=w*r[i]/d[i];
}

__global__ void scatterFineK(const double* fine,double* inter,const int32_t* map,int nFine){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nFine)return; int j=map[i]-1; if(j<0)return;
  inter[3*j]=fine[3*i]; inter[3*j+1]=fine[3*i+1]; inter[3*j+2]=fine[3*i+2];
}

__global__ void restrictTransposeK(const double* inter,double* coarse,const int32_t* T,const double* coeff,
  const double* op,const int32_t* eNod,int nPatch,int nElem,int nInter,int nCoarse,int patchBy8){
  int id=blockIdx.x*blockDim.x+threadIdx.x, total=nElem*8*3; if(id>=total)return;
  int c=id%3, q=id/3, a=q%8, e=q/8; int cn=eNod[e+a*nElem]-1; if(cn<0||cn>=nCoarse)return;
  double s=0.0; for(int p=0;p<nPatch;++p){ int j=T[p+e*nPatch]-1; if(j<0||j>=nInter||coeff[j]==0.0)continue;
    s += W(op,nPatch,p,a,patchBy8)*inter[3*j+c]/coeff[j]; }
  atomicAddD(&coarse[3*cn+c],s);
}

__global__ void interpElemK(const double* coarse,double* inter,const int32_t* T,const double* op,const int32_t* eNod,
  int nPatch,int nElem,int nInter,int nCoarse,int patchBy8){
  int id=blockIdx.x*blockDim.x+threadIdx.x, total=nElem*nPatch*3; if(id>=total)return;
  int c=id%3,q=id/3,p=q%nPatch,e=q/nPatch,j=T[p+e*nPatch]-1; if(j<0||j>=nInter)return;
  double v=0.0; for(int a=0;a<8;++a){ int cn=eNod[e+a*nElem]-1; if(cn>=0&&cn<nCoarse)v+=W(op,nPatch,p,a,patchBy8)*coarse[3*cn+c]; }
  atomicAddD(&inter[3*j+c],v);
}

__global__ void normalizeInterK(double* inter,const double* coeff,int nInter){
  int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=3*nInter)return; int j=id/3; inter[id]=coeff[j]!=0.0?inter[id]/coeff[j]:0.0;
}

__global__ void gatherAddK(const double* inter,double* fine,const int32_t* map,int nFine){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nFine)return; int j=map[i]-1; if(j<0)return;
  fine[3*i]+=inter[3*j]; fine[3*i+1]+=inter[3*j+1]; fine[3*i+2]+=inter[3*j+2];
}

static void restrictLevel(const double* fine,double* coarse,double* inter,const TDev& t,int th){
  int bi=(3*t.nInter+th-1)/th,bf=(t.nFine+th-1)/th,bc=(3*t.nCoarse+th-1)/th;
  int total=t.nElem*8*3,br=(total+th-1)/th;
  zeroK<<<bi,th>>>(inter,3*t.nInter); CUDA_CHECK(cudaGetLastError());
  scatterFineK<<<bf,th>>>(fine,inter,t.solidMap,t.nFine); CUDA_CHECK(cudaGetLastError());
  zeroK<<<bc,th>>>(coarse,3*t.nCoarse); CUDA_CHECK(cudaGetLastError());
  restrictTransposeK<<<br,th>>>(inter,coarse,t.T,t.coeff,t.op,t.eNod,t.nPatch,t.nElem,t.nInter,t.nCoarse,t.opPatchBy8);
  CUDA_CHECK(cudaGetLastError());
}

static void interpAddLevel(const double* coarse,double* fine,double* inter,const TDev& t,int th){
  int bi=(3*t.nInter+th-1)/th,bf=(t.nFine+th-1)/th,total=t.nElem*t.nPatch*3,bp=(total+th-1)/th;
  zeroK<<<bi,th>>>(inter,3*t.nInter); CUDA_CHECK(cudaGetLastError());
  interpElemK<<<bp,th>>>(coarse,inter,t.T,t.op,t.eNod,t.nPatch,t.nElem,t.nInter,t.nCoarse,t.opPatchBy8); CUDA_CHECK(cudaGetLastError());
  normalizeInterK<<<bi,th>>>(inter,t.coeff,t.nInter); CUDA_CHECK(cudaGetLastError());
  gatherAddK<<<bf,th>>>(inter,fine,t.solidMap,t.nFine); CUDA_CHECK(cudaGetLastError());
}

static void coarseSolve(const mxArray* handle,const double* dR,double* dZ,int n){
  mxArray* rhs=mxCreateDoubleMatrix((mwSize)n,1,mxREAL);
  CUDA_CHECK(cudaMemcpy(mxGetDoubles(rhs),dR,sizeof(double)*(size_t)n,cudaMemcpyDeviceToHost));
  mxArray* lhs[1]={nullptr}; mxArray* args[2]={const_cast<mxArray*>(handle),rhs};
  mxArray* trap=mexCallMATLABWithTrap(1,lhs,2,args,"feval"); mxDestroyArray(rhs);
  if(trap){ mxDestroyArray(trap); mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:coarse","Coarse callback failed."); }
  if(!lhs[0]||!mxIsDouble(lhs[0])||mxIsComplex(lhs[0])||mxIsSparse(lhs[0])||mxGetNumberOfElements(lhs[0])!=(size_t)n){
    if(lhs[0])mxDestroyArray(lhs[0]); mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:coarseOut","Coarse callback returned wrong vector."); }
  CUDA_CHECK(cudaMemcpy(dZ,mxGetDoubles(lhs[0]),sizeof(double)*(size_t)n,cudaMemcpyHostToDevice)); mxDestroyArray(lhs[0]);
}
void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]){
  if(nrhs!=11) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:nrhs","Expected 11 inputs.");
  if(nlhs>3) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:nlhs","Outputs: [z1,r2,r3].");

  const mxArray *r1=prhs[0],*mh2=prhs[1],*mh3=prhs[2],*d1m=prhs[3],*f1m=prhs[4],
                *d2m=prhs[5],*f2m=prhs[6],*d3m=prhs[7],*f3m=prhs[8],*handle=prhs[10];
  double omega=mxGetScalar(prhs[9]);
  if(!mxIsDouble(r1)||mxIsComplex(r1)||mxIsSparse(r1)) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:r1","r1 must be full double.");
  const mxArray* ds[3]={d1m,d2m,d3m};
  for(int i=0;i<3;++i) if(!mxIsDouble(ds[i])||mxIsComplex(ds[i])||mxIsSparse(ds[i]))
    mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:diag","diagK must be full double.");
  if(!mxIsClass(handle,"function_handle")) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:handle","Last input must be function handle.");

  int n1=(int)mxGetNumberOfElements(d1m), n2=(int)mxGetNumberOfElements(d2m), n3=(int)mxGetNumberOfElements(d3m);
  if(mxGetNumberOfElements(r1)!=(size_t)n1||n1%3||n2%3||n3%3) mexErrMsgIdAndTxt("sgld_vcycle3_exact_cuda:size","Bad DOF sizes.");
  int nn1=n1/3,nn2=n2/3,nn3=n3/3;
  THost h12=parseLevel(mh2,nn1,nn2), h23=parseLevel(mh3,nn2,nn3);
  unsigned char *hf1=fixedU8(f1m,n1),*hf2=fixedU8(f2m,n2),*hf3=fixedU8(f3m,n3);

  size_t b1=sizeof(double)*(size_t)n1,b2=sizeof(double)*(size_t)n2,b3=sizeof(double)*(size_t)n3;
  double *R1=nullptr,*Z1=nullptr,*R2=nullptr,*Z2=nullptr,*R3=nullptr,*Z3=nullptr,*D1=nullptr,*D2=nullptr,*D3=nullptr,*I12=nullptr,*I23=nullptr;
  unsigned char *F1=nullptr,*F2=nullptr,*F3=nullptr; TDev t12{},t23{};
#define MALLOC(P,B) CUDA_CHECK(cudaMalloc((void**)&(P),(B)))
  MALLOC(R1,b1); MALLOC(Z1,b1); MALLOC(R2,b2); MALLOC(Z2,b2); MALLOC(R3,b3); MALLOC(Z3,b3);
  MALLOC(D1,b1); MALLOC(D2,b2); MALLOC(D3,b3); MALLOC(F1,(size_t)n1); MALLOC(F2,(size_t)n2); MALLOC(F3,(size_t)n3);
  MALLOC(I12,sizeof(double)*(size_t)(3*h12.nInter)); MALLOC(I23,sizeof(double)*(size_t)(3*h23.nInter));
  upload(h12,t12); upload(h23,t23);
  CUDA_CHECK(cudaMemcpy(R1,mxGetDoubles(r1),b1,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(D1,mxGetDoubles(d1m),b1,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(D2,mxGetDoubles(d2m),b2,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(D3,mxGetDoubles(d3m),b3,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(F1,hf1,(size_t)n1,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(F2,hf2,(size_t)n2,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(F3,hf3,(size_t)n3,cudaMemcpyHostToDevice));
  mxFree(hf1); mxFree(hf2); mxFree(hf3);

  const int th=256,bk1=(n1+th-1)/th,bk2=(n2+th-1)/th,bk3=(n3+th-1)/th;
  jacobiAssignK<<<bk1,th>>>(R1,D1,F1,Z1,omega,n1); CUDA_CHECK(cudaGetLastError());
  restrictLevel(R1,R2,I12,t12,th); zeroFixedK<<<bk2,th>>>(R2,F2,n2); CUDA_CHECK(cudaGetLastError());
  jacobiAssignK<<<bk2,th>>>(R2,D2,F2,Z2,omega,n2); CUDA_CHECK(cudaGetLastError());
  restrictLevel(R2,R3,I23,t23,th); zeroFixedK<<<bk3,th>>>(R3,F3,n3); CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  coarseSolve(handle,R3,Z3,n3); zeroFixedK<<<bk3,th>>>(Z3,F3,n3); CUDA_CHECK(cudaGetLastError());
  interpAddLevel(Z3,Z2,I23,t23,th); zeroFixedK<<<bk2,th>>>(Z2,F2,n2); CUDA_CHECK(cudaGetLastError());
  jacobiAddK<<<bk2,th>>>(R2,D2,F2,Z2,omega,n2); CUDA_CHECK(cudaGetLastError());
  interpAddLevel(Z2,Z1,I12,t12,th); zeroFixedK<<<bk1,th>>>(Z1,F1,n1); CUDA_CHECK(cudaGetLastError());
  jacobiAddK<<<bk1,th>>>(R1,D1,F1,Z1,omega,n1); zeroFixedK<<<bk1,th>>>(Z1,F1,n1); CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  plhs[0]=mxCreateDoubleMatrix((mwSize)n1,1,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetDoubles(plhs[0]),Z1,b1,cudaMemcpyDeviceToHost));
  if(nlhs>=2){ plhs[1]=mxCreateDoubleMatrix((mwSize)n2,1,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetDoubles(plhs[1]),R2,b2,cudaMemcpyDeviceToHost)); }
  if(nlhs>=3){ plhs[2]=mxCreateDoubleMatrix((mwSize)n3,1,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetDoubles(plhs[2]),R3,b3,cudaMemcpyDeviceToHost)); }

  cudaFree(R1);cudaFree(Z1);cudaFree(R2);cudaFree(Z2);cudaFree(R3);cudaFree(Z3);cudaFree(D1);cudaFree(D2);cudaFree(D3);
  cudaFree(F1);cudaFree(F2);cudaFree(F3);cudaFree(I12);cudaFree(I23);freeT(t12);freeT(t23);
}
