#include "FftwToolkit.h"
struct MallocDeleter {
    void operator()(cufftDoubleReal* ptr) const {
        free(ptr);
    }
    void operator()(cufftDoubleComplex* ptr) const {
        free(ptr);
    }
};
__global__ void hatConvCalculate(cufftDoubleReal *Rsrc1,cufftDoubleReal *Rsrc2,int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<n)
	{
		Rsrc1[idx] = Rsrc1[idx]*Rsrc2[idx];
	}
}
void hatConv(scftData* sc1,cufftDoubleComplex *rslt, cufftDoubleComplex *src1, cufftDoubleComplex *src2)
{
    
    checkCudaErrors(cudaMemset(sc1->realGpu3,0,sizeof(cufftDoubleReal)*sc1->realDofs));
    checkCudaErrors(cudaMemset(sc1->realGpu2,0,sizeof(cufftDoubleReal)*sc1->realDofs));
    FftwC2R(src1,sc1->realGpu3,sc1);
    FftwC2R(src2,sc1->realGpu2,sc1);
	hatConvCalculate<<<(sc1->realDofs-1)/1024+1,1024>>>(sc1->realGpu3,sc1->realGpu2,sc1->realDofs);
    FftwR2C(sc1->realGpu3,rslt,sc1);

}
double Intergral_space(scftData* sc1,cufftDoubleComplex *src1, cufftDoubleComplex *src2)
{
    // cufftDoubleReal *Rsrc1,*Rsrc2;
    // cufftDoubleComplex *rslt,*rsltC;
    // checkCudaErrors(cudaMalloc((void**)&Rsrc1,sizeof(cufftDoubleReal)*sc1->realDofs));
    // checkCudaErrors(cudaMemset(Rsrc1,0,sizeof(cufftDoubleReal)*sc1->realDofs));
    // checkCudaErrors(cudaMalloc((void**)&Rsrc2,sizeof(cufftDoubleReal)*sc1->realDofs));
    // checkCudaErrors(cudaMemset(Rsrc2,0,sizeof(cufftDoubleReal)*sc1->realDofs));
    // checkCudaErrors(cudaMalloc((void**)&rslt,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
    // checkCudaErrors(cudaMemset(rslt,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
    // rsltC=(cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*1);
    // memset(rsltC,0,sizeof(cufftDoubleComplex)*1);

    checkCudaErrors(cudaMemset(sc1->realGpu3,0,sizeof(cufftDoubleReal)*sc1->realDofs));
    checkCudaErrors(cudaMemset(sc1->realGpu2,0,sizeof(cufftDoubleReal)*sc1->realDofs));
    checkCudaErrors(cudaMemset(sc1->cplxGpu2,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
    memset(sc1->cplxC1,0,sizeof(cufftDoubleComplex)*1);
    FftwC2R(src1,sc1->realGpu3,sc1);
    FftwC2R(src2,sc1->realGpu2,sc1);
	hatConvCalculate<<<(sc1->realDofs-1)/1024+1,1024>>>(sc1->realGpu3,sc1->realGpu2,sc1->realDofs);
    FftwR2C(sc1->realGpu3,sc1->cplxGpu2,sc1);
    checkCudaErrors(cudaMemcpy(sc1->cplxC1,sc1->cplxGpu2,sizeof(cufftDoubleComplex)*1,cudaMemcpyDeviceToHost));
    double val=sc1->cplxC1[0].x;
    // checkCudaErrors(cudaFree(rslt));
    // free(rsltC);
	// checkCudaErrors(cudaFree(Rsrc1));
	// checkCudaErrors(cudaFree(Rsrc2));
    return val;
    
}

// void FftwC2R(cufftDoubleComplex *Corig,cufftDoubleReal *Rrslt, scftData* sc1)
// {
  

//     // cufftDoubleComplex *Q_Ctmp;/sc1->cplxGpu1
//     // checkCudaErrors(cudaMalloc((void**)&Q_Ctmp,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
//     // checkCudaErrors(cudaMemset(Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
//     // cufftDoubleReal *Q_Rtmp;
//     // checkCudaErrors(cudaMalloc((void**)&Q_Rtmp,sizeof(cufftDoubleReal)*sc1->realDofs));
//     // checkCudaErrors(cudaMemset(Q_Rtmp,0,sizeof(cufftDoubleReal)*sc1->realDofs));
//     checkCudaErrors(cudaMemset(sc1->cplxGpu1,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
//     checkCudaErrors(cudaMemcpy(sc1->cplxGpu1,Corig,sizeof(cufftDoubleComplex)*sc1->cplxDofs,cudaMemcpyDeviceToDevice));
//     checkCudaErrors(cufftExecZ2D(sc1->plan,sc1->cplxGpu1,Rrslt));
//     // checkCudaErrors(cudaFree(Q_Ctmp));
//     // checkCudaErrors(cudaFree(Q_Rtmp));
// }
void FftwC2R(cufftDoubleComplex *Corig,cufftDoubleReal *Rrslt, scftData* sc1)
{


    // cufftDoubleComplex *Q_Ctmp;/sc1->cplxGpu1
    // checkCudaErrors(cudaMalloc((void**)&Q_Ctmp,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
    // checkCudaErrors(cudaMemset(Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
    // cufftDoubleReal *Q_Rtmp;
    // checkCudaErrors(cudaMalloc((void**)&Q_Rtmp,sizeof(cufftDoubleReal)*sc1->realDofs));
    // checkCudaErrors(cudaMemset(Q_Rtmp,0,sizeof(cufftDoubleReal)*sc1->realDofs));
    checkCudaErrors(cudaMemset(sc1->cplxGpu1,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
    checkCudaErrors(cudaMemcpy(sc1->cplxGpu1,Corig,sizeof(cufftDoubleComplex)*sc1->cplxDofs,cudaMemcpyDeviceToDevice));
    checkCudaErrors(cufftExecZ2D(sc1->plan,sc1->cplxGpu1,Rrslt));
    // checkCudaErrors(cudaMemcpy(Rrslt,sc1->realGpu1,sizeof(cufftDoubleReal)*sc1->realDofs,cudaMemcpyDeviceToDevice));
    // checkCudaErrors(cudaFree(Q_Ctmp));
    // checkCudaErrors(cudaFree(Q_Rtmp));
}

void FftwR2C(cufftDoubleReal *Rrslt, cufftDoubleComplex *Corig,scftData* sc1)
{
    
    double realDofs1=1.0/(double)sc1->realDofs;

    checkCudaErrors(cufftExecD2Z(sc1->plan1,Rrslt,Corig));
   
    checkCudaErrors(cublasZdscal(sc1->handle,sc1->cplxDofs,&realDofs1,Corig,1));
  
    // checkCudaErrors(cudaMemcpy(Corig,sc1->cplxGpu1,sizeof(cufftDoubleComplex)*sc1->cplxDofs,cudaMemcpyDeviceToDevice));
    // checkCudaErrors(cudaFree(Q_Ctmp));
    // checkCudaErrors(cudaFree(Q_Rtmp));
   
}


