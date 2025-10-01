#pragma once
#include <memory>
#include <vector>
#include "Head.h"
struct scftData
{
    int phase,dN,ItMax,n1,DimCpt,realDofs,cplxDofs,Nspecies,Nblock,Nblend,DimPhy,Ndeg,Ns;
    double dsMax,fA,fB,chi;
    double *rcpBoxGPU,*dirBoxGPU,*Gsquare,*indKspaceGPU;
    double *RangesN;
    double **Ranges;
    cublasHandle_t handle;
    cufftHandle plan,plan1;
    cufftDoubleComplex  *fieldWGPU0,*fieldWGPU1,*fieldWplus;
    cufftDoubleReal *realGpu2,*realGpu3;
    cufftDoubleComplex *cplxGpu1,*cplxGpu2,*cplxC1;
    double *fieldReal;
    cufftDoubleComplex *Q_Ctmp,*shuzu;
    cufftDoubleReal *Qreal,*WReal;
};
// struct scftData
// {
//     int Nspecies;
//     int DimCpt;
//     int cplxDofs;
//     int realDofs;
//     int Nblock;
// 	int Nblend;
//     int DimPhy;
// 	double ds;
    
// };
