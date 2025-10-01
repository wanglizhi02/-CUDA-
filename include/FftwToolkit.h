#pragma once
#include "Head.h"
#include <memory>
// 	//cufft的
void hatConv(scftData* sc1,cufftDoubleComplex *rslt, cufftDoubleComplex *src1, cufftDoubleComplex *src2);
double Intergral_space(scftData* sc1,cufftDoubleComplex *src1, cufftDoubleComplex *src2);
void FftwC2R(cufftDoubleComplex *Corig,cufftDoubleReal *Rrslt, scftData* sc1);
void FftwR2C(cufftDoubleReal *Rrslt, cufftDoubleComplex *Corig,scftData* sc1);
__global__ void hatConvCalculate(cufftDoubleReal *Rsrc1,cufftDoubleReal *Rsrc2,int n);


