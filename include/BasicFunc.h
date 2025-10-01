#ifndef __BasicFunc_h
#define __BasicFunc_h

#include "Head.h"
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void FuncsLinear1Cplx(cufftDoubleComplex *rslt, int n,
					  const double a1, const cufftDoubleComplex *F1);
__global__ void FuncsLinear2Cplx(cufftDoubleComplex *rslt, int n,
					  const double a1, const cufftDoubleComplex *F1,
					  const double a2, const cufftDoubleComplex *F2);
__global__ void FuncsLinear3Cplx(cufftDoubleComplex *rslt, int n,
					  const double a1, const cufftDoubleComplex *F1,
					  const double a2, const cufftDoubleComplex *F2,
					  const double a3, const cufftDoubleComplex *F3);
__global__ void FuncsLinear4Cplx(cufftDoubleComplex *rslt, int n,
					  const double a1, const cufftDoubleComplex *F1,
					  const double a2, const cufftDoubleComplex *F2,
					  const double a3, const cufftDoubleComplex *F3,
					  const double a4, const cufftDoubleComplex *F4);
__global__ void FuncsLinear5Cplx(cufftDoubleComplex *rslt, int n,
					  const double a1, const cufftDoubleComplex *F1,
					  const double a2, const cufftDoubleComplex *F2,
					  const double a3, const cufftDoubleComplex *F3,
					  const double a4, const cufftDoubleComplex *F4,
					  const double a5, const cufftDoubleComplex *F5);
__global__ void integration(cufftDoubleComplex *phi, 
                 cufftDoubleComplex *integrand,
                 int index,double singQ,int n,double ds ,int cplxDofs);	

double normRealInfty(cufftDoubleReal *src, int n);
				  					  					  					  					  
#endif
