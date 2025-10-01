#include "BasicFunc.h"
__global__ void FuncsLinear1Cplx(cufftDoubleComplex *rslt, int n,
					  const double a1, const cufftDoubleComplex *F1)
					  {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
	{
		rslt[i].x = a1 * F1[i].x;
		rslt[i].y = a1 * F1[i].y;
	}
					  }

__global__ void FuncsLinear2Cplx(cufftDoubleComplex *rslt, int n,
					  const double a1, const cufftDoubleComplex *F1,
					  const double a2, const cufftDoubleComplex *F2)
					  {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
	{
		rslt[i].x = a1 * F1[i].x + a2 * F2[i].x;
		rslt[i].y = a1 * F1[i].y + a2 * F2[i].y;
	}
					  }

__global__ void FuncsLinear3Cplx(cufftDoubleComplex *rslt, int n,
					  const double a1, const cufftDoubleComplex *F1,
					  const double a2, const cufftDoubleComplex *F2,
					  const double a3, const cufftDoubleComplex *F3)
					  {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
	{
		rslt[i].x = a1 * F1[i].x + a2 * F2[i].x+ a3 * F3[i].x;
		rslt[i].y = a1 * F1[i].y + a2 * F2[i].y+ a3 * F3[i].y;
	}
					  }

__global__ void FuncsLinear4Cplx(cufftDoubleComplex *rslt, int n,
					  const double a1, const cufftDoubleComplex *F1,
					  const double a2, const cufftDoubleComplex *F2,
					  const double a3, const cufftDoubleComplex *F3,
					  const double a4, const cufftDoubleComplex *F4)
					  {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
	{
		rslt[i].x = a1 * F1[i].x + a2 * F2[i].x+ a3 * F3[i].x+ a4 * F4[i].x;
		rslt[i].y = a1 * F1[i].y + a2 * F2[i].y+ a3 * F3[i].y+ a4 * F4[i].y;
	}
					  }

__global__ void FuncsLinear5Cplx(cufftDoubleComplex *rslt, int n,
					  const double a1, const cufftDoubleComplex *F1,
					  const double a2, const cufftDoubleComplex *F2,
					  const double a3, const cufftDoubleComplex *F3,
					  const double a4, const cufftDoubleComplex *F4,
					  const double a5, const cufftDoubleComplex *F5)
					  {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
	{
		rslt[i].x = a1 * F1[i].x + a2 * F2[i].x+ a3 * F3[i].x+ a4 * F4[i].x+ a5 * F5[i].x;
		rslt[i].y = a1 * F1[i].y + a2 * F2[i].y+ a3 * F3[i].y+ a4 * F4[i].y+ a5 * F5[i].y;
	}
					  }
__global__ void integration(cufftDoubleComplex *phi, 
                 cufftDoubleComplex *integrand,
                 int index,double singQ,int n,double ds ,int cplxDofs)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if(k < cplxDofs)
	{
        phi[k].x  = -0.625 * ((integrand+(index)*cplxDofs+k)->x+(integrand+(index+n)*cplxDofs+k)->x);
        phi[k].y  = -0.625 * ((integrand+(index)*cplxDofs+k)->y+(integrand+(index+n)*cplxDofs+k)->y);
        phi[k].x += 1.0/6.0 * ((integrand+(index+1)*cplxDofs+k)->x+(integrand+(index+n-1)*cplxDofs+k)->x);
        phi[k].y += 1.0/6.0 * ((integrand+(index+1)*cplxDofs+k)->y+(integrand+(index+n-1)*cplxDofs+k)->y);
        phi[k].x -= 1.0/24.0 * ((integrand+(index+2)*cplxDofs+k)->x+(integrand+(index+n-2)*cplxDofs+k)->x);
        phi[k].y -= 1.0/24.0 * ((integrand+(index+2)*cplxDofs+k)->y+(integrand+(index+n-2)*cplxDofs+k)->y);
  		for(int i = 0; i < n+1; i++)
        {
            phi[k].x += (integrand+(index+i)*cplxDofs+k)->x; 
            phi[k].y += (integrand+(index+i)*cplxDofs+k)->y; 
        }
	}
}

double normRealInfty(cufftDoubleReal *src, int n)
{
	double tmp;
	double rslt = 0.0;
	for(int i = 0; i < n; i++)
	{
		tmp = src[i];
		rslt = (rslt > tmp ? rslt : tmp);
	}
	return rslt;
}

