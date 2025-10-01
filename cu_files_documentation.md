
\begin{enumerate}
\item \textbf{BasicFunc.cu} 
实现基础数学函数和线性运算，包括复数线性组合和积分运算的CUDA核函数

\item \textbf{FftwToolkit.cu}
提供快速傅里叶变换相关工具函数，实现频域和空间域之间的转换操作

\item \textbf{Initialization.cu} 
实现数据初始化和结果输出功能，包括场初始化、数据写入和矩阵操作

\item \textbf{SCFTBaseAB.cu}
实现SCFT的核心算法和求解器，包括扩散方程求解、场更新和能量计算

\end{enumerate}
## \subsubsection{BasicFunc.cu}
\begin{lstlisting}
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
\end{lstlisting}

## \subsubsection{FftwToolkit.cu}
\begin{lstlisting}
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
    return val;
    
}

void FftwC2R(cufftDoubleComplex *Corig,cufftDoubleReal *Rrslt, scftData* sc1)
{
    checkCudaErrors(cudaMemset(sc1->cplxGpu1,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
    checkCudaErrors(cudaMemcpy(sc1->cplxGpu1,Corig,sizeof(cufftDoubleComplex)*sc1->cplxDofs,cudaMemcpyDeviceToDevice));
    checkCudaErrors(cufftExecZ2D(sc1->plan,sc1->cplxGpu1,Rrslt));
}

void FftwR2C(cufftDoubleReal *Rrslt, cufftDoubleComplex *Corig,scftData* sc1)
{
    
    double realDofs1=1.0/(double)sc1->realDofs;

    checkCudaErrors(cufftExecD2Z(sc1->plan1,Rrslt,Corig));
   
    checkCudaErrors(cublasZdscal(sc1->handle,sc1->cplxDofs,&realDofs1,Corig,1));
  
}
\end{lstlisting}

## \subsubsection{Initialization.cu}
\begin{lstlisting}
#include "Initialization.h"
#include <fstream>
#include "scftData.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "cublas_v2.h"
#include <memory>
void initFieldFourier(int DimCpt,int cplxDofs,int ** indKspace,cufftDoubleComplex *field, const char *fname)
{
	int ftmp;
	int initDof;
	FILE *fp;

	if((fp = fopen(fname, "r")) == NULL)
	{
		 printf("Cannot open file.\n");
		 exit(1);
	}
	else
    {
        printf("File Open Success !\n");
    }

	ftmp = fscanf(fp, "%d", &initDof);
	printf("initDof = %d\n", initDof);

	int **fin = (int **)malloc(sizeof(int*)*initDof);
	for(int i = 0; i < initDof; i++)
		fin[i] = (int *)malloc(sizeof(int)*DimCpt);

	cufftDoubleComplex *fieldInput = (cufftDoubleComplex* )malloc(sizeof(cufftDoubleComplex)*initDof);

	for(int i = 0; i < initDof; i++)
	{
		for(int j = 0; j < DimCpt; j++)
		{
			ftmp = fscanf(fp, "%d", &(fin[i][j]));
		}
		ftmp = fscanf(fp, "%lf", &(fieldInput[i].x));
		ftmp = fscanf(fp, "%lf", &(fieldInput[i].y));
	}
	for(int i = 0; i < cplxDofs; i ++)
	{
		for(int j = 0; j < initDof; j++)
		{
			if(DimCpt == 2)
			{
				if(fin[j][0]==indKspace[i][0] && fin[j][1]==indKspace[i][1])
				{
					field[i].x = fieldInput[j].x;
					field[i].y = fieldInput[j].y;
				}
			}
			
			if(DimCpt == 3)
			{
				if(fin[j][0]==indKspace[i][0] && fin[j][1]==indKspace[i][1] && fin[j][2]==indKspace[i][2])
				{
					field[i].x= fieldInput[j].x;
					field[i].y = fieldInput[j].y;
				}
			}
		}
	}
		

	fclose(fp);
	for(int i = 0; i < initDof; i++) free(fin[i]);
	free(fin);
	free(fieldInput);
}

__global__ void MatsAdd(double *dst, double d, double *src, double s, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
	{
		dst[i]=d*dst[i]+s*src[i];
	}
}
void MatPrint(double *matrix, int n,FILE *fp0)
{
	double *matrixC = (double *)malloc(sizeof(double)*n*n);
	memset(matrixC,0,sizeof(double)*n*n);
	checkCudaErrors(cudaMemcpy(matrixC,matrix,sizeof(double)*n*n,cudaMemcpyDeviceToHost));
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			fprintf(fp0,"matrix[%d][%d] = %.20f  ", i, j, matrixC[i*n+j]);
		}
		fprintf(fp0,"\n");
	}
	free(matrixC);
}

__global__ void gradB_matrix(double *matirx,  int n,int m,double dh)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n&&j<m)
	{
		if(i==j)
		{
			matirx[i*m+i]+=dh;
		}
	}
}
void writeRealData(scftData* sc1,cufftDoubleComplex *field, const char *fname)
{
    int ftmp;
    FILE *fp;
    if((fp = fopen(fname, "r")) == NULL)
    {   
         printf("Cannot open file!\n");
         exit(1);
    }

    for(int i = 0; i < sc1->realDofs; i++)
    {   
            ftmp = fscanf(fp,"%lf", &(sc1->fieldReal[i]));
			sc1->fieldReal[i]+=std::pow(10,-15);
    }	
    fclose(fp);
	checkCudaErrors(cudaMemcpy(sc1->realGpu3,sc1->fieldReal,sizeof(cufftDoubleReal)*sc1->realDofs,cudaMemcpyHostToDevice));
	FftwR2C(sc1->realGpu3,field,sc1);
}

void writeRst1(scftData *sc1,double *dirBox1,double diffEnergy,double resinftyW,double resinftyB,double singQ,double Energy,double internalEnergy,double entropicEnergy)
{
	char fname[255];
	FILE *fp2;
	sprintf(fname, "/home/kaijiang/wlz/AB3Dr/build/rst1[%d][%.4f.%.4f]-[%.2f].dat", sc1->phase, sc1->fA, sc1->fB, sc1->chi);
	fp2=fopen(fname,"a");
	if((fp2=fopen(fname,"a"))==NULL)
	{
        printf("can not open the Rst file\n");
	    exit(0);
    }
	fprintf(fp2, "%.4f\t  %.4f\t  %.4f\t  %.6e\t  %.6e\t  %.6e\t   %.7e\t  %.7e\t  %.7e\t  %.7e\t\n", 
			dirBox1[0*sc1->DimCpt+0], dirBox1[1*sc1->DimCpt+1],dirBox1[2*sc1->DimCpt+2], diffEnergy, resinftyW, resinftyB, singQ, Energy, internalEnergy, entropicEnergy);
	fclose(fp2);
}
void writeData(scftData *sc1,int *NCpt,int iterator,double *dirBox1,double diffEnergy,double resinftyW,double resinftyB,double singQ,double Energy,double internalEnergy,double entropicEnergy)
{
	char fname2[255];                                                           
	sprintf(fname2, "/home/kaijiang/wlz/AB3Dr/build/Data[%d][%.4f.%.4f]-[%.2f].dat", sc1->phase, sc1->fA, sc1->fB, sc1->chi);
	FILE *fp2=fopen(fname2,"w");     
	if((fp2=fopen(fname2,"w"))==NULL)
	{
        printf("can not open the Data file\n");
	    exit(0);
    }
	fprintf(fp2," %d\t %d\t %d\t  %.5f\t %d\t  %d\n", NCpt[0], NCpt[1], NCpt[2],sc1->dsMax, sc1->phase, iterator);
	fprintf(fp2,"%.4f\t %.4f\t  \n", sc1->fA, sc1->fB);
	fprintf(fp2,"%.2f\t \n", sc1->chi);
	fprintf(fp2,"%.4f\t %.4f\t %.4f\t \n", dirBox1[0*sc1->DimCpt+0], dirBox1[1*sc1->DimCpt+1],dirBox1[2*sc1->DimCpt+2]);
	fprintf(fp2, "%.20e\t %.20e\n",diffEnergy, singQ);
	fprintf(fp2, "%.20e\t %.20e\t\n",resinftyW,  resinftyB);
	fprintf(fp2, "%.20e\t %.20e\t %.20e\n",Energy, internalEnergy, entropicEnergy);
	fclose(fp2);  
	
}
void writeEnergy(scftData *sc1,double diffEnergy,double internalEnergy,double entropicEnergy,double Energy)
{
	char fname3[255];                                                            
	sprintf(fname3, "/home/kaijiang/wlz/AB3Dr/build/energy.[phase=%d].[fB=%.2f]-chiAB=%.2f.dat", sc1->phase, sc1->fB, sc1->chi);
	FILE *fp2=fopen(fname3,"a");   
	if((fp2=fopen(fname3,"a"))==NULL)
	{
        printf("can not open the Energy file\n");
	    exit(0);
    }  
    fprintf(fp2, "%.3f \t %.20e\t %.20e\t %.20e\n",sc1->fB,internalEnergy, entropicEnergy, Energy);
	fclose(fp2);  

}
void write_rho(double **RhoReal,  int iter,scftData* sc1)
{                                                                               
    char fname2[255];   
    sprintf(fname2,"/home/kaijiang/wlz/AB3Dr/build/rho1[%d][%.4f.%.4f].[%.2f].[%d].txt",sc1->phase, sc1->fA, sc1->fB, sc1->chi, iter);
	FILE*fp2=fopen(fname2,"w");  
    if((fp2=fopen(fname2,"w"))==NULL)
    {
        printf("can not open the rho1 file\n");
        exit(0);
    }                                                            
     
    for(int i = 0; i < sc1->realDofs; i++)
    {
        fprintf(fp2,"%lf %lf \n", RhoReal[0][i], RhoReal[1][i]);
    }
    fclose(fp2);                                                                 
}
\end{lstlisting}

## \subsubsection{SCFTBaseAB.cu}
\begin{lstlisting}
#include "SCFTBaseAB.h"
#include "BasicFunc.h"
#include "FftwToolkit.h"
#include "scftData.h"
#include <stdio.h>
#include"Initialization.h"
#include<memory>
#include "Mytimer.h"
__global__ void order21_mde(double *Gsquare,cufftDoubleComplex*Q_Ctmp,int cplxDofs,double ds,double P)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<cplxDofs)
	{
		Q_Ctmp[idx].x *= exp(-Gsquare[idx]*ds/(2.0*P));
		Q_Ctmp[idx].y *= exp(-Gsquare[idx]*ds/(2.0*P));
	}
}
__global__ void order22_mde(cufftDoubleReal *WReal,cufftDoubleReal *QReal,int realDofs,double ds,double P)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<realDofs)
	{
		QReal[idx] *= exp(-WReal[idx]*ds/P);
	}
}
__global__ void adams4Linear(cufftDoubleComplex *rslt1,cufftDoubleComplex *rslt2, int n, const cufftDoubleComplex *hatQ, int index, int cplxDofs,int i)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k<n)
    {
        rslt1[k].x= 4.0*( hatQ+(index+i)*cplxDofs+k)->x-6.0*( hatQ+(index+i-1)*cplxDofs+k)->x+4.0*( hatQ+(index+i-2)*cplxDofs+k)->x-1.0*(hatQ+(index+i-3)*cplxDofs+k)->x;
        rslt1[k].y= 4.0*( hatQ+(index+i)*cplxDofs+k)->y-6.0*( hatQ+(index+i-1)*cplxDofs+k)->y+4.0*( hatQ+(index+i-2)*cplxDofs+k)->y-1.0*(hatQ+(index+i-3)*cplxDofs+k)->y;
        rslt2[k].x= 4.0*( hatQ+(index+i)*cplxDofs+k)->x-3.0*( hatQ+(index+i-1)*cplxDofs+k)->x+ 4.0/3.0*( hatQ+(index+i-2)*cplxDofs+k)->x-1.0/4.0*(hatQ+(index+i-3)*cplxDofs+k)->x;
        rslt2[k].y= 4.0*( hatQ+(index+i)*cplxDofs+k)->y-3.0*( hatQ+(index+i-1)*cplxDofs+k)->y+ 4.0/3.0*( hatQ+(index+i-2)*cplxDofs+k)->y-1.0/4.0*(hatQ+(index+i-3)*cplxDofs+k)->y;

    }

}
__global__ void adams(int n,cufftDoubleComplex *hatQ,cufftDoubleComplex *qRhs,cufftDoubleComplex *wqConv,double *Gsquare,double ds)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k<n)
    {
        hatQ[k].x=(qRhs[k].x-ds*wqConv[k].x)/ (25.0/12.0 + Gsquare[k]*ds);
        hatQ[k].y=(qRhs[k].y-ds*wqConv[k].y)/ (25.0/12.0 + Gsquare[k]*ds);

    }

}

__global__ void orderRich(int n,cufftDoubleComplex *hatQ,cufftDoubleComplex *q1,cufftDoubleComplex *q2)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k<n)
    {
        hatQ[k].x= -1.0/3.0*q1[k].x+4.0/3.0*q2[k].x;
        hatQ[k].y= -1.0/3.0*q1[k].y+4.0/3.0*q2[k].y;

    }

}
void MDESolver4Adams(double *range,double rangeN,scftData* sc1,cufftDoubleComplex *hatQ,cufftDoubleComplex* fieldW,int *index,
                    int* NCpt,double* Gsquare)
{
    double ds = (range[1] - range[0]) / rangeN;
    double realDofs1=1.0/(double)sc1->realDofs;
    
    checkCudaErrors(cudaMemset(sc1->Qreal,0,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->WReal,0,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
	checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*sc1->cplxDofs));

    int realDofs=sc1->realDofs;
    int cplxDofs=sc1->cplxDofs;
    
    FftwC2R(fieldW,sc1->WReal,sc1);
    int index1=*index;
   
    for(int i=0;i<rangeN;i++)
    {
        if(i<3)
        {
             checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*cplxDofs));
            for(int t=0;t<2;t++)
            {
                checkCudaErrors(cudaMemcpy(sc1->Q_Ctmp,hatQ+(*index+i)*cplxDofs,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToDevice));
                double P=std::pow(2.0,t);
                for(int j=0;j<P;j++)
                {
                    order21_mde<<<(cplxDofs-1)/1024+1,1024>>>(Gsquare,sc1->Q_Ctmp,cplxDofs,ds,P);

                    FftwC2R(sc1->Q_Ctmp,sc1->Qreal,sc1);

                    order22_mde<<<(realDofs-1)/1024+1,1024>>>(sc1->WReal,sc1->Qreal,realDofs,ds,P);

                    FftwR2C(sc1->Qreal,sc1->Q_Ctmp,sc1);

                    order21_mde<<<(cplxDofs-1)/1024+1,1024>>>(Gsquare,sc1->Q_Ctmp,cplxDofs,ds,P);

                }   
                checkCudaErrors(cudaMemcpy(sc1->shuzu+t*cplxDofs,sc1->Q_Ctmp,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToDevice));

            }
        
            double Linear2index1=-1.0/3.0;
            double Linear2index2=4.0/3.0;

            FuncsLinear2Cplx<<<(cplxDofs-1)/1024+1,1024>>>(hatQ+(*index+i+1)*cplxDofs,cplxDofs,Linear2index1,sc1->shuzu,Linear2index2,sc1->shuzu+cplxDofs);
        }
        else  
        {
	        checkCudaErrors(cudaMemset(sc1->Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
            checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*sc1->cplxDofs));
            adams4Linear<<<(cplxDofs-1)/1024+1,1024>>>(sc1->shuzu,sc1->Q_Ctmp,cplxDofs,hatQ, *index,cplxDofs,i);
            hatConv(sc1,sc1->shuzu+1*cplxDofs,fieldW, sc1->shuzu);
            adams<<<(cplxDofs-1)/1024+1,1024>>>(cplxDofs,hatQ+(index1+i+1)*cplxDofs,sc1->Q_Ctmp,sc1->shuzu+1*cplxDofs,Gsquare,ds);

        }
    }
    *index+=rangeN;
}

void updatePropagator(scftData *sc1,scftVariable* var,int *NCpt)
{
	int it_q =0;
	int it_qplus =0;
	cufftDoubleComplex *frdQGpuInit,*bakQGpuInit;
	frdQGpuInit = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*1);
	bakQGpuInit = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*1);
	memset(frdQGpuInit, 0, sizeof(cufftDoubleComplex)*1);
	memset(bakQGpuInit, 0, sizeof(cufftDoubleComplex)*1);
	frdQGpuInit[0].x=1.0;
	frdQGpuInit[0].y=0.0;

	bakQGpuInit[0].x=1.0;
	bakQGpuInit[0].y=0.0;

	checkCudaErrors(cudaMemcpy(var->frdQGpu,frdQGpuInit,sizeof(cufftDoubleComplex)*1,cudaMemcpyHostToDevice));
	MDESolver4Adams(sc1->Ranges[0],sc1->RangesN[0],sc1,var->frdQGpu,sc1->fieldWGPU0, &it_q,NCpt,sc1->Gsquare);
    MDESolver4Adams(sc1->Ranges[1],sc1->RangesN[1],sc1,var->frdQGpu,sc1->fieldWGPU1, &it_q,NCpt,sc1->Gsquare);

	checkCudaErrors(cudaMemcpy(var->bakQGpu,bakQGpuInit,sizeof(cufftDoubleComplex)*1,cudaMemcpyHostToDevice));
    MDESolver4Adams(sc1->Ranges[1],sc1->RangesN[1],sc1,var->bakQGpu,sc1->fieldWGPU1, &it_qplus,NCpt,sc1->Gsquare);
    MDESolver4Adams(sc1->Ranges[0],sc1->RangesN[0],sc1,var->bakQGpu,sc1->fieldWGPU0, &it_qplus,NCpt,sc1->Gsquare);

	free(frdQGpuInit);
	free(bakQGpuInit);
}

double updataeQ(cufftDoubleComplex *frdQGpu,scftData *sc1)
{
	checkCudaErrors(cudaDeviceSynchronize());	
    memset(sc1->cplxC1,0,sizeof(cufftDoubleComplex)*1);
	checkCudaErrors(cudaMemcpy(sc1->cplxC1,frdQGpu+(sc1->cplxDofs)*(sc1->Ns-1),sizeof(cufftDoubleComplex)*1,cudaMemcpyDeviceToHost));
	double tt=sc1->cplxC1[0].x;
	return tt;
}

void updateOrderParameter(scftData *sc1,scftVariable* var)
{
	for(int i = 0; i < sc1->Ns; i++)
	{
		int it_inv1 = sc1->Ns-1-i;
		hatConv(sc1,var->hatq_qplusGpu+(sc1->cplxDofs)*i,var->frdQGpu+(sc1->cplxDofs)*i,var->bakQGpu+(sc1->cplxDofs)*it_inv1);
	}
   
	int index = 0;
	for(int i = 0; i < sc1->Nspecies; i++)
	{
	double ds=(sc1->Ranges[i][1]-sc1->Ranges[i][0])/sc1->RangesN[i];
	integration<<<(sc1->cplxDofs-1)/1024+1,1024>>>(var->rhoGpu+(sc1->cplxDofs)*i, var->hatq_qplusGpu, index, var->singQ[0],sc1->RangesN[i],ds,sc1->cplxDofs);
    double rhoIndex=ds/var->singQ[0];
    cublasZdscal(sc1->handle,sc1->cplxDofs,&rhoIndex,var->rhoGpu+(sc1->cplxDofs)*i,1);
	index+=sc1->RangesN[i];
	}
}

double calPartialF(struct scftData *sc1,scftVariable* var,int *NCpt)
{
	double partialF = 0;
	getGsquare(sc1);
	updatePropagator(sc1,var,NCpt);
    double Q=updataeQ(var->frdQGpu,sc1);
	partialF -= std::log(Q);
	return partialF;
}

void updateField(scftData *sc1,scftVariable* var,int *NCpt,double *resinftyW,double *resinftyB,FILE *fp0)
{
	SimpleMixing(sc1,var,NCpt,resinftyW,resinftyB,fp0);
}

double updateHamilton(scftData *sc1,scftVariable* var,FILE *fp0)
{
	double tmpAB,tmpA, tmpB, tmpWplus,internalEnergy1,entropicEnergy1;
	checkCudaErrors(cudaMemset(sc1->Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
	tmpAB = Intergral_space(sc1,var->rhoGpu+(sc1->cplxDofs)*0,var->rhoGpu+(sc1->cplxDofs)*1);

	tmpA = Intergral_space(sc1,sc1->fieldWGPU0,var->rhoGpu+(sc1->cplxDofs)*0);

	tmpB = Intergral_space(sc1,sc1->fieldWGPU1,var->rhoGpu+(sc1->cplxDofs)*1);
	FuncsLinear2Cplx<<<(sc1->cplxDofs-1)/1024+1,1024>>>(sc1->Q_Ctmp,sc1->cplxDofs,1.0,var->rhoGpu+(sc1->cplxDofs)*0,1.0,var->rhoGpu+(sc1->cplxDofs)*1);
	tmpWplus = Intergral_space(sc1,sc1->fieldWplus,sc1->Q_Ctmp);
	internalEnergy1 = sc1->chi*sc1->Ndeg*tmpAB - tmpA -tmpB;
	entropicEnergy1=var->entropicEnergy[0];
	entropicEnergy1= - std::log(var->singQ[0]);
	double Energy= internalEnergy1 + entropicEnergy1;
	var->internalEnergy[0]=internalEnergy1;
	var->entropicEnergy[0]=entropicEnergy1;
	fprintf(fp0,"H = %.20f\t %.20f\t \n", internalEnergy1, sc1->chi*sc1->Ndeg*tmpAB); 
	return Energy;
}
\end{lstlisting}
