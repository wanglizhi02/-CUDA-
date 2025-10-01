\begin{enumerate}
\textbf{FftwToolkit.h} 文件定义了FFT变换相关的工具函数，用于频域和空间域之间的转换操作。
\textbf{Head.h} 文件包含了项目所需的所有头文件和一些基本定义，是项目的主要头文件。
\textbf{helper_cuda.h} 文件是NVIDIA CUDA工具包的一部分，提供了CUDA错误处理、设备查询等辅助功能。文件较长，仅展示部分内容。
\textbf{helper_string.h} 文件也是NVIDIA CUDA工具包的一部分，提供了字符串处理和命令行参数解析等功能。
\textbf{Initialization.h} 文件定义了数据初始化和结果输出相关的函数。
\textbf{Mytimer.h} 文件定义了一个计时器类，用于性能评估和时间测量。
\textbf{SCFTBaseAB.h} 文件定义了SCFT求解器的核心功能，包括求解扩散方程、更新场和能量计算等功能。
\textbf{BasicFunc.h} 文件定义了基础数学函数，主要包含复数线性组合和积分运算的CUDA核函数声明。
\textbf{scftData.h} 文件定义了SCFT计算所需的数据结构，包含网格、场、变换句柄等信息。
\textbf{scftVariable.h} 文件定义了SCFT计算过程中的变量结构，包含密度场、传播子等。
\end{enumerate}
## \subsubsection{BasicFunc.h}
\begin{lstlisting}
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
\end{lstlisting}

## \subsubsection{FftwToolkit.h}
\begin{lstlisting}
#pragma once
#include "Head.h"
#include <memory>
// 	//cufft的
void hatConv(scftData* sc1,cufftDoubleComplex *rslt, cufftDoubleComplex *src1, cufftDoubleComplex *src2);
double Intergral_space(scftData* sc1,cufftDoubleComplex *src1, cufftDoubleComplex *src2);
void FftwC2R(cufftDoubleComplex *Corig,cufftDoubleReal *Rrslt, scftData* sc1);
void FftwR2C(cufftDoubleReal *Rrslt, cufftDoubleComplex *Corig,scftData* sc1);
__global__ void hatConvCalculate(cufftDoubleReal *Rsrc1,cufftDoubleReal *Rsrc2,int n);

\end{lstlisting}

## \subsubsection{Head.h}
\begin{lstlisting}
#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <math.h>
#include "fftw3.h"
#include <cublas_v2.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <helper_cuda.h>
#include "scftData.h"
#include "scftVariable.h"
#include <fstream>
#include <memory>
#include <vector>
#define PI 3.14159265358979323846
\end{lstlisting}

## \subsubsection{Initialization.h}
\begin{lstlisting}
#ifndef __Initialization_h
#define __Initialization_h
#include <cufft.h>
#include "FftwToolkit.h"
// void initialize();
void initFieldFourier(int DimCpt,int cplxDofs,int ** indKspace,cufftDoubleComplex *field, const char *fname);
void writeRealData(scftData* sc1,cufftDoubleComplex *field,const char *fname);
__global__ void MatsAdd(double *dst, double d, double *src, double s, int n);
void MatPrint(double *matrix, int n,FILE* fp0);
__global__ void gradB_matrix(double *matirx,  int n,int m,double dh);
void writeEnergy(scftData *sc1,double diffEnergy,double internalEnergy,double entropicEnergy,double Energy);
void writeData(scftData *sc1,int *NCpt,int iterator,double *dirBox1,double diffEnergy,double resinftyW,double resinftyB,double singQ,double Energy,double internalEnergy,double entropicEnergy);
void writeRst1(scftData *sc1,double *dirBox1,double diffEnergy,double resinftyW,double resinftyB,double singQ,double Energy,double internalEnergy,double entropicEnergy);
void write_rho(double **RhoReal,  int iter,scftData* sc1);
#endif
\end{lstlisting}

## \subsubsection{Mytimer.h}
\begin{lstlisting}
#ifndef __mytimer_h_
#define __mytimer_h_

#include <sys/time.h>

class mytimer_t {
    private:
	long time_total;
	long time_previous;

	struct timeval time_start;

	struct timeval time_end;

    public:
	mytimer_t() : time_total(0), time_previous(0) {}

	void start() { gettimeofday(&time_start, NULL); }

	void pause() {
	    gettimeofday(&time_end, NULL);
	    time_previous = time_total;
	    time_total += (time_end.tv_sec - time_start.tv_sec) * 1000000 + (time_end.tv_usec - time_start.tv_usec);
	}

	void reset() { time_total = 0; time_previous = 0;}

	double get_current_time() { return time_total * 1e-6; }
	double get_previous_time() { return time_previous * 1e-6; }
};

#endif

\end{lstlisting}

## \subsubsection{SCFTBaseAB.h}
\begin{lstlisting}
#pragma once
#include "Head.h"
#include "scftData.h"
#include "FftwToolkit.h"
__global__ void order21_mde(double *Gsquare,cufftDoubleComplex*Q_Ctmp,int cplxDofs,double ds,double P);
__global__ void order22_mde(cufftDoubleReal *WReal,cufftDoubleReal *QReal,int realDofs,double ds,double P);
__global__ void adams(int n,cufftDoubleComplex *hatQ,cufftDoubleComplex *qRhs,cufftDoubleComplex *wqConv,double *Gsquare,double ds);
void MDESolver4Adams(double *range,double rangeN,scftData* sc1,cufftDoubleComplex *hatQ,cufftDoubleComplex* fieldW,int *index,
                    int* NCpt,double* Gsquare);
void MDESolver4AdamsPrintf(double *range,double rangeN,scftData* sc1,cufftDoubleComplex *hatQ,cufftDoubleComplex* fieldW,int *index,
                    int* NCpt,double* Gsquare);
void MDESolver2OrderPrintf(double *range,double rangeN,scftData* sc1,cufftDoubleComplex *hatQ,cufftDoubleComplex* fieldW,int *index,
                    int* NCpt,double* Gsquare);
void MDESolver2Order(double *range,double rangeN,scftData* sc1,cufftDoubleComplex *hatQ,cufftDoubleComplex* fieldW,int *index,
                    int* NCpt,double* Gsquare);
void write_rho(double **RhoReal,  int iter,scftData* sc1);
__global__ void getDirBox(double  *mat,int n);
__global__ void getRecipLattice(double* dBox,double* rBox, int dim);
__global__ void setOnes(double *data,int n);
__global__ void setSquare(double *data,int n);
void getGsquare(scftData *sc1);
void updatePropagator(scftData *sc1,scftVariable* var,int *NCpt);
// double updataeQ(cufftDoubleComplex *frdQGpu,int cplxDofs,int Ns);
double updataeQ(cufftDoubleComplex *frdQGpu,scftData *sc1);
void updateOrderParameter(scftData *sc1,scftVariable* var);
double calPartialF(struct scftData *sc1,scftVariable* var,int *NCpt);
void get_gradB(scftData *sc1,scftVariable* var,int *NCpt,double* gradB1,double dh);
void SimpleMixing(scftData *sc1,scftVariable* var,int *NCpt,double *resinftyW,double *resinftyB,FILE *fp0);
void updateField(scftData *sc1,scftVariable* var,int *NCpt,double *resinftyW,double *resinftyB,FILE *fp0);
double updateHamilton(scftData *sc1,scftVariable* var,FILE *fp0);\end{lstlisting}

## \subsubsection{scftData.h}
\begin{lstlisting}
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
\end{lstlisting}

## \subsubsection{scftVariable.h}
\begin{lstlisting}
#pragma once
#include "scftData.h"
#include <memory>
#include<vector>
struct scftVariable
{
    double*internalEnergy,*entropicEnergy,*singQ,*dirBox1;

	cufftDoubleComplex  *rhoGpu,
						*frdQGpu,
						*bakQGpu,
						*hatq_qplusGpu;

	cufftDoubleReal *rhorealGPU;
	double** rhoreal;
};
\end{lstlisting}

