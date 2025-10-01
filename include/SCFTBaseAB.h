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
double updateHamilton(scftData *sc1,scftVariable* var,FILE *fp0);