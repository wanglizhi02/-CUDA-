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
