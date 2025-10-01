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
	FILE *fp; // = fopen(fname, "r");

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
		// if(i < 10)
		// printf("fieldInput[i].x=%f, fieldInput[i].y=%f\n", i, fieldInput[i].x, i, fieldInput[i].y);
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
					//if(j<10)
					//printf("field[%d][0]=%f, field[%d][1]=%f\n", i, field[i].x, i, field[i].y);

				}
			}
		}
	}
		

	fclose(fp);
	for(int i = 0; i < initDof; i++) free(fin[i]);
	free(fin);
	free(fieldInput);
}
	// writeRealData(sc1->realDofs,sc1->cplxDofs,sc1,fieldWGPU0,WReal,fname,NCpt);


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
    FILE *fp; // = fopen(fname, "r");
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
	// 
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

