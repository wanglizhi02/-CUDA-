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
    
    // std::unique_ptr<double[]> doubleC = std::make_unique<double[]>(sc1->cplxDofs);
	// std::fill_n(doubleC.get(), sc1->cplxDofs, 0);
    checkCudaErrors(cudaMemset(sc1->Qreal,0,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->WReal,0,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
	checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*sc1->cplxDofs));
    	//cufft的

    int realDofs=sc1->realDofs;
    int cplxDofs=sc1->cplxDofs;
    
    FftwC2R(fieldW,sc1->WReal,sc1);
    int index1=*index;
   
    // for(int i=0;i<rangeN;i++)
    for(int i=0;i<rangeN;i++)
    {
        if(i<3)
        {
             checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*cplxDofs));
            // for(int t=0;t<2;t++)
            for(int t=0;t<2;t++)
            {
                checkCudaErrors(cudaMemcpy(sc1->Q_Ctmp,hatQ+(*index+i)*cplxDofs,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToDevice));
                double P=std::pow(2.0,t);
                // for(int j=0;j<P;j++)
                for(int j=0;j<P;j++)
                {
                    order21_mde<<<(cplxDofs-1)/1024+1,1024>>>(Gsquare,sc1->Q_Ctmp,cplxDofs,ds,P);

                    FftwC2R(sc1->Q_Ctmp,sc1->Qreal,sc1);

                    // order22_mde<<<(cplxDofs-1)/1024+1,1024>>>(sc1->WReal,sc1->Qreal,realDofs,ds,P);
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
            // FuncsLinear4Cplx<<<(cplxDofs-1)/1024+1,1024>>>(shuzutt           ,cplxDofs, 4.0, hatQ+(*index+i)*cplxDofs, -6.0, hatQ+(*index+i-1)*cplxDofs,
            //                                                                4.0, hatQ+(*index+i-2)*cplxDofs, -1.0,hatQ+(*index+i-3)*cplxDofs);

            // FuncsLinear4Cplx<<<(cplxDofs-1)/1024+1,1024>>>(sc1->Q_Ctmp,cplxDofs, 4.0, hatQ+(*index+i)*cplxDofs, -3.0,  hatQ+(*index+i-1)*cplxDofs,
            //                                                           4.0/3.0,  hatQ+(*index+i-2)*cplxDofs,  -1.0/4.0,hatQ+(*index+i-3)*cplxDofs);
            adams4Linear<<<(cplxDofs-1)/1024+1,1024>>>(sc1->shuzu,sc1->Q_Ctmp,cplxDofs,hatQ, *index,cplxDofs,i);
            hatConv(sc1,sc1->shuzu+1*cplxDofs,fieldW, sc1->shuzu);
            adams<<<(cplxDofs-1)/1024+1,1024>>>(cplxDofs,hatQ+(index1+i+1)*cplxDofs,sc1->Q_Ctmp,sc1->shuzu+1*cplxDofs,Gsquare,ds);

        }
  

    }
    *index+=rangeN;
	
   
}
void MDESolver4AdamsPrintf(double *range,double rangeN,scftData* sc1,cufftDoubleComplex *hatQ,cufftDoubleComplex* fieldW,int *index,
                    int* NCpt,double* Gsquare)
{
    double ds = (range[1] - range[0]) / rangeN;
    double realDofs1=1.0/(double)sc1->realDofs;
    
    // std::unique_ptr<double[]> doubleC = std::make_unique<double[]>(sc1->cplxDofs);
	// std::fill_n(doubleC.get(), sc1->cplxDofs, 0);
    checkCudaErrors(cudaMemset(sc1->Qreal,0,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->WReal,0,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
	checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*sc1->cplxDofs));
    // cufftDoubleComplex* shuzutt,* shuzutt1;
    // checkCudaErrors(cudaMalloc((void**)&shuzutt,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
	// checkCudaErrors(cudaMemset(shuzutt,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
    // checkCudaErrors(cudaMalloc((void**)&shuzutt1,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
	// checkCudaErrors(cudaMemset(shuzutt1,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
    cufftDoubleComplex *cplxC=(cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*sc1->cplxDofs);
    memset(cplxC,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs);    
    char fname1[]="/home/kaijiang/wlz/AB3Dr/out/BDF4pro.txt";
 	FILE *fp0=fopen(fname1,"w+");  
    if((fp0=fopen(fname1,"w+"))==NULL)
	{
        printf("can not open the out file\n");
	    exit(0);
    }
    	//cufft的

    int realDofs=sc1->realDofs;
    int cplxDofs=sc1->cplxDofs;
    
    FftwC2R(fieldW,sc1->WReal,sc1);
    int index1=*index;
   
    // for(int i=0;i<rangeN;i++)
    for(int i=0;i<4;i++)
    {
        if(i<3)
        {
             checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*cplxDofs));
            // for(int t=0;t<2;t++)
            for(int t=0;t<2;t++)
            {
                checkCudaErrors(cudaMemcpy(sc1->Q_Ctmp,hatQ+(*index+i)*cplxDofs,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToDevice));
                double P=std::pow(2.0,t);
                // for(int j=0;j<P;j++)
                for(int j=0;j<P;j++)
                {
                    order21_mde<<<(cplxDofs-1)/1024+1,1024>>>(Gsquare,sc1->Q_Ctmp,cplxDofs,ds,P);

                    FftwC2R(sc1->Q_Ctmp,sc1->Qreal,sc1);

                    // order22_mde<<<(cplxDofs-1)/1024+1,1024>>>(sc1->WReal,sc1->Qreal,realDofs,ds,P);
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
            // FuncsLinear4Cplx<<<(cplxDofs-1)/1024+1,1024>>>(shuzutt           ,cplxDofs, 4.0, hatQ+(*index+i)*cplxDofs, -6.0, hatQ+(*index+i-1)*cplxDofs,
            //                                                                4.0, hatQ+(*index+i-2)*cplxDofs, -1.0,hatQ+(*index+i-3)*cplxDofs);

            // FuncsLinear4Cplx<<<(cplxDofs-1)/1024+1,1024>>>(sc1->Q_Ctmp,cplxDofs, 4.0, hatQ+(*index+i)*cplxDofs, -3.0,  hatQ+(*index+i-1)*cplxDofs,
            //                                                           4.0/3.0,  hatQ+(*index+i-2)*cplxDofs,  -1.0/4.0,hatQ+(*index+i-3)*cplxDofs);
            adams4Linear<<<(cplxDofs-1)/1024+1,1024>>>(sc1->shuzu,sc1->Q_Ctmp,cplxDofs,hatQ, *index,cplxDofs,i);
            hatConv(sc1,sc1->shuzu+1*cplxDofs,fieldW, sc1->shuzu);
            adams<<<(cplxDofs-1)/1024+1,1024>>>(cplxDofs,hatQ+(index1+i+1)*cplxDofs,sc1->Q_Ctmp,sc1->shuzu+1*cplxDofs,Gsquare,ds);

        }
        checkCudaErrors(cudaMemcpy(cplxC,hatQ+(*index+i+1)*cplxDofs,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
        for(int j=0;j<20;j++)
        fprintf(fp0,"%d %.20f %.20f\n",*index+i,cplxC[j].x,cplxC[j].y);
        
        fprintf(fp0,"=========================================\n");
       
  

    }
    *index+=rangeN;
	// checkCudaErrors(cudaFree(shuzutt));
	// checkCudaErrors(cudaFree(shuzutt1));
	fclose(fp0);
    free(cplxC);


    
   
   
}

   


	// MDESolver2Order(Ranges[0],RangesN[0],sc1,sc1->cplxDofs,sc1->realDofs,fieldWGPU1,frdQGpu,WReal, &it_q,NCpt,Gsquare);

void MDESolver2OrderPrintf(double *range,double rangeN,scftData* sc1,cufftDoubleComplex *hatQ,cufftDoubleComplex* fieldW,int *index,
                    int* NCpt,double* Gsquare)
{
    double ds = (range[1] - range[0]) / rangeN;
    double realDofs1=1.0/(double)sc1->realDofs;
    int ifprintf=0;
    // std::unique_ptr<double[]> doubleC = std::make_unique<double[]>(sc1->cplxDofs);
	// std::fill_n(doubleC.get(), sc1->cplxDofs, 0);
    checkCudaErrors(cudaMemset(sc1->Qreal,0,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->WReal,0,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
	checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*sc1->cplxDofs));
     cufftDoubleComplex *shuzuC=(cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*sc1->cplxDofs);
    memset(shuzuC,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs);
     cufftDoubleReal *shuzuCR=(cufftDoubleReal*)malloc(sizeof(cufftDoubleReal)*sc1->realDofs);
    memset(shuzuCR,0,sizeof(cufftDoubleReal)*sc1->realDofs);    
    char fname1[]="/home/kaijiang/wlz/AB3Dr/out/data_2order_final1.txt";
 	FILE *fp2=fopen(fname1,"w+");  
    if((fp2=fopen(fname1,"w+"))==NULL)
	{
        printf("can not open the out file\n");
	    exit(0);
    }


    	//cufft的

    int realDofs=sc1->realDofs;
    int cplxDofs=sc1->cplxDofs;
    ifprintf=0;
    if(ifprintf)
    {
         fprintf(fp2, "realDofs %d\n",realDofs);
    }
    ifprintf=0;

    
    FftwC2R(fieldW,sc1->WReal,sc1);
        ifprintf=0;
    if(ifprintf)
    {
        checkCudaErrors(cudaMemcpy(shuzuC,fieldW,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
        for(int i=0;i<20;i++)
        {

            fprintf(fp2, "fieldW %.20f %.20f\n",shuzuC[i].x,shuzuC[i].y);
        }
        fprintf(fp2,"---------------\n");
        checkCudaErrors(cudaMemcpy(shuzuCR,sc1->WReal,sizeof(cufftDoubleReal)*realDofs,cudaMemcpyDeviceToHost));
        for(int i=0;i<20;i++)
        {
        fprintf(fp2, "WReal %.20f\n",shuzuCR[i]);
        }
        fprintf(fp2,"-------------------\n");
    }
    // ifprintf=0;

   
    // for(int i=0;i<rangeN;i++)
    for(int i=0;i<3;i++)
    {
        checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*cplxDofs));
        // for(int t=0;t<2;t++)
        for(int t=0;t<2;t++)
        {
            checkCudaErrors(cudaMemcpy(sc1->Q_Ctmp,hatQ+(*index+i)*cplxDofs,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToDevice));
            double P=std::pow(2.0,t);
            // for(int j=0;j<P;j++)
            for(int j=0;j<P;j++)
            {
                order21_mde<<<(cplxDofs-1)/1024+1,1024>>>(Gsquare,sc1->Q_Ctmp,cplxDofs,ds,P);
                 if(ifprintf)
                 {
                    checkCudaErrors(cudaMemcpy(shuzuC,sc1->Q_Ctmp,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
                    for(int i=0;i<20;i++)
                    {

                        fprintf(fp2, "tmp1 %.20f %.20f\n",shuzuC[i].x,shuzuC[i].y);
                    }
                    fprintf(fp2,"---------------\n");
                 }

                FftwC2R(sc1->Q_Ctmp,sc1->Qreal,sc1);
                    if(ifprintf)
                 {
                    checkCudaErrors(cudaMemcpy(shuzuCR,sc1->Qreal,sizeof(cufftDoubleReal)*realDofs,cudaMemcpyDeviceToHost));
                    for(int i=0;i<20;i++)
                    {
                        fprintf(fp2,"Qreal1 %.20f\n",shuzuCR[i]);
                    }
                    fprintf(fp2,"---------------\n");
                 }


                // order22_mde<<<(cplxDofs-1)/1024+1,1024>>>(sc1->WReal,sc1->Qreal,realDofs,ds,P);
                order22_mde<<<(realDofs-1)/1024+1,1024>>>(sc1->WReal,sc1->Qreal,realDofs,ds,P);
                 if(ifprintf)
                 {
                    checkCudaErrors(cudaMemcpy(shuzuCR,sc1->Qreal,sizeof(cufftDoubleReal)*realDofs,cudaMemcpyDeviceToHost));
                    for(int i=0;i<20;i++)
                    {
                        // fprintf(fp2,"Qreal2 %.20f\n",shuzuCR[realDofs-1-i]);
                        fprintf(fp2,"Qreal2 %.20f\n",shuzuCR[i]);
                    }
                    fprintf(fp2,"---------------\n");
                 }

                FftwR2C(sc1->Qreal,sc1->Q_Ctmp,sc1);
                                 if(ifprintf)
                 {
                    checkCudaErrors(cudaMemcpy(shuzuC,sc1->Q_Ctmp,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
                    for(int i=0;i<20;i++)
                    {
                        fprintf(fp2, "tmp2 %.20f %.20f\n",shuzuC[i].x,shuzuC[i].y);
                    }
                    fprintf(fp2,"---------------\n");

                 }
                order21_mde<<<(cplxDofs-1)/1024+1,1024>>>(Gsquare,sc1->Q_Ctmp,cplxDofs,ds,P);

            }   
            checkCudaErrors(cudaMemcpy(sc1->shuzu+t*cplxDofs,sc1->Q_Ctmp,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToDevice));
            if(ifprintf)
            {
            checkCudaErrors(cudaMemcpy(shuzuC,sc1->Q_Ctmp,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
            for(int i=0;i<20;i++)
            {
                fprintf(fp2,"tmp3 %.20f %.20f\n",shuzuC[i].x,shuzuC[i].y);
            }
            fprintf(fp2,"---------------\n");
            }

        }
      
        double Linear2index1=-1.0/3.0;
        double Linear2index2=4.0/3.0;
		
		// orderRich<<<(cplxDofs-1)/1024+1,1024>>>(cplxDofs,hatQ+(*index+i+1)*cplxDofs,sc1->shuzu,sc1->shuzu+cplxDofs);
        FuncsLinear2Cplx<<<(cplxDofs-1)/1024+1,1024>>>(hatQ+(*index+i+1)*cplxDofs,cplxDofs,Linear2index1,sc1->shuzu,Linear2index2,sc1->shuzu+cplxDofs);
		ifprintf=1;
         if(ifprintf)
         {
             fprintf(fp2,"%d\n",i);
             checkCudaErrors(cudaMemcpy(shuzuC,sc1->shuzu,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
             for(int ii=0;ii<20;ii++)
            {
                fprintf(fp2,"shuzu1 %.20f %.20f\n",shuzuC[ii].x,shuzuC[ii].y);
            }
            fprintf(fp2,"---------------\n");

            checkCudaErrors(cudaMemcpy(shuzuC,sc1->shuzu+cplxDofs,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
             for(int ii=0;ii<20;ii++)
            {
                fprintf(fp2,"shuzu2 %.20f %.20f\n",shuzuC[ii].x,shuzuC[ii].y);
            }
            fprintf(fp2,"---------------\n");

            checkCudaErrors(cudaMemcpy(shuzuC,hatQ+(*index+i+1)*cplxDofs,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
            for(int ii=0;ii<20;ii++)
            {
                fprintf(fp2,"final %.20f %.20f\n",shuzuC[ii].x,shuzuC[ii].y);
            }
            fprintf(fp2,"---------------\n");
         }
        ifprintf=0;
        

    }
    *index+=rangeN;
    fclose(fp2);
    free(shuzuC);
    free(shuzuCR);
   
   
}


// {
//     char fname2[]="2Order.txt";
//     FILE *fp2=fopen(fname2,"w+");  
//     if((fp2=fopen(fname2,"w+"))==NULL)
// 	{
//         printf("can not open this file\n");
// 	    exit(0);
//     }   

//     bool ifprintf=0;
//     double ds = (range[1] - range[0]) / rangeN;
//     double realDofs1=1.0/(double)sc1->realDofs;
//     cufftDoubleComplex *Q_Ctmp,*shuzu,*shuzuC;
//     double* doubleC;
//     cufftDoubleReal *Qreal,*shuzuCR,*WReal;


//     shuzuC=(cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*sc1->cplxDofs);
//     memset(shuzuC,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs);

//     shuzuCR=(cufftDoubleReal*)malloc(sizeof(cufftDoubleReal)*sc1->realDofs);
//     memset(shuzuCR,0,sizeof(cufftDoubleReal)*sc1->realDofs);

//     doubleC=(double*)malloc(sizeof(double)*sc1->cplxDofs);
//     memset(doubleC,0,sizeof(double)*sc1->cplxDofs);

//     checkCudaErrors(cudaMalloc((void**)&Qreal,sizeof(cufftDoubleReal)*sc1->realDofs));
// 	checkCudaErrors(cudaMemset(Qreal,0,sizeof(cufftDoubleReal)*sc1->realDofs));

//     checkCudaErrors(cudaMalloc((void**)&WReal,sizeof(cufftDoubleReal)*sc1->realDofs));
// 	checkCudaErrors(cudaMemset(WReal,0,sizeof(cufftDoubleReal)*sc1->realDofs));

//     checkCudaErrors(cudaMalloc((void**)&Q_Ctmp,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
// 	checkCudaErrors(cudaMemset(Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));

//     checkCudaErrors(cudaMalloc((void**)&shuzu,sizeof(cufftDoubleComplex)*2*sc1->cplxDofs));
// 	checkCudaErrors(cudaMemset(shuzu,0,sizeof(cufftDoubleComplex)*2*sc1->cplxDofs));
    
    
//     	//cufft的

//     int realDofs=sc1->realDofs;
//     int cplxDofs=sc1->cplxDofs;
//     ifprintf=0;
//     if(ifprintf)
//     {
//          fprintf(fp2, "realDofs %d\n",realDofs);
//     }
//     FftwC2R(fieldW,WReal,sc1);
//     ifprintf=1;
//     if(ifprintf)
//     {
//         checkCudaErrors(cudaMemcpy(shuzuC,fieldW,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
//         for(int i=0;i<20;i++)
//         {

//             fprintf(fp2, "fieldW %.20f %.20f\n",shuzuC[i].x,shuzuC[i].y);
//         }
//         fprintf(fp2,"---------------\n");
//         checkCudaErrors(cudaMemcpy(shuzuCR,WReal,sizeof(cufftDoubleReal)*realDofs,cudaMemcpyDeviceToHost));
//         for(int i=0;i<50;i++)
//         {
//         fprintf(fp2, "WReal %.20f\n",shuzuCR[i]);
//         }
//         fprintf(fp2,"-------------------\n");
//     }
//     // ifprintf=0;
//     // for(int i=0;i<rangeN;i++)
//     for(int i=0;i<1;i++)
//     {
//         checkCudaErrors(cudaMemset(shuzu,0,sizeof(cufftDoubleComplex)*2*cplxDofs));
//         // for(int t=0;t<2;t++)
//         for(int t=0;t<2;t++)
//         {
//             checkCudaErrors(cudaMemcpy(Q_Ctmp,hatQ+(*index+i)*cplxDofs,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToDevice));
//             double P=std::pow(2.0,t);
//             // for(int j=0;j<P;j++)
//             for(int j=0;j<P;j++)
//             {
//                 order21_mde<<<(cplxDofs-1)/1024+1,1024>>>(Gsquare,Q_Ctmp,cplxDofs,ds,P);

//                  if(ifprintf)
//                  {
//                     checkCudaErrors(cudaMemcpy(shuzuC,Q_Ctmp,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
//                     for(int i=0;i<20;i++)
//                     {

//                         fprintf(fp2, "tmp1 %.20f %.20f\n",shuzuC[i].x,shuzuC[i].y);
//                     }
//                     fprintf(fp2,"---------------\n");
//                  }

//                 FftwC2R(Q_Ctmp,Qreal,sc1);
//                  if(ifprintf)
//                  {
//                     checkCudaErrors(cudaMemcpy(shuzuCR,Qreal,sizeof(cufftDoubleReal)*realDofs,cudaMemcpyDeviceToHost));
//                     for(int i=0;i<20;i++)
//                     {
//                         fprintf(fp2,"Qreal1 %.20f\n",shuzuCR[i]);
//                     }
//                     fprintf(fp2,"---------------\n");
//                  }
                
//                 // order22_mde<<<(cplxDofs-1)/1024+1,1024>>>(WReal,Qreal,realDofs,ds,P);
//                 order22_mde<<<(realDofs-1)/1024+1,1024>>>(WReal,Qreal,realDofs,ds,P);
//                  if(ifprintf)
//                  {
//                     checkCudaErrors(cudaMemcpy(shuzuCR,Qreal,sizeof(cufftDoubleReal)*realDofs,cudaMemcpyDeviceToHost));
//                 for(int i=0;i<20;i++)
//                 {
//                     // fprintf(fp2,"Qreal2 %.20f\n",shuzuCR[realDofs-1-i]);
//                     fprintf(fp2,"Qreal2 %.20f\n",shuzuCR[i]);
//                 }
//                 fprintf(fp2,"---------------\n");
//                  }
                
//                 FftwR2C(Qreal,Q_Ctmp,sc1);
//                  if(ifprintf)
//                  {
//                     checkCudaErrors(cudaMemcpy(shuzuC,Q_Ctmp,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
//                     for(int i=0;i<20;i++)
//                     {
//                         fprintf(fp2, "tmp2 %.20f %.20f\n",shuzuC[i].x,shuzuC[i].y);
//                     }
//                     fprintf(fp2,"---------------\n");

//                  }


//                 order21_mde<<<(cplxDofs-1)/1024+1,1024>>>(Gsquare,Q_Ctmp,cplxDofs,ds,P);
//                  if(ifprintf)
//                  {
//                     checkCudaErrors(cudaMemcpy(shuzuC,Q_Ctmp,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
//                     for(int i=0;i<20;i++)
//                     {
//                         fprintf(fp2,"tmp3 %.20f %.20f\n",shuzuC[i].x,shuzuC[i].y);
//                     }
//                     fprintf(fp2,"---------------\n");
//                  }
//             }   
//             checkCudaErrors(cudaMemcpy(shuzu+t*cplxDofs,Q_Ctmp,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToDevice));
            
//         }
      
//         double Linear2index1=-1.0/3.0;
//         double Linear2index2=4.0/3.0;

//         FuncsLinear2Cplx<<<(cplxDofs-1)/1024+1,1024>>>(hatQ+(*index+i+1)*cplxDofs,cplxDofs,Linear2index1,shuzu,Linear2index2,shuzu+cplxDofs);
  
//         ifprintf=1;
//          if(ifprintf)
//          {
//              fprintf(fp2,"%d\n",i);
//              checkCudaErrors(cudaMemcpy(shuzuC,shuzu,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
//              for(int ii=0;ii<20;ii++)
//             {
//                 fprintf(fp2,"shuzu1 %.20f %.20f\n",shuzuC[ii].x,shuzuC[ii].y);
//             }
//             fprintf(fp2,"---------------\n");

//             checkCudaErrors(cudaMemcpy(shuzuC,shuzu+cplxDofs,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
//              for(int ii=0;ii<20;ii++)
//             {
//                 fprintf(fp2,"shuzu2 %.20f %.20f\n",shuzuC[ii].x,shuzuC[ii].y);
//             }
//             fprintf(fp2,"---------------\n");

//             checkCudaErrors(cudaMemcpy(shuzuC,hatQ+(*index+i+1)*cplxDofs,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToHost));
//             for(int ii=0;ii<20;ii++)
//             {
//                 fprintf(fp2,"final %.20f %.20f\n",shuzuC[ii].x,shuzuC[ii].y);
//             }
//             fprintf(fp2,"---------------\n");
//          }
//         // ifprintf=0;


//     }
//     *index+=rangeN;
// 	fclose(fp2);

//     // ifprintf=1;
//     // if(ifprintf)
// 	// {
// 	// fclose(fp2);
// 	// }
//     checkCudaErrors(cudaFree(Q_Ctmp));
   
//     checkCudaErrors(cudaFree(shuzu));

//     checkCudaErrors(cudaFree(Qreal));
//     checkCudaErrors(cudaFree(WReal));

//     free(shuzuC);
//     free(shuzuCR);
//     free(doubleC);
    
   
// }

void MDESolver2Order(double *range,double rangeN,scftData* sc1,cufftDoubleComplex *hatQ,cufftDoubleComplex* fieldW,int *index,
                    int* NCpt,double* Gsquare)
{
    double ds = (range[1] - range[0]) / rangeN;
    double realDofs1=1.0/(double)sc1->realDofs;
    
    // std::unique_ptr<double[]> doubleC = std::make_unique<double[]>(sc1->cplxDofs);
	// std::fill_n(doubleC.get(), sc1->cplxDofs, 0);
    checkCudaErrors(cudaMemset(sc1->Qreal,0,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->WReal,0,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
	checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*sc1->cplxDofs));
      
    	//cufft的

    int realDofs=sc1->realDofs;
    int cplxDofs=sc1->cplxDofs;
    
    FftwC2R(fieldW,sc1->WReal,sc1);
   
    // for(int i=0;i<rangeN;i++)
    for(int i=0;i<rangeN;i++)
    {
        checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*cplxDofs));
        // for(int t=0;t<2;t++)
        for(int t=0;t<2;t++)
        {
            checkCudaErrors(cudaMemcpy(sc1->Q_Ctmp,hatQ+(*index+i)*cplxDofs,sizeof(cufftDoubleComplex)*cplxDofs,cudaMemcpyDeviceToDevice));
            double P=std::pow(2.0,t);
            // for(int j=0;j<P;j++)
            for(int j=0;j<P;j++)
            {
                order21_mde<<<(cplxDofs-1)/1024+1,1024>>>(Gsquare,sc1->Q_Ctmp,cplxDofs,ds,P);

                FftwC2R(sc1->Q_Ctmp,sc1->Qreal,sc1);

                // order22_mde<<<(cplxDofs-1)/1024+1,1024>>>(sc1->WReal,sc1->Qreal,realDofs,ds,P);
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
    *index+=rangeN;
   
   
   
}


__global__ void getDirBox(double  *mat,int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < n*n )
    {
        int i = idx/n;                               
        int j = idx%n;
        if (i==j)
            mat[idx] = 8.5;
     }
}
__global__ void getRecipLattice(double* dBox,double* rBox, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < dim*dim )
    {
        int i = idx/dim;                               
        int j = idx%dim;
        if (i==j)
            rBox[idx] = 2*PI/dBox[idx];
       
     }
}
__global__ void setOnes(double *data,int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<n)
	{
		data[idx] = 1.0;
	}
}
__global__ void setSquare(double *data,int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<n)
	{
		data[idx] = data[idx]*data[idx];
	}
}
void getGsquare(scftData *sc1)
{
	double alpha=1.0,beta=0.0;   
	double *Gsquaretmp,*d_x;
	checkCudaErrors(cudaMalloc((void**)&Gsquaretmp,sizeof(double)*sc1->cplxDofs*sc1->DimCpt));
	checkCudaErrors(cudaMemset(Gsquaretmp,0,sizeof(double)*sc1->cplxDofs*sc1->DimCpt));
    checkCudaErrors(cudaMalloc((void**)&d_x,sizeof(double)*sc1->DimCpt));
	checkCudaErrors(cudaMemset(d_x,0,sizeof(double)*sc1->DimCpt));
	setOnes<<<1,sc1->DimCpt>>>(d_x,sc1->DimCpt);
	checkCudaErrors(cublasDgemm(sc1->handle,CUBLAS_OP_T,CUBLAS_OP_T,sc1->cplxDofs,sc1->DimCpt,sc1->DimCpt,&alpha,sc1->indKspaceGPU,sc1->DimCpt,sc1->rcpBoxGPU,sc1->DimCpt,&beta,Gsquaretmp,sc1->cplxDofs));
	setSquare<<<(sc1->cplxDofs*sc1->DimCpt-1)/1024+1,1024>>>(Gsquaretmp,sc1->cplxDofs*sc1->DimCpt);
	checkCudaErrors(cublasDgemv(sc1->handle,CUBLAS_OP_N,sc1->cplxDofs,sc1->DimCpt,&alpha,Gsquaretmp,sc1->cplxDofs,d_x,1,&beta,sc1->Gsquare,1));
	checkCudaErrors(cudaFree(d_x));
	checkCudaErrors(cudaFree(Gsquaretmp));
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
    MDESolver4Adams(sc1->Ranges[1],sc1->RangesN[1],sc1,var->bakQGpu,sc1->fieldWGPU1, &it_qplus,NCpt,sc1->Gsquare);// A
	
    MDESolver4Adams(sc1->Ranges[0],sc1->RangesN[0],sc1,var->bakQGpu,sc1->fieldWGPU0, &it_qplus,NCpt,sc1->Gsquare);// A

	free(frdQGpuInit);
	free(bakQGpuInit);
    // MDESolver4Adams(sc1->Ranges[0],sc1->RangesN[0], sc1->cplxDofs,sc1->realDofs,sc1->fieldWGPU0,frdQGpu, &it_q,NCpt,sc1->Gsquare);// A
    // MDESolver2Order(sc1->Ranges[1],sc1->RangesN[1], sc1->cplxDofs,sc1->realDofs,sc1->fieldWGPU1,frdQGpu, &it_q,NCpt,sc1->Gsquare);// A
	
}

double updataeQ(cufftDoubleComplex *frdQGpu,scftData *sc1)
{
	// cufftDoubleComplex *Qtt;
	// Qtt=(cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex )*1);
	checkCudaErrors(cudaDeviceSynchronize());	
    memset(sc1->cplxC1,0,sizeof(cufftDoubleComplex)*1);
	checkCudaErrors(cudaMemcpy(sc1->cplxC1,frdQGpu+(sc1->cplxDofs)*(sc1->Ns-1),sizeof(cufftDoubleComplex)*1,cudaMemcpyDeviceToHost));
	double tt=sc1->cplxC1[0].x;
	// printf("Q1=%.20f\n",tt);	
	// free(Qtt);
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
	// for(int i = 0; i < sc1->Nspecies; i++)

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
	// void updatePropagator(double **sc1->Ranges,double *sc1->RangesN,scftData *sc1,cufftDoubleComplex *sc1->fieldWGPU0,cufftDoubleComplex *sc1->fieldWGPU1,cufftDoubleComplex *frdQGpu,
	// 					cufftDoubleComplex *bakQGpu,cufftDoubleReal *WReal,cufftDoubleReal *WReal1,int *NCpt,double *sc1->Gsquare)
	updatePropagator(sc1,var,NCpt);
    double Q=updataeQ(var->frdQGpu,sc1);
	partialF -= std::log(Q);
	return partialF;
}
void get_gradB(scftData *sc1,scftVariable* var,int *NCpt,double* gradB1,double dh,FILE *fp0)
{
	double *oldBGPU;
	checkCudaErrors(cudaMalloc((void**)&oldBGPU,sizeof(double)*sc1->DimCpt*sc1->DimCpt));
	checkCudaErrors(cudaMemset(oldBGPU,0,sizeof(double)*sc1->DimCpt*sc1->DimCpt));
	checkCudaErrors(cudaMemcpy(oldBGPU,sc1->rcpBoxGPU,sizeof(double)*sc1->DimCpt*sc1->DimCpt,cudaMemcpyDeviceToDevice));
	dim3 block(sc1->DimCpt,sc1->DimCpt);
	dim3 grid(1,1);
	gradB_matrix<<<grid,block>>>(sc1->rcpBoxGPU,sc1->DimCpt,sc1->DimCpt,dh);
	checkCudaErrors(cudaDeviceSynchronize());
	double FR =calPartialF(sc1,var,NCpt);
	checkCudaErrors(cudaMemcpy(sc1->rcpBoxGPU,oldBGPU,sizeof(double)*sc1->DimCpt*sc1->DimCpt,cudaMemcpyDeviceToDevice));
	gradB_matrix<<<grid,block>>>(sc1->rcpBoxGPU,sc1->DimCpt,sc1->DimCpt,-dh);
	checkCudaErrors(cudaDeviceSynchronize());
	double FL =calPartialF(sc1,var,NCpt);
	checkCudaErrors(cudaMemcpy(sc1->rcpBoxGPU,oldBGPU,sizeof(double)*sc1->DimCpt*sc1->DimCpt,cudaMemcpyDeviceToDevice));
    fprintf(fp0,"[FR, FL] = [%.20f, \t %.20f]\n", FR, FL);
	memset(gradB1,0,sizeof(double)*sc1->DimCpt*sc1->DimCpt);
	for(int i = 0; i < sc1->DimCpt; i++)
	{
		gradB1[i*sc1->DimCpt+i] = (FR-FL)/(2.0*dh);
	}
	checkCudaErrors(cudaFree(oldBGPU));

}
void MatsAdd(double **dst, double d, double **src, double s, int n, int m)
{
	for(int i = 0; i < n; i++)
		for(int j = 0; j < m; j++)
			dst[i][j] = d*dst[i][j] + s*src[i][j];
}
void SimpleMixing(scftData *sc1,scftVariable* var,int *NCpt,double *resinftyW,double *resinftyB,FILE *fp0)
{
	double *resGradW, *resGradB;
	resGradW  = (double *)malloc(sizeof(double)*sc1->Nspecies);
	resGradB = (double *)malloc(sizeof(double)*sc1->DimCpt);

	for(int i = 0; i < sc1->Nspecies; i++) 
	{
		resGradW[i] = 0.0;
	}
	for(int i = 0; i < sc1->DimCpt; i++) 
		resGradB[i] = 0.0;
	double *gradB,*gradBGPU; 
	cufftDoubleComplex  *gradWGpu,*fieldWplusC;
	cufftDoubleComplex *gradW;
	gradB= (double *)malloc(sizeof(double*)*sc1->DimCpt*sc1->DimCpt);
	memset(gradB,0,sizeof(double)*sc1->DimCpt*sc1->DimCpt);
	gradW = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex)*sc1->Nspecies);
	memset(gradW,0,sizeof(cufftDoubleComplex)*sc1->Nspecies);
	fieldWplusC = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex)*1);
	memset(fieldWplusC,0,sizeof(cufftDoubleComplex*)*1);
	checkCudaErrors(cudaMalloc((void**)&gradBGPU,sizeof(cufftDoubleComplex)*sc1->DimCpt*sc1->DimCpt));
	checkCudaErrors(cudaMemset(gradBGPU,0,sizeof(cufftDoubleComplex) * sc1->DimCpt*sc1->DimCpt));
	checkCudaErrors(cudaMalloc((void**)&gradWGpu,sizeof(cufftDoubleComplex) * (sc1->Nspecies)*(sc1->cplxDofs)));
	checkCudaErrors(cudaMemset(gradWGpu,0,sizeof(cufftDoubleComplex) * (sc1->Nspecies)*(sc1->cplxDofs)));

	double dt1,dt2,dt3;
	cufftDoubleComplex cublasZaxpyAlpha1,cublasZaxpyAlpha2;
	 dt1 = 0.1;
	 dt2 = 0.1;
	 dt3 = 1.0e-06;
	cublasZaxpyAlpha1.x= dt1;
	cublasZaxpyAlpha1.y=0.0;
	cublasZaxpyAlpha2.x= dt2;
	cublasZaxpyAlpha2.y=0.0;
	 FuncsLinear3Cplx<<<(sc1->cplxDofs-1)/1024+1,1024>>>(gradWGpu                  ,sc1->cplxDofs,sc1->chi*sc1->Ndeg,var->rhoGpu+(sc1->cplxDofs)*1,1.0,sc1->fieldWplus, -1.0,sc1->fieldWGPU0);
	 FuncsLinear3Cplx<<<(sc1->cplxDofs-1)/1024+1,1024>>>(gradWGpu+(sc1->cplxDofs)*1,sc1->cplxDofs,sc1->chi*sc1->Ndeg,var->rhoGpu+(sc1->cplxDofs)*0,1.0,sc1->fieldWplus, -1.0,sc1->fieldWGPU1);
	 checkCudaErrors(cublasZaxpy(sc1->handle,sc1->cplxDofs,&cublasZaxpyAlpha1,gradWGpu                  ,1.0,sc1->fieldWGPU0,1.0));
	 checkCudaErrors(cublasZaxpy(sc1->handle,sc1->cplxDofs,&cublasZaxpyAlpha2,gradWGpu+(sc1->cplxDofs)*1,1.0,sc1->fieldWGPU1,1.0));
	//  //update w_plus
	FuncsLinear2Cplx<<<(sc1->cplxDofs-1)/1024+1,1024>>>(sc1->fieldWplus,sc1->cplxDofs,0.5,sc1->fieldWGPU0,0.5,sc1->fieldWGPU1);
	checkCudaErrors(cudaMemcpy(fieldWplusC,sc1->fieldWplus,sizeof(cufftDoubleComplex)*1,cudaMemcpyDeviceToHost));
	fieldWplusC[0].x -= 0.5*sc1->chi*sc1->Ndeg;
	checkCudaErrors(cudaMemcpy(sc1->fieldWplus,fieldWplusC,sizeof(cufftDoubleComplex)*1,cudaMemcpyHostToDevice));
	int maxgradWIndex1,maxgradWIndex2;
	checkCudaErrors(cublasIzamax(sc1->handle,sc1->cplxDofs,gradWGpu                  ,1,&maxgradWIndex1));
	checkCudaErrors(cublasIzamax(sc1->handle,sc1->cplxDofs,gradWGpu+(sc1->cplxDofs)*1,1,&maxgradWIndex2));
	
	checkCudaErrors(cudaMemcpy(gradW,gradWGpu+maxgradWIndex1-1,sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gradW+1,gradWGpu+(sc1->cplxDofs)*1+maxgradWIndex2-1,sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost));

	resGradW[0]=gradW[0].x*gradW[0].x+gradW[0].y*gradW[0].y;
	resGradW[1]=gradW[1].x*gradW[1].x+gradW[1].y*gradW[1].y;
	resGradW[0]=sqrt(resGradW[0]);
	resGradW[1]=sqrt(resGradW[1]);
	fprintf(fp0,"[res1, res2]  %.20f %.20f\n",resGradW[0],resGradW[1]);
	checkCudaErrors(cudaFree(gradWGpu));


	// //update domain
	double dh = 1e-05;
	int matrixSize = sc1->DimCpt*sc1->DimCpt;
	get_gradB(sc1,var,NCpt,gradB,dh,fp0);
	
	cudaMemcpy(gradBGPU,gradB,sizeof(double)*sc1->DimCpt*sc1->DimCpt,cudaMemcpyHostToDevice);    


	MatsAdd<<<1,matrixSize>>>(sc1->rcpBoxGPU,1.0,gradBGPU,dt3,matrixSize);
	fprintf(fp0,"\n===== Direct Box ============= \n");
	
	getRecipLattice<<<1,matrixSize>>>(sc1->rcpBoxGPU, sc1->dirBoxGPU,sc1->DimCpt);
	MatPrint(sc1->dirBoxGPU, sc1->DimCpt,fp0);
	// printf("\n");
    double** gradBB;
    	gradBB = (double **)malloc(sizeof(double*)*sc1->DimCpt);
    for(int i = 0; i < sc1->DimCpt; i++)
	{
		gradBB[i] = (double *)malloc(sizeof(double)*sc1->DimCpt);
		memset(gradBB[i],0,sizeof(double)*sc1->DimCpt);

	}
    for(int i=0;i<sc1->DimCpt;i++)
    {
    for(int j=0;j<sc1->DimCpt;j++)
    {
        gradBB[i][j]=gradB[i*sc1->DimCpt+j];
    }
    }
	for(int i=0;i<sc1->DimCpt;i++)
	{
		resGradB[i]=normRealInfty(gradBB[i], sc1->DimCpt);
	}
	fprintf(fp0,"[resB1, resB2, resB3] = [%.20e, \t %.20e, \t %.20e]\n", resGradB[0],resGradB[1],resGradB[2]);
	resinftyW[0]= normRealInfty(resGradW, sc1->Nspecies);
	resinftyB[0]= normRealInfty(resGradB, sc1->DimCpt);
	// fprintf(fp0,"resinftyW=%.20f resinftyB=%.20f\n",resinftyW[0],resinftyB[0]);

	free(resGradW);
	free(resGradB);
	free(gradB);
	checkCudaErrors(cudaFree(gradBGPU));
	free(fieldWplusC);
	// 释放指向行的指针数组
	free(gradW);
    for (int i = 0; i <sc1->DimCpt; i++) 
	{
    free(gradBB[i]); // 或者 cudaFree(gradW[i])，取决于你如何分配的内存
	}
	// 释放指向行的指针数组
	free(gradBB);

}

void updateField(scftData *sc1,scftVariable* var,int *NCpt,double *resinftyW,double *resinftyB,FILE *fp0)
{
	SimpleMixing(sc1,var,NCpt,resinftyW,resinftyB,fp0);

}
double updateHamilton(scftData *sc1,scftVariable* var,FILE *fp0)
{
	// cufftDoubleComplex *rhoGpu,double* singQ,double*internalEnergy,double*entropicEnergy,
	double tmpAB,tmpA, tmpB, tmpWplus,internalEnergy1,entropicEnergy1;
    //rhotmp 用sc1->Q_Ctmp做替代
	checkCudaErrors(cudaMemset(sc1->Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
	tmpAB = Intergral_space(sc1,var->rhoGpu+(sc1->cplxDofs)*0,var->rhoGpu+(sc1->cplxDofs)*1);

	tmpA = Intergral_space(sc1,sc1->fieldWGPU0,var->rhoGpu+(sc1->cplxDofs)*0);
	// printf("tmpA = %.20e\n", tmpAB);

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




