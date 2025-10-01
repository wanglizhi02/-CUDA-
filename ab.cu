#include <Head.h>
#include <Mytimer.h>
#include "Initialization.h"
#include "FftwToolkit.h"
#include "BasicFunc.h"
#include "SCFTBaseAB.h"
void scftDataMemAlloc(scftData *sc1,int *NCpt)
{
	checkCudaErrors(cublasCreate(&sc1->handle));
    checkCudaErrors(cufftPlan3d(&sc1->plan1,NCpt[0],NCpt[1],NCpt[2],CUFFT_D2Z));
    checkCudaErrors(cufftPlan3d(&sc1->plan,NCpt[0],NCpt[1],NCpt[2],CUFFT_Z2D));


	checkCudaErrors(cudaMalloc((void**)&sc1->Gsquare,sizeof(double)*sc1->cplxDofs));
	checkCudaErrors(cudaMemset(sc1->Gsquare,0,sizeof(double)*sc1->cplxDofs));

	checkCudaErrors(cudaMalloc((void**)&sc1->dirBoxGPU,sizeof(double)*sc1->DimCpt*sc1->DimCpt));
	checkCudaErrors(cudaMemset(sc1->dirBoxGPU,0,sizeof(double)*sc1->DimCpt*sc1->DimCpt));

	checkCudaErrors(cudaMalloc((void**)&sc1->rcpBoxGPU,sizeof(double)*sc1->DimCpt*sc1->DimCpt));
	checkCudaErrors(cudaMemset(sc1->rcpBoxGPU,0,sizeof(double)*sc1->DimCpt*sc1->DimCpt));

	checkCudaErrors(cudaMalloc((void**)&sc1->indKspaceGPU,sizeof(double)*sc1->cplxDofs*sc1->DimCpt));
	checkCudaErrors(cudaMemset(sc1->indKspaceGPU,0,sizeof(double)*sc1->cplxDofs*sc1->DimCpt));

	checkCudaErrors(cudaMalloc((void**)&sc1->fieldWGPU0,sizeof(cufftDoubleComplex) * sc1->cplxDofs));
	checkCudaErrors(cudaMemset(sc1->fieldWGPU0,0,sizeof(cufftDoubleComplex) * sc1->cplxDofs));

	checkCudaErrors(cudaMalloc((void**)&sc1->fieldWGPU1,sizeof(cufftDoubleComplex) * sc1->cplxDofs));
	checkCudaErrors(cudaMemset(sc1->fieldWGPU1,0,sizeof(cufftDoubleComplex) * sc1->cplxDofs));

	checkCudaErrors(cudaMalloc((void**)&sc1->fieldWplus,sizeof(cufftDoubleComplex) * sc1->cplxDofs));
	checkCudaErrors(cudaMemset(sc1->fieldWplus,0,sizeof(cufftDoubleComplex) * sc1->cplxDofs));

	sc1->RangesN = (double *)malloc(sizeof(double)*sc1->Nblock);
	memset(sc1->RangesN,0,sizeof(double)*sc1->Nblock);

	sc1->Ranges = (double **)malloc(sizeof(double*)*sc1->Nblock);
    for(int i = 0; i < sc1->Nblock; i++)
    {
        sc1->Ranges[i] = (double *)malloc(sizeof(double)*2);
		memset(sc1->Ranges[i],0,sizeof(double)*2);
    }

	sc1->fieldReal = (double *)malloc(sizeof(double)*sc1->realDofs);
	memset(sc1->fieldReal,0,sizeof(double)*sc1->realDofs);

	 
	sc1->cplxC1=(cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*1);
    memset(sc1->cplxC1,0,sizeof(cufftDoubleComplex)*1);
	
    checkCudaErrors(cudaMalloc((void**)&sc1->realGpu2,sizeof(cufftDoubleReal)*sc1->realDofs));
    checkCudaErrors(cudaMemset(sc1->realGpu2,0,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMalloc((void**)&sc1->realGpu3,sizeof(cufftDoubleReal)*sc1->realDofs));
    checkCudaErrors(cudaMemset(sc1->realGpu3,0,sizeof(cufftDoubleReal)*sc1->realDofs));

    checkCudaErrors(cudaMalloc((void**)&sc1->cplxGpu1,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
    checkCudaErrors(cudaMemset(sc1->cplxGpu1,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
	checkCudaErrors(cudaMalloc((void**)&sc1->cplxGpu2,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
    checkCudaErrors(cudaMemset(sc1->cplxGpu2,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));


	checkCudaErrors(cudaMalloc((void**)&sc1->Qreal,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->Qreal,0,sizeof(cufftDoubleReal)*sc1->realDofs));

    checkCudaErrors(cudaMalloc((void**)&sc1->WReal,sizeof(cufftDoubleReal)*sc1->realDofs));
	checkCudaErrors(cudaMemset(sc1->WReal,0,sizeof(cufftDoubleReal)*sc1->realDofs));

    checkCudaErrors(cudaMalloc((void**)&sc1->Q_Ctmp,sizeof(cufftDoubleComplex)*sc1->cplxDofs));
	checkCudaErrors(cudaMemset(sc1->Q_Ctmp,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs));

    checkCudaErrors(cudaMalloc((void**)&sc1->shuzu,sizeof(cufftDoubleComplex)*2*sc1->cplxDofs));
	checkCudaErrors(cudaMemset(sc1->shuzu,0,sizeof(cufftDoubleComplex)*2*sc1->cplxDofs));




}
void scftDataMemAllocRelease(scftData *sc1)
{
	checkCudaErrors(cublasDestroy(sc1->handle));
	checkCudaErrors(cufftDestroy(sc1->plan));
	checkCudaErrors(cufftDestroy(sc1->plan1));

	checkCudaErrors(cudaFree(sc1->Gsquare));
	checkCudaErrors(cudaFree(sc1->dirBoxGPU));
	checkCudaErrors(cudaFree(sc1->rcpBoxGPU));
	checkCudaErrors(cudaFree(sc1->indKspaceGPU));
	checkCudaErrors(cudaFree(sc1->fieldWGPU0));
	checkCudaErrors(cudaFree(sc1->fieldWGPU1));
	checkCudaErrors(cudaFree(sc1->fieldWplus));

	for (int i = 0; i < sc1->Nblock; i++) 
	{
        free(sc1->Ranges[i]); 
    }
	free(sc1->Ranges);
	free(sc1->RangesN);

	free(sc1->fieldReal);
	free(sc1->cplxC1);
	
	checkCudaErrors(cudaFree(sc1->realGpu2));
	checkCudaErrors(cudaFree(sc1->realGpu3));
	checkCudaErrors(cudaFree(sc1->cplxGpu1));
	checkCudaErrors(cudaFree(sc1->cplxGpu2));

	 checkCudaErrors(cudaFree(sc1->Q_Ctmp));
    checkCudaErrors(cudaFree(sc1->shuzu));

    checkCudaErrors(cudaFree(sc1->Qreal));
    checkCudaErrors(cudaFree(sc1->WReal));

}

void scftDataInitValue(scftData *sc1,int *NCpt)
{
	sc1->phase=1;
    sc1->fA=0.4;
	sc1->fB=1-sc1->fA;
    sc1->chi=0.14;
    sc1->dN=200;
    sc1->ItMax=10000;
    sc1->Ndeg=100;
    sc1->dsMax=1.0/sc1->dN;
	sc1->Nspecies=2;
	sc1->Nblock=2;
	sc1->Nblend=1;
	sc1->DimPhy=3;
// 总共的自由度
	sc1->realDofs = 1;
	sc1->cplxDofs = 1;
	for(int i = 0; i < sc1->DimCpt-1; i++)
	{
		sc1->realDofs *= NCpt[i];
		sc1->cplxDofs *= NCpt[i];
	}
	sc1->realDofs *= NCpt[sc1->DimCpt-1];
	sc1->cplxDofs *= (NCpt[sc1->DimCpt-1]/2+1);

	scftDataMemAlloc(sc1,NCpt);

    std::unique_ptr<int[]> Nsplit= std::make_unique<int[]>(sc1->Nblock);

	// int *Nsplit;
	// Nsplit = (int *)malloc(sizeof(int)*sc1->Nblock);
	for(int i = 0; i < sc1->Nblock; i++) Nsplit[i] = 0;
    Nsplit[0] = round((sc1->fA+1.0e-8) / sc1->dsMax);
    Nsplit[1] = round((sc1->fB+1.0e-8) / sc1->dsMax);
	sc1->Ranges[0][0] = 0.0;
	sc1->Ranges[0][1] = sc1->fA;
	sc1->Ranges[1][0] = sc1->Ranges[0][1];
	sc1->Ranges[1][1] = sc1->Ranges[0][1]+sc1->fB;

	sc1->Ns=0;
	for(int i = 0; i < sc1->Nblock; i++) 
		sc1->Ns += Nsplit[i];
	sc1->Ns = sc1->Ns+1;

	for(int i = 0; i < sc1->Nblock; i++)
	{
		sc1->RangesN[i] = (double)Nsplit[i];
		// printf("sc1->RangesN[%d]=%.20f\n",i,sc1->RangesN[i]);
	}
	printf("\t sc1->Ns:%d\n",sc1->Ns);
	printf("\t Discrete points in s:[Nsplit:A-B]=[%d-%d]\n", Nsplit[0],Nsplit[1]);
}

void scftVariableMemAlloc(scftVariable *var,scftData *sc1)
{

    var->singQ = (double *)malloc(sizeof(double)*sc1->Nblend);
	memset(var->singQ,0,sizeof(double)*sc1->Nblend);
    
  	var->dirBox1 = (double *)malloc(sizeof(double)*sc1->DimCpt*sc1->DimCpt);
	memset(var->dirBox1,0,sizeof(double)*sc1->DimCpt*sc1->DimCpt);

	var->internalEnergy = (double *)malloc(sizeof(double)*1);
	memset(var->internalEnergy,0,sizeof(double)*1);

    var->entropicEnergy = (double *)malloc(sizeof(double)*1);
	memset(var->entropicEnergy,0,sizeof(double)*1);

	var->rhoreal = (double **)malloc(sizeof(double*)*sc1->Nspecies);
	for(int i = 0; i < sc1->Nspecies; i++)
	{
        var->rhoreal[i] = (double *)malloc(sizeof(double)*(sc1->realDofs));
		memset(var->rhoreal[i],0,sizeof(double)*(sc1->realDofs));
    }

	
	checkCudaErrors(cudaMalloc((void**)&var->rhoGpu,sizeof(cufftDoubleComplex) * (sc1->Nspecies)*(sc1->cplxDofs)));
	checkCudaErrors(cudaMemset(var->rhoGpu,0,sizeof(cufftDoubleComplex) * (sc1->Nspecies)*(sc1->cplxDofs)));
    checkCudaErrors(cudaMalloc((void**)&var->frdQGpu,sizeof(cufftDoubleComplex)*sc1->cplxDofs*sc1->Ns));
	checkCudaErrors(cudaMemset(var->frdQGpu,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs*sc1->Ns));
	checkCudaErrors(cudaMalloc((void**)&var->bakQGpu,sizeof(cufftDoubleComplex)*sc1->cplxDofs*sc1->Ns));
	checkCudaErrors(cudaMemset(var->bakQGpu,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs*sc1->Ns));
	checkCudaErrors(cudaMalloc((void**)&var->hatq_qplusGpu,sizeof(cufftDoubleComplex)*sc1->cplxDofs*sc1->Ns));
	checkCudaErrors(cudaMemset(var->hatq_qplusGpu,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs*sc1->Ns));
    checkCudaErrors(cudaMalloc((void**)&var->rhorealGPU,sizeof(cufftDoubleReal)*sc1->Nspecies*sc1->realDofs));
	checkCudaErrors(cudaMemset(var->rhorealGPU,0,sizeof(cufftDoubleReal)*sc1->Nspecies*sc1->realDofs));

}
void scftVariableMemAllocRelease(scftVariable *var,scftData *sc1)
{
    
	free(var->singQ);
	free(var->dirBox1);
    free(var->internalEnergy);
	free(var->entropicEnergy);
	for (int i = 0; i < sc1->Nspecies; i++) 
	{
        free(var->rhoreal[i]); 		
    }
	free(var->rhoreal);
	checkCudaErrors(cudaFree(var->rhoGpu));
	checkCudaErrors(cudaFree(var->frdQGpu));
	checkCudaErrors(cudaFree(var->bakQGpu));
	checkCudaErrors(cudaFree(var->hatq_qplusGpu));
	checkCudaErrors(cudaFree(var->rhorealGPU));

}
void getIndex(int **kspace, int n, int dim, int *ndeg)
{
	// printf("getIndex\n");
	int *k = (int *)malloc(sizeof(int)*dim);
	for(int i = 0; i < dim; i++) k[i] = 0;	

	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < dim-1; j++)
		{
			if(k[j] > ndeg[j]/2)
				kspace[i][j] = k[j] - ndeg[j];
			else
				kspace[i][j] = k[j];
		}
		kspace[i][dim-1] = k[dim-1];

		k[dim-1] ++;
		if(k[dim-1] > ndeg[dim-1]/2)
		{
			k[dim-1] = 0;
			k[dim-2] ++;
		}
		for(int jj = dim-2; jj > 0; jj--)
		{
			if(k[jj] > ndeg[jj]-1)
			{
				k[jj] = 0;
				k[jj-1] ++;
			}
		}
	}
	free(k);
}

int main()
{
    mytimer_t timer0,timer1;//timer0 算总时间 timer1 算gpu时间
	timer0.reset();
	timer0.start();//计算总时间的
    double TOL;
	TOL=1.0e-8;
	char fname1[]="/home/kaijiang/wlz/AB3Dr/out/out.txt";
 	FILE *fp0=fopen(fname1,"w+");  
    if((fp0=fopen(fname1,"w+"))==NULL)
	{
        printf("can not open the out file\n");
	    exit(0);
    }

	mytimer_t timer;
	timer.reset();
	timer.start();
    struct scftData *sc1;
	sc1= (scftData *)malloc(sizeof(scftData)*1);
	sc1->DimCpt = 3; //cplx space;
	int* NCpt = (int *)malloc(sizeof(int)*sc1->DimCpt);
	int pp=64;
	for(int i = 0; i < sc1->DimCpt; i++)
		NCpt[i] = pp;	
	// NCpt[0]=510;
	// NCpt[1]=76;
	// NCpt[2]=76;
	TOL=1.0e-8;
    scftDataInitValue(sc1,NCpt);
	struct scftVariable *var;
	var= (scftVariable *)malloc(sizeof(scftVariable)*1);
    // auto var = new scftVariable();
    scftVariableMemAlloc(var,sc1);
	double *indKspace1;
	int **indKspace;
	indKspace1=(double*)malloc(sizeof(double)*sc1->cplxDofs*sc1->DimCpt);
	memset(indKspace1,0,sizeof(double)*sc1->cplxDofs*sc1->DimCpt);
	 indKspace = (int **)malloc(sizeof(int*)*sc1->cplxDofs);
	for(int i = 0; i < sc1->cplxDofs; i++)
	{
		indKspace[i] = (int *)malloc(sizeof(int)*sc1->DimCpt);
		memset(indKspace[i],0,sizeof(int)*sc1->DimCpt);
	}
	timer.pause();
	fprintf(fp0, "\t\t time cost of memory allocation : %f seconds\n", timer.get_current_time());
	
	double *doubleC = (double *)malloc(sizeof(double)*sc1->cplxDofs);
	memset(doubleC,0,sizeof(double)*sc1->cplxDofs);
	cufftDoubleComplex *cufftDoubleComplexC = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex)*sc1->cplxDofs);
	memset(cufftDoubleComplexC,0,sizeof(cufftDoubleComplex)*sc1->cplxDofs);

	timer.reset();
	timer.start();
	getIndex(indKspace, sc1->cplxDofs, sc1->DimCpt, NCpt);
	for(int i=0;i<sc1->cplxDofs;i++)
	{
		for(int j=0;j<sc1->DimCpt;j++)
		{
			indKspace1[i*sc1->DimCpt+j]=(double)indKspace[i][j];
		}
	}
	checkCudaErrors(cudaMemcpy(sc1->indKspaceGPU,indKspace1,sizeof(double)*sc1->cplxDofs*sc1->DimCpt,cudaMemcpyHostToDevice));
	free(indKspace1);
	for (int i = 0; i < sc1->cplxDofs; i++) 
	{
		free(indKspace[i]); 
    }
    free(indKspace);

	timer.pause();
	fprintf(fp0, "\t\t time cost of getIndex : %f seconds\n\n", timer.get_current_time());
	
	// var->dirBox1[0*sc1->DimCpt+0]=170;
	// var->dirBox1[1*sc1->DimCpt+1]=25.5;
	// var->dirBox1[2*sc1->DimCpt+2]=25.5;
	double dd=8.5;
	var->dirBox1[0*sc1->DimCpt+0]=dd;
	var->dirBox1[1*sc1->DimCpt+1]=dd;
	var->dirBox1[2*sc1->DimCpt+2]=dd;
	checkCudaErrors(cudaMemcpy(sc1->dirBoxGPU,var->dirBox1,sizeof(double)*sc1->DimCpt*sc1->DimCpt,cudaMemcpyHostToDevice));
	getRecipLattice<<<1,sc1->DimCpt*sc1->DimCpt>>>(sc1->dirBoxGPU, sc1->rcpBoxGPU,sc1->DimCpt);
	
	timer.reset();
	timer.start();
	//init field
	char fnameread[150];
	// sprintf(fnameread, "/home/kaijiang/wlz/initData/rhoA_twist2_bulk1_170.00_25.50_25.50.txt");
	// sprintf(fnameread, "/home/kaijiang/wlz/initData/rho_DG_128_128_128.txt");
	sprintf(fnameread, "/home/kaijiang/wlz/initData/DG/DGA_64.txt");
	writeRealData(sc1,sc1->fieldWGPU0,fnameread);
	cufftDoubleComplex cublasZaxpyfield;
	cublasZaxpyfield.x=-1.0;
	cublasZaxpyfield.y=0.0;
	checkCudaErrors(cublasZaxpy(sc1->handle,sc1->cplxDofs,&cublasZaxpyfield,sc1->fieldWGPU0,1.0,sc1->fieldWGPU1,1.0));
	// FuncsLinear1Cplx<<<(sc1->cplxDofs-1)/1024+1,1024>>>(sc1->fieldWGPU1, sc1->cplxDofs, -1.0, sc1->fieldWGPU0);
	timer.pause();
	fprintf(fp0, "\t\t time cost of initRho : %f seconds\n\n", timer.get_current_time());

	fprintf(fp0, "\t ------- Discrete Modes ------- \n");
	for(int i = 0; i < sc1->DimCpt; i++)
		fprintf(fp0, "\t NCpt[%d] = %d", i, NCpt[i]);
	fprintf(fp0, "\n");

	fprintf(fp0, "\n\n\t********************************* PARAMETERS ************************************* \n");
	fprintf(fp0, "\t*\t\t   Nspecies = %d, \t Nblend = %d, \t Nblock = %d              *\n", sc1->Nspecies, sc1->Nblend, sc1->Nblock);
	fprintf(fp0, "\t*\t\t[chiAB-fA-fB] = [%f-%f-%f]      *\n", sc1->chi, sc1->fA, sc1->fB);
	fprintf(fp0, " \t cplxDofs = %d,\t realDofs = %d, \t phase = %d\n", sc1->cplxDofs, sc1->realDofs, sc1->phase);

	double error,Energy, oldEnergy, diffEnergy;
	double resinftyW[1];
	double resinftyB[1];
	oldEnergy = 100.0;
	Energy = 100.0;
	int iterator = 0;
// // 	// std::unique_ptr<double[]> doubleC = std::make_unique<double[]>(sc1->cplxDofs);
// // 	// std::fill_n(doubleC.get(), sc1->cplxDofs, 0);
	timer1.reset();
	timer1.start();
	cudaEvent_t start, stop;
	float time;




do{
	iterator++;
	fprintf(fp0,"iterator=%d\n",iterator);
	getGsquare(sc1);

	timer.reset();
	timer.start();
	checkCudaErrors(cudaDeviceSynchronize());	
	updatePropagator(sc1,var,NCpt);
	checkCudaErrors(cudaDeviceSynchronize());	
	timer.pause();

	fprintf(fp0, "\t\t time cost of updatePropagator : %f seconds\n", timer.get_current_time());



	timer.reset();
	timer.start();
	checkCudaErrors(cudaDeviceSynchronize());	
	var->singQ[0]=updataeQ(var->frdQGpu,sc1);
	fprintf(fp0, "Q=%.20f\n",var->singQ[0]);	
	// double bakQ0=updataeQ(var->bakQGpu,sc1);
	// fprintf(fp0, "backQ=%.20f\n",bakQ0);	
	checkCudaErrors(cudaDeviceSynchronize());	
	timer.pause();
    fprintf(fp0, "\t\t time cost of updateQ          : %f seconds\n", timer.get_current_time());
	timer.reset();
	timer.start();
	checkCudaErrors(cudaDeviceSynchronize());	
    updateOrderParameter(sc1,var);
	checkCudaErrors(cudaDeviceSynchronize());	
	timer.pause();
    fprintf(fp0, "\t\t time cost of updateOrderParameter    : %f seconds\n", timer.get_current_time());
	

	
	timer.reset();
	timer.start();
	checkCudaErrors(cudaDeviceSynchronize());	
	updateField(sc1,var,NCpt,resinftyW,resinftyB,fp0);
	checkCudaErrors(cudaDeviceSynchronize());	
	timer.pause();
    fprintf(fp0, "\t\t time cost of updateField      : %f seconds\n", timer.get_current_time());

	 //updateHamilton
	timer.reset();
	timer.start();
	checkCudaErrors(cudaDeviceSynchronize());	
	Energy=updateHamilton(sc1,var,fp0);
	checkCudaErrors(cudaDeviceSynchronize());	
	timer.pause();
    fprintf(fp0, "\t\t time cost of updateHamilton   : %f seconds\n", timer.get_current_time());
	
	for(int i=0;i< sc1->Nspecies; i++)
	{
		FftwC2R(var->rhoGpu+(sc1->cplxDofs)*i, var->rhorealGPU+(sc1->realDofs)*i,sc1);
		checkCudaErrors(cudaMemcpy(var->rhoreal[i],var->rhorealGPU+i*(sc1->realDofs),sizeof(cufftDoubleReal)*sc1->realDofs,cudaMemcpyDeviceToHost));
	}

	diffEnergy = fabs(Energy-oldEnergy);
	fprintf(fp0, "diffEnergy %f\n",diffEnergy);
	
	write_rho(var->rhoreal, iterator,sc1);//	


	cudaMemcpy(var->dirBox1,sc1->dirBoxGPU,sizeof(double)*sc1->DimCpt*sc1->DimCpt,cudaMemcpyDeviceToHost);

    writeRst1(sc1,var->dirBox1,diffEnergy,resinftyW[0],resinftyB[0],var->singQ[0],Energy,var->internalEnergy[0],var->entropicEnergy[0]);


	fprintf(fp0, "================================================\n");
	fprintf(fp0, "ITERATOR %d:  singQ = %.20e\t, Energy=%.20e\t,  diffhm =%.20e, resinftyW[0] = %.20e,  resinftyB[0]=%.20e\n", iterator, var->singQ[0],  Energy, diffEnergy, resinftyW[0],  resinftyB[0]);	
    fprintf(fp0,  "dirBox:%.5f\t %.5f\t %.5f\t\n internalEnergy[0]  %.20e\t entropicEnergy[0] %.20e\n",var->dirBox1[0*sc1->DimCpt+0], var->dirBox1[1*sc1->DimCpt+1],var->dirBox1[2*sc1->DimCpt+2], var->internalEnergy[0], var->entropicEnergy[0]);
	fprintf(fp0, "================================================\n");

    oldEnergy = Energy;
		// if (iterator > sc1->ItMax)
		if (iterator >sc1->ItMax)
			break;

		if(diffEnergy > 10000)
			break;
	}while(diffEnergy> TOL);
    writeData(sc1,NCpt,iterator,var->dirBox1,diffEnergy,resinftyW[0],resinftyB[0],var->singQ[0],Energy,var->internalEnergy[0],var->entropicEnergy[0]);
	writeEnergy(sc1,diffEnergy,var->internalEnergy[0],var->entropicEnergy[0],Energy);


	free(doubleC);
	free(cufftDoubleComplexC);
	free(NCpt);
    scftVariableMemAllocRelease(var,sc1);
    scftDataMemAllocRelease(sc1);
    free(sc1);
    free(var);
	timer0.pause();
	timer1.pause();
	fprintf(fp0, "\n=== GPU Time cost: %f seconds, \t %f minutes, \t %f hours\n\n", timer1.get_current_time(), double(timer1.get_current_time()/60.0), double(timer1.get_current_time()/3600.0));
	fprintf(fp0, "\n=== Time cost: %f seconds, \t %f minutes, \t %f hours\n\n", timer0.get_current_time(), double(timer0.get_current_time()/60.0), double(timer0.get_current_time()/3600.0));
	fprintf(fp0, "\n\n ======================================   END PROGRAM  ======================================\n\n");
	fclose(fp0);

	return 0;

}