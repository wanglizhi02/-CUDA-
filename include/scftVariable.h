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