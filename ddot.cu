#include "ddot.hpp"
#include <stdio.h>

__global__ void dot(double * x, double * y, double * z, int n) {
	int index = threadIdx.x + blockIdx.x *blockDim.x;
	if (index < n) {
		z[index] = x[index] * y[index];
	}
	
	__syncthreads();
}

int ddot (const int n, const double * const x, const double * const y, 
	  double * const result, double & time_allreduce)
{
	double * z = (double*)malloc(n * sizeof(double));
	int threadsPerBlock = 512;
	int blocksPerGrid = (n + threadsPerBlock - 1)/threadsPerBlock;
	double * x_d = NULL;
	cudaMalloc((void**)&x_d, n*sizeof(double));
	double * y_d = NULL;
	cudaMalloc((void**)&y_d, n*sizeof(double));
	double * z_d = NULL;
	cudaMalloc((void**)&z_d, n*sizeof(double));
	cudaMemcpy(x_d, x, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, n*sizeof(double), cudaMemcpyHostToDevice);
	dot<<<blocksPerGrid, 512>>>(x_d, y_d, z_d, n);
	cudaThreadSynchronize();
	cudaMemcpy(z, z_d, n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);
	double sum = 0;
	for (int i = 0; i < n; i++) {
		sum += z[i];
	}
	free(z);
	*result = sum;
	//printf("What is %f \n", *result);
	cudaDeviceReset();	
	return(0);
}
