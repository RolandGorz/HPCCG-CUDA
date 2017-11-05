#include "waxpby.hpp"
#include <stdio.h>
__global__ void vectorAdd(double *x, double *y, double *w, int numElements, const double alpha, const double beta) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;

    	if (i < numElements) {
		if (alpha==1.0) {
    			w[i] = x[i] + beta * y[i];
 		}
  		else if(beta==1.0) {
			w[i] = alpha * x[i] + y[i];
  		}
  		else {
    			w[i] = alpha * x[i] + beta * y[i];
  		}
	}
}

int waxpby (const int n, const double alpha, const double * const x, 
	    const double beta, const double * const y, 
		     double * const w)
{
	size_t size = n * sizeof(double);
	double *d_x = NULL;
	cudaMalloc((void **)&d_x, size);
	double *d_y = NULL;
	cudaMalloc((void **)&d_y, size);
	double *d_w = NULL;
	cudaMalloc((void **)&d_w, size);
	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
	int threadsPerBlock = 512;
	int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_w, n, alpha, beta);
	cudaThreadSynchronize();
	cudaMemcpy(w, d_w, size, cudaMemcpyDeviceToHost);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_w);
	cudaDeviceReset();
	return(0);
}
