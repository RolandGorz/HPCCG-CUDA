#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cassert>
#include <string>
#include <cmath>
#include "HPC_sparsemv.hpp"

__global__ void mult(double * matrix, double * x, double * y, int * inds, int * nnz, int nrow) { //nnz is a summation

	int i = blockDim.x * blockIdx.x + threadIdx.x;
    	if(i < nrow) {
		double answer = 0;
		int start;
		int end;
		if (i == 0) {
			start = 0;
			end = nnz[0];
		} else {
			start = nnz[i - 1];
			end = nnz[i];
		}                          
    		for (int j = start; j < end; j++) {
      			answer += matrix[j] * x[inds[j]];
    		}
		y[i] = answer;
  	}
}

int HPC_sparsemv( HPC_Sparse_Matrix *A, const double * const x, double * const y)
{
	int nrow = A->local_nrow;
	int nnz = A->local_nnz;
	int ncol = A->local_ncol;
	double * d_matrix = NULL;
	cudaMalloc((void**)&d_matrix, nnz * sizeof(double));
	cudaMemcpy(d_matrix, A->list_of_vals, nnz * sizeof(double), cudaMemcpyHostToDevice);
	double * d_x = NULL;
	cudaMalloc((void**)&d_x, ncol * sizeof(double));
	cudaMemcpy(d_x, x, ncol * sizeof(double), cudaMemcpyHostToDevice);
	double * d_y = NULL;
	cudaMalloc((void**)&d_y, nrow * sizeof(double));
	int * d_inds = NULL;
	cudaMalloc((void**)&d_inds, nnz * sizeof(int));
	cudaMemcpy(d_inds, A->list_of_inds, nnz * sizeof(int), cudaMemcpyHostToDevice);
	int * d_nnz = NULL;
	cudaMalloc((void**)&d_nnz, nrow * sizeof(int));
	cudaMemcpy(d_nnz, A->nnz_in_row, nrow * sizeof(int), cudaMemcpyHostToDevice);
	int threadsPerBlock = 512;
	int blocksPerGrid =(nnz + threadsPerBlock - 1) / threadsPerBlock;
	mult<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_x, d_y, d_inds,d_nnz, nrow);
	cudaThreadSynchronize();
	cudaMemcpy(y, d_y, nrow * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceReset();
  	return(0);
}
