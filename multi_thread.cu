#include "functions.h"
#include <stdio.h>


__global__ void matrixMultiplyMultiThread(double *a, double *b, double *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}


void multiThreadMultiply(double* a, double* b, double* c, int N) {
    double *d_a, *d_b, *d_c;

    
    cudaMalloc((void**)&d_a, N*N*sizeof(double));
    cudaMalloc((void**)&d_b, N*N*sizeof(double));
    cudaMalloc((void**)&d_c, N*N*sizeof(double));

    
    cudaMemcpy(d_a, a, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, N*N*sizeof(double), cudaMemcpyHostToDevice);

    
    dim3 threadsPerBlock(16, 16);  
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);  

    
    matrixMultiplyMultiThread<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

   
    cudaMemcpy(c, d_c, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
