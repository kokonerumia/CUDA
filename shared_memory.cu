#include "functions.h"
#include <stdio.h>

#define TILE_WIDTH 16  

__global__ void matrixMultiplyShared(double *a, double *b, double *c, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    __shared__ double s_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ double s_b[TILE_WIDTH][TILE_WIDTH];

    double sum = 0.0;
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {

        if (m * TILE_WIDTH + tx < N && row < N)
            s_a[ty][tx] = a[row * N + m * TILE_WIDTH + tx];
        else
            s_a[ty][tx] = 0.0;

        if (m * TILE_WIDTH + ty < N && col < N)
            s_b[ty][tx] = b[(m * TILE_WIDTH + ty) * N + col];
        else
            s_b[ty][tx] = 0.0;

        __syncthreads();  


        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_a[ty][k] * s_b[k][tx];
        }

        __syncthreads(); 
    }

    if (row < N && col < N)
        c[row * N + col] = sum;
}

void sharedMemoryMultiply(double* a, double* b, double* c, int N) {
    double *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, N*N*sizeof(double));
    cudaMalloc((void**)&d_b, N*N*sizeof(double));
    cudaMalloc((void**)&d_c, N*N*sizeof(double));

    cudaMemcpy(d_a, a, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, N*N*sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    matrixMultiplyShared<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
