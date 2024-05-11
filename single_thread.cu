#include "functions.h"
#include <stdio.h>

__global__ void matrixMultiplySingleThread(double *a, double *b, double *c, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx == 0) { 
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                if (a[i * N + k] != 0.0) {
                    for (int j = 0; j < N; j++) {
                        c[i * N + j] += a[i * N + k] * b[k * N + j];
                    }
                }
            }
        }
    }
}

void singleThreadMultiply(double* a, double* b, double* c, int N) {
    double *d_a, *d_b, *d_c;
    

    cudaMalloc((void**)&d_a, N*N*sizeof(double));
    cudaMalloc((void**)&d_b, N*N*sizeof(double));
    cudaMalloc((void**)&d_c, N*N*sizeof(double));


    cudaMemcpy(d_a, a, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, N*N*sizeof(double), cudaMemcpyHostToDevice);


    matrixMultiplySingleThread<<<1, 1>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
