#include "functions.h"
#include <stdio.h>

void cpuMultiply(double* a, double* b, double* c, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            c[i * N + j] = 0.0;
        }
    }


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

#ifdef DEBUG
int main() {
    const int N = 4;
    double a[N*N], b[N*N], c[N*N];  


    double init_a[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    double init_b[] = {1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1};
    for (int i = 0; i < N*N; i++) {
        a[i] = init_a[i];
        b[i] = init_b[i];
        c[i] = 0;  
    }

    cpuMultiply(a, b, c, N);

    printf("Matrix C (Result):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5.2f ", c[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}
#endif
