#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

void cpuMultiply(double* a, double* b, double* c, int N);
void singleThreadMultiply(double* a, double* b, double* c, int N);
void multiThreadMultiply(double* a, double* b, double* c, int N);
void sharedMemoryMultiply(double* a, double* b, double* c, int N);

#ifdef __cplusplus
}
#endif

#endif
