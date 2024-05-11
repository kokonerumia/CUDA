#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "functions.h"

using namespace std;
using namespace std::chrono;

extern "C" void cpuMultiply(double* a, double* b, double* c, int N);
extern "C" void singleThreadMultiply(double* a, double* b, double* c, int N);
extern "C" void multiThreadMultiply(double* a, double* b, double* c, int N);
extern "C" void sharedMemoryMultiply(double* a, double* b, double* c, int N);

#define SIZE 2048
#define SPARSITY 0.05  


void generateSparseMatrix(double *mat, int N, double sparsity) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double r = static_cast<double>(rand()) / RAND_MAX;
            mat[i * N + j] = r < sparsity ? (rand() % 10 + 1) : 0.0;
        }
    }
}

void generateDenseMatrix(double *mat, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i * N + j] = rand() % 10 + 1;
        }
    }
}

int main() {
    srand(time(NULL));

    double *a, *b, *c;
    a = new double[SIZE * SIZE];
    b = new double[SIZE * SIZE];
    c = new double[SIZE * SIZE];

    // 行列の生成
    generateSparseMatrix(a, SIZE, SPARSITY);
    generateDenseMatrix(b, SIZE);

    // タイマーの設定
    auto start = high_resolution_clock::now();
    auto end = high_resolution_clock::now();

    // CPU
    start = high_resolution_clock::now();
    cpuMultiply(a, b, c, SIZE);
    end = high_resolution_clock::now();
    cout << "CPU Time: " 
         << duration_cast<milliseconds>(end - start).count() 
         << " ms" << endl;

    // GPUで1スレッド
    start = high_resolution_clock::now();
    singleThreadMultiply(a, b, c, SIZE);
    end = high_resolution_clock::now();
    cout << "Single Thread GPU Time: "
         << duration_cast<milliseconds>(end - start).count()
         << " ms" << endl;

    // GPUでマルチスレッド
    start = high_resolution_clock::now();
    multiThreadMultiply(a, b, c, SIZE);
    end = high_resolution_clock::now();
    cout << "Multi Thread GPU Time: "
         << duration_cast<milliseconds>(end - start).count()
         << " ms" << endl;

    // GPU with 共有メモリ
    start = high_resolution_clock::now();
    sharedMemoryMultiply(a, b, c, SIZE);
    end = high_resolution_clock::now();
    cout << "Shared Memory GPU Time: "
         << duration_cast<milliseconds>(end - start).count()
         << " ms" << endl;


    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
