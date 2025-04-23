#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

#define N 10000000  // Increase N to 10 million for measurable CPU time

__global__ void vectorAdd(int* d_a, int* d_b, int* d_c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

class VectorAddition {
public:
    void performVectorAddition() {
        int* a, * b, * c, * d;
        int* d_a, * d_b, * d_c;
        size_t size = N * sizeof(int);

        a = (int*)malloc(size);
        b = (int*)malloc(size);
        c = (int*)malloc(size);
        d = (int*)malloc(size);

        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            a[i] = rand() % 100;
            b[i] = rand() % 100;
        }

        // Measure CPU time using high_resolution_clock for better accuracy
        auto start_cpu = high_resolution_clock::now();
        // Run the addition multiple times to ensure measurable CPU time
        for (int j = 0; j < 10; j++) {  // Run the loop 10 times
            for (int i = 0; i < N; i++) {
                c[i] = a[i] + b[i];
            }
        }
        auto end_cpu = high_resolution_clock::now();
        auto cpu_duration = duration_cast<microseconds>(end_cpu - start_cpu).count();
        double cpu_time = cpu_duration / 1e6; // in seconds

        // CUDA memory allocation and kernel execution
        cudaMalloc((void**)&d_a, size);
        cudaMalloc((void**)&d_b, size);
        cudaMalloc((void**)&d_c, size);
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, N);
        cudaEventRecord(stop);

        cudaMemcpy(d, d_c, size, cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stop);
        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);

        // Compare results
        bool match = true;
        for (int i = 0; i < N; i++) {
            if (c[i] != d[i]) {
                match = false;
                break;
            }
        }

        // Output results
        printf("Vector Addition\n");
        printf("CPU Time: %.6f s\n", cpu_time);
        printf("GPU Time: %.6f ms\n", gpu_time);
        if (cpu_time > 0)
            printf("Speedup Factor: %.2f\n", (cpu_time) * 1000 / gpu_time);
        else
            printf("Speedup Factor: N/A (CPU time was too small)\n");
        printf("Arrays Match: %s\n", match ? "Yes" : "No");

        // Cleanup
        free(a); free(b); free(c); free(d);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

int main() {
    VectorAddition vectorAdder;
    vectorAdder.performVectorAddition();
    return 0;
}
