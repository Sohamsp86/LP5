#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 1000000   // Number of training samples
#define THREADS_PER_BLOCK 256

__global__ void compute_partial_sums(float* x, float* y, float* x_sum, float* y_sum, float* xy_sum, float* xx_sum, int n) {
    __shared__ float sx[THREADS_PER_BLOCK], sy[THREADS_PER_BLOCK], sxy[THREADS_PER_BLOCK], sxx[THREADS_PER_BLOCK];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    sx[tid] = sy[tid] = sxy[tid] = sxx[tid] = 0.0f;

    if (idx < n) {
        float xi = x[idx];
        float yi = y[idx];
        sx[tid] = xi;
        sy[tid] = yi;
        sxy[tid] = xi * yi;
        sxx[tid] = xi * xi;
    }

    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sx[tid] += sx[tid + stride];
            sy[tid] += sy[tid + stride];
            sxy[tid] += sxy[tid + stride];
            sxx[tid] += sxx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(x_sum, sx[0]);
        atomicAdd(y_sum, sy[0]);
        atomicAdd(xy_sum, sxy[0]);
        atomicAdd(xx_sum, sxx[0]);
    }
}

void linear_regression_serial(float* x, float* y, int n, float& w, float& b) {
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (int i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }
    w = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    b = (sum_y - w * sum_x) / n;
}

int main() {
    float *x, *y;
    x = new float[N];
    y = new float[N];

    // Seed and generate synthetic data: y = 2x + 5 + noise
    srand(time(0));
    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        y[i] = 2.0f * x[i] + 5.0f + static_cast<float>(rand()) / RAND_MAX;
    }

    // Serial Execution
    float w_serial, b_serial;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    linear_regression_serial(x, y, N, w_serial, b_serial);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double time_cpu = std::chrono::duration<double>(end_cpu - start_cpu).count();

    // GPU memory allocation
    float *x_d, *y_d;
    cudaMalloc(&x_d, N * sizeof(float));
    cudaMalloc(&y_d, N * sizeof(float));

    // Copy to GPU
    cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);

    float *x_sum_d, *y_sum_d, *xy_sum_d, *xx_sum_d;
    float h_x_sum = 0, h_y_sum = 0, h_xy_sum = 0, h_xx_sum = 0;

    cudaMalloc(&x_sum_d, sizeof(float));
    cudaMalloc(&y_sum_d, sizeof(float));
    cudaMalloc(&xy_sum_d, sizeof(float));
    cudaMalloc(&xx_sum_d, sizeof(float));

    cudaMemcpy(x_sum_d, &h_x_sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_sum_d, &h_y_sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xy_sum_d, &h_xy_sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xx_sum_d, &h_xx_sum, sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Parallel Execution
    auto start_gpu = std::chrono::high_resolution_clock::now();
    compute_partial_sums<<<blocks, THREADS_PER_BLOCK>>>(x_d, y_d, x_sum_d, y_sum_d, xy_sum_d, xx_sum_d, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double time_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();

    // Copy result back
    cudaMemcpy(&h_x_sum, x_sum_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_y_sum, y_sum_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_xy_sum, xy_sum_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_xx_sum, xx_sum_d, sizeof(float), cudaMemcpyDeviceToHost);

    float w_gpu = (N * h_xy_sum - h_x_sum * h_y_sum) / (N * h_xx_sum - h_x_sum * h_x_sum);
    float b_gpu = (h_y_sum - w_gpu * h_x_sum) / N;

    // Speedup
    float speedup = time_cpu / time_gpu;

    // Output
    std::cout << "===== Linear Regression Results =====" << std::endl;
    std::cout << "Serial:  y = " << w_serial << "x + " << b_serial << std::endl;
    std::cout << "Parallel (CUDA): y = " << w_gpu << "x + " << b_gpu << std::endl;
    std::cout << "Serial Time: " << time_cpu << " sec" << std::endl;
    std::cout << "Parallel Time: " << time_gpu << " sec" << std::endl;
    std::cout << "Speed-up Factor: " << speedup << "x" << std::endl;
    std::cout << "Sample Prediction x = 20: y = " << (w_gpu * 20 + b_gpu) << std::endl;

    // Cleanup
    cudaFree(x_d); cudaFree(y_d);
    cudaFree(x_sum_d); cudaFree(y_sum_d); cudaFree(xy_sum_d); cudaFree(xx_sum_d);
    delete[] x;
    delete[] y;

    return 0;
}

