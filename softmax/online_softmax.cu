#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include "utils.h"
#include "utils.cu"


__global__
void y(float *m, float *n, int N){
    int index = threadIdx.x; /// (0, 1, 2)

    float m_prev = -INFINITY;
    float d_prev = 0.0;

    for (int i = 0; i < N; i++){
        float x_i = m[index * N + i];
        float m_curr = max_val(x_i, m_prev);
        float d_curr = d_prev * exp(m_prev - m_curr) + exp(x_i - m_curr);

        m_prev = m_curr;
        d_prev = d_curr;
    }


    // devide each element by denominator
    for (int i = 0; i < N; i++){
        n[index * N + i] = exp(m[index * N + i] - m_prev) / d_prev;
    }
}

int main(){
    int N = 5;
    float *h_m, *h_n; // host variable
    float *d_m, *d_n; // device varables

    // Memory allocation in host
    int size = N * N * sizeof(float);
    h_m = (float *)malloc(size);
    h_n = (float *)malloc(size);

    for (int i = 0; i < N; i++){
        h_m[i] = random_normal_clamped(-10, 10);
    }

    // Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // Allocate device memory and copy data from host to device
    cudaEventRecord(start);
    cudaMalloc(&d_m, size);
    cudaMalloc(&d_n, size);
    cudaEventRecord(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    // copy data from host to device
    cudaEventRecord(start);
    cudaMemcpy(d_m, h_m, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    // launch kernel
    int blocksize = N;
    cudaEventRecord(start);
    mykernel<<<1, blocksize>>>(d_m, d_n, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> kernel execution time: %f ms\n", ms);

    // copy data from device to host
    cudaEventRecord(start);
    cudaMemcpy(h_n, d_n, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to Host transfer time: %f ms\n", ms);
    // verify
    // cpu benchmark
    struct timespec start_cpu, end_cpu;
    clock_gettime(CLOCK_MONOTONIC, &start_cpu);
    h_n  = softmax(h_m, h_n, N);
    clock_gettime(CLOCK_MONOTONIC, &end_cpu);
    double time_taken = (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e9;
    printf(">> CPU execution time: %f ms\n", time_taken);

    cudaFree(d_m);
    cudaFree(d_n);
    free(h_m);
    free(h_n);
    return 0;
}

