#include <cuda_runtime.h>
#include <iostream>
#include "kernels/naive_softmax.cu"
#include "kernels/online_softmax.cu"

#define N 1024
#define NUM_ITERATIONS 100  // Run multiple iterations for more stable timing

void compare_results(float *h_output_naive, float *h_output_online, int size) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = std::abs(h_output_naive[i] - h_output_online[i]);
        max_diff = std::max(max_diff, diff);
        if (i < 10) {  // Print first 10 elements for quick verification
            std::cout << "Index " << i << ": Naive=" << h_output_naive[i]
                      << " | Online=" << h_output_online[i] 
                      << " | Diff=" << diff << std::endl;
        }
    }
    std::cout << "Maximum difference between implementations: " << max_diff << std::endl;
}

void check_cuda_error(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " 
                  << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

int main() {
    float *h_input, *h_output_naive, *h_output_online;
    float *d_input, *d_output_naive, *d_output_online;
    
    // Allocate host memory
    h_input = new float[N];
    h_output_naive = new float[N];
    h_output_online = new float[N];

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    check_cuda_error(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda_error(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Allocate GPU memory
    cudaEventRecord(start);
    check_cuda_error(cudaMalloc(&d_input, N * sizeof(float)), "cudaMalloc d_input");
    check_cuda_error(cudaMalloc(&d_output_naive, N * sizeof(float)), "cudaMalloc d_output_naive");
    check_cuda_error(cudaMalloc(&d_output_online, N * sizeof(float)), "cudaMalloc d_output_online");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i) / N;
    }

    // Transfer data to GPU
    cudaEventRecord(start);
    check_cuda_error(cudaMemcpy(d_input, h_input, N * sizeof(float), 
                               cudaMemcpyHostToDevice), "cudaMemcpy HtoD");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to Device Transfer time: %f ms\n", ms);

    // Warmup runs
    run_naive_softmax(d_input, d_output_naive, N);
    run_online_softmax(d_input, d_output_online, N);
    cudaDeviceSynchronize();

    // Benchmark naive implementation
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        run_naive_softmax(d_input, d_output_naive, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Naive Softmax Kernel time (avg over %d runs): %f ms\n", 
           NUM_ITERATIONS, ms / NUM_ITERATIONS);

    // Benchmark online implementation
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        run_online_softmax(d_input, d_output_online, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Online Softmax Kernel time (avg over %d runs): %f ms\n", 
           NUM_ITERATIONS, ms / NUM_ITERATIONS);

    // Transfer results back to host
    cudaEventRecord(start);
    check_cuda_error(cudaMemcpy(h_output_naive, d_output_naive, N * sizeof(float),
                               cudaMemcpyDeviceToHost), "cudaMemcpy DtoH naive");
    check_cuda_error(cudaMemcpy(h_output_online, d_output_online, N * sizeof(float),
                               cudaMemcpyDeviceToHost), "cudaMemcpy DtoH online");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to Host transfer time: %f ms\n", ms);

    // Compare results
    compare_results(h_output_naive, h_output_online, N);

    // Cleanup
    delete[] h_input;
    delete[] h_output_naive;
    delete[] h_output_online;
    cudaFree(d_input);
    cudaFree(d_output_naive);
    cudaFree(d_output_online);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}