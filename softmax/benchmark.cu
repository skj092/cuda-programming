#include <cuda_runtime.h>
#include <iostream>
#include "kernels/naive_softmax.cu"
#include "kernels/online_softmax.cu"

#define N 1024

void compare_results(float *h_output_naive, float *h_output_online, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << "Index " << i << ": Naive=" << h_output_naive[i]
                  << " | Online=" << h_output_online[i] << std::endl;
    }
}

int main() {
    float *h_input, *h_output_naive, *h_output_online;
    float *d_input, *d_output_naive, *d_output_online;

    h_input = new float[N];
    h_output_naive = new float[N];
    h_output_online = new float[N];

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output_naive, N * sizeof(float));
    cudaMalloc(&d_output_online, N * sizeof(float));

    for (int i = 0; i < N; i++)
        h_input[i] = static_cast<float>(i) / N;

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    run_naive_softmax(d_input, d_output_naive, N);
    run_online_softmax(d_input, d_output_online, N);

    cudaMemcpy(h_output_naive, d_output_naive, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_online, d_output_online, N * sizeof(float), cudaMemcpyDeviceToHost);

    compare_results(h_output_naive, h_output_online, N);

    delete[] h_input;
    delete[] h_output_naive;
    delete[] h_output_online;
    cudaFree(d_input);
    cudaFree(d_output_naive);
    cudaFree(d_output_online);

    return 0;
}
