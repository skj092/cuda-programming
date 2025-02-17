#include <cuda_runtime.h>
#include <iostream>

__global__ void naive_softmax(float *m, float *n, int N) {
    int index = threadIdx.x; /// (0, 1, 2)

    float ma = 0;
    for (int i = 0; i < N; i++){
    ma = fmax(m[index*N + i], ma);
    }

    // find denominator values
    float s = 0;
    for (int i = 0; i < N; i++){
        s += exp(m[index*N + i] - ma);
    }

    // devide each element by denominator
    for (int i = 0; i < N; i++){
        n[index * N + i] = exp(m[index * N + i] - ma) / s;
    }
}

void run_naive_softmax(float *d_input, float *d_output, int N) {
    naive_softmax<<<(N + 255) / 256, 256>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
}
